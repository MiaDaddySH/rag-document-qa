from app.config import settings
from app.embedding import embed_text
from app.llm_client import get_llm_client
from app.vector_store import search_similar_chunks

# 负责 RAG（Retrieval-Augmented Generation）的核心流程：
# 根据用户的问题，先把问题转换成向量，在向量数据库中检索相关的文本 chunks，
# 然后把这些 chunks 作为上下文，调用 LLM 生成最终的答案。
def build_context(retrieved_chunks: list[dict]) -> str:
    context_parts = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        text = chunk.get("text", "") or ""
        filename = chunk.get("filename", "unknown")
        chunk_index = chunk.get("chunk_index", -1)

        context_parts.append(
            f"[Source {i}] filename={filename}, chunk_index={chunk_index}\n{text}"
        )

    return "\n\n".join(context_parts)

# 构建最终的答案时，除了返回生成的文本，还会返回每个被检索到的 chunk 的来源信息，方便前端展示。
def build_sources(retrieved_chunks: list[dict]) -> list[dict]:
    sources = []

    for chunk in retrieved_chunks:
        sources.append(
            {
                "filename": chunk.get("filename"),
                "chunk_index": chunk.get("chunk_index"),
                "score": chunk.get("score"),
                "start_char": chunk.get("start_char"),
                "end_char": chunk.get("end_char"),
            }
        )

    return sources

# 调用 LLM 生成最终的答案。
# 在 system 提示中明确告诉模型只能根据提供的上下文来回答问题，如果上下文中没有答案，就说不知道。
def generate_answer(question: str, context: str) -> str:
    if not settings.azure_openai_deployment:
        raise ValueError("AZURE_OPENAI_DEPLOYMENT is not configured.")

    client = get_llm_client()

    response = client.chat.completions.create(
        model=settings.azure_openai_deployment,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant for document question answering. "
                    "Answer only based on the provided context. "
                    "If the answer is not in the context, say you don't know."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{question}",
            },
        ],
    )

    return response.choices[0].message.content or ""

# 这个函数是整个 RAG 流程的入口，接受用户的问题和可选的 top_k 参数，
# 以及可选的 filename 参数（如果用户只想查询特定文件）。
# 它会调用上面定义的函数来完成整个流程，并返回最终的答案和相关信息。
def answer_question(
    question: str,
    top_k: int = 3,
    filename: str | None = None,
) -> dict:
    if not question.strip():
        raise ValueError("Question must not be empty.")

    query_vector = embed_text(question)
    retrieved_chunks = search_similar_chunks(
        query_vector=query_vector,
        limit=top_k,
        filename=filename,
    )
    context = build_context(retrieved_chunks)
    answer = generate_answer(question=question, context=context)
    sources = build_sources(retrieved_chunks)

    return {
        "question": question,
        "filename": filename,
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": retrieved_chunks,
    }
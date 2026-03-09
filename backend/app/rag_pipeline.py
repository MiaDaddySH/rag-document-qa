from app.config import settings
from app.embedding import embed_text
from app.llm_client import get_llm_client
from app.vector_store import search_similar_chunks

# RAG pipeline 的核心函数，负责处理用户问题，检索相关 chunks，并生成答案。
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

# 根据问题和检索到的上下文，调用 LLM 生成答案。
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
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question:\n{question}"
                ),
            },
        ],
    )

    return response.choices[0].message.content or ""

# 主函数，接受用户问题，执行 RAG 流程，并返回答案和相关信息。
def answer_question(question: str, top_k: int = 3) -> dict:
    if not question.strip():
        raise ValueError("Question must not be empty.")

    query_vector = embed_text(question)
    retrieved_chunks = search_similar_chunks(query_vector=query_vector, limit=top_k)
    context = build_context(retrieved_chunks)

    answer = generate_answer(question=question, context=context)

    return {
        "question": question,
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
    }
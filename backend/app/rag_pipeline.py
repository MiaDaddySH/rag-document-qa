import re

from app.config import settings
from app.embedding import embed_text
from app.llm_client import get_llm_client, run_openai_with_retry
from app.vector_store import search_similar_chunks

# 负责 RAG（Retrieval-Augmented Generation）的核心流程：
# 根据用户的问题，先把问题转换成向量，在向量数据库中检索相关的文本 chunks，
# 然后把这些 chunks 作为上下文，调用 LLM 生成最终的答案。
def build_context(retrieved_chunks: list[dict], max_chars: int) -> str:
    context_parts = []
    total_chars = 0

    for i, chunk in enumerate(retrieved_chunks, start=1):
        text = chunk.get("text", "") or ""
        filename = chunk.get("filename", "unknown")
        chunk_index = chunk.get("chunk_index", -1)

        part = f"[Source {i}] filename={filename}, chunk_index={chunk_index}\n{text}"
        if max_chars > 0 and total_chars + len(part) > max_chars:
            remaining = max_chars - total_chars
            if remaining <= 0:
                break
            part = part[:remaining]
        context_parts.append(part)
        total_chars += len(part)
        if max_chars > 0 and total_chars >= max_chars:
            break

    return "\n\n".join(context_parts)

# 先按相似度阈值过滤，再保留顺序，保证上下文质量。
def filter_retrieved_chunks(retrieved_chunks: list[dict]) -> list[dict]:
    min_score = settings.rag_min_score
    if min_score <= 0:
        return retrieved_chunks

    filtered = []
    for chunk in retrieved_chunks:
        score = chunk.get("score")
        if score is None:
            continue
        if score >= min_score:
            filtered.append(chunk)

    return filtered


def normalize_vector_score(score: float | None) -> float:
    if score is None:
        return 0.0
    if score < 0:
        return max(0.0, (score + 1.0) / 2.0)
    return min(score, 1.0)


def tokenize_text(text: str) -> set[str]:
    tokens = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]", (text or "").lower())
    return {token for token in tokens if token}


def compute_keyword_overlap_score(question: str, text: str) -> float:
    question_tokens = tokenize_text(question)
    if not question_tokens:
        return 0.0
    text_tokens = tokenize_text(text)
    if not text_tokens:
        return 0.0
    overlap_count = len(question_tokens.intersection(text_tokens))
    return overlap_count / len(question_tokens)


def rerank_chunks(question: str, retrieved_chunks: list[dict], final_top_k: int) -> list[dict]:
    vector_weight = settings.rag_rerank_vector_weight
    keyword_weight = settings.rag_rerank_keyword_weight
    weight_sum = vector_weight + keyword_weight
    if weight_sum <= 0:
        vector_weight = 1.0
        keyword_weight = 0.0
    else:
        vector_weight = vector_weight / weight_sum
        keyword_weight = keyword_weight / weight_sum

    scored_chunks = []
    for chunk in retrieved_chunks:
        text = chunk.get("text", "") or ""
        vector_score = normalize_vector_score(chunk.get("score"))
        keyword_score = compute_keyword_overlap_score(question, text)
        rerank_score = (vector_score * vector_weight) + (keyword_score * keyword_weight)
        scored_chunks.append(
            {
                **chunk,
                "vector_score": vector_score,
                "keyword_score": keyword_score,
                "rerank_score": rerank_score,
            }
        )

    reranked = sorted(
        scored_chunks,
        key=lambda item: (
            item.get("rerank_score", 0.0),
            item.get("score", 0.0) or 0.0,
        ),
        reverse=True,
    )
    return reranked[:final_top_k]

# 构建最终的答案时，除了返回生成的文本，还会返回每个被检索到的 chunk 的来源信息，方便前端展示。
def build_sources(retrieved_chunks: list[dict]) -> list[dict]:
    sources = []

    for chunk in retrieved_chunks:
        sources.append(
            {
                "filename": chunk.get("filename"),
                "chunk_index": chunk.get("chunk_index"),
                "score": chunk.get("score"),
                "rerank_score": chunk.get("rerank_score"),
                "keyword_score": chunk.get("keyword_score"),
                "vector_score": chunk.get("vector_score"),
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

    response = run_openai_with_retry(
        operation=lambda: client.chat.completions.create(
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
        ),
        operation_name="generate_chat_completion",
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
    final_top_k = max(1, top_k)
    retrieval_limit = max(
        final_top_k,
        final_top_k * settings.rag_retrieval_multiplier,
    )

    query_vector = embed_text(question)
    retrieved_chunks = search_similar_chunks(
        query_vector=query_vector,
        limit=retrieval_limit,
        filename=filename,
    )
    filtered_chunks = filter_retrieved_chunks(retrieved_chunks)
    reranked_chunks = rerank_chunks(question=question, retrieved_chunks=filtered_chunks, final_top_k=final_top_k)
    context = build_context(reranked_chunks, settings.rag_context_max_chars)
    if not context.strip():
        return {
            "question": question,
            "filename": filename,
            "answer": "没有找到相关内容，我不知道。",
            "sources": [],
            "retrieved_chunks": [],
        }
    answer = generate_answer(question=question, context=context)
    sources = build_sources(reranked_chunks)

    return {
        "question": question,
        "filename": filename,
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": reranked_chunks,
    }

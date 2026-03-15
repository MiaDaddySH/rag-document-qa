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
    normalized = set()
    for token in tokens:
        if not token:
            continue
        if token.isdigit():
            continue
        if len(token) == 1 and token.isascii() and token.isalpha():
            continue
        normalized.add(token)
    return normalized


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


def compute_answer_confidence(retrieved_chunks: list[dict], context: str) -> float:
    if not retrieved_chunks:
        return 0.0
    top_score = float(retrieved_chunks[0].get("rerank_score") or 0.0)
    top_scores = [float(chunk.get("rerank_score") or 0.0) for chunk in retrieved_chunks[:3]]
    avg_top_score = sum(top_scores) / max(1, len(top_scores))
    context_target_chars = max(1, min(settings.rag_context_max_chars, 2000))
    context_coverage = min(1.0, len(context) / context_target_chars)
    confidence = (0.55 * top_score) + (0.35 * avg_top_score) + (0.10 * context_coverage)
    return max(0.0, min(1.0, confidence))


def is_model_uncertain(answer: str) -> bool:
    lowered = (answer or "").lower()
    markers = [
        "i don't know",
        "do not know",
        "unknown",
        "不确定",
        "我不知道",
        "没有找到相关内容",
    ]
    return any(marker in lowered for marker in markers)


def split_answer_sentences(answer: str) -> list[str]:
    normalized = (answer or "").replace("\n", " ").strip()
    if not normalized:
        return []
    raw_parts = re.split(r"(?<=[。！？.!?])\s+", normalized)
    sentences = []
    for part in raw_parts:
        sentence = part.strip()
        if not sentence:
            continue
        sentence_tokens = tokenize_text(sentence)
        if len(sentence_tokens) < 2:
            continue
        sentences.append(sentence)
    return sentences


def extract_evidence_excerpt(text: str, claim_tokens: set[str], max_chars: int = 220) -> str:
    source_text = text or ""
    if not source_text:
        return ""
    lowered = source_text.lower()
    matched_token = next((token for token in claim_tokens if token in lowered and len(token) > 1), None)
    if not matched_token:
        return source_text[:max_chars]
    start_index = max(0, lowered.find(matched_token) - 60)
    end_index = min(len(source_text), start_index + max_chars)
    return source_text[start_index:end_index]


def build_evidence_citations(answer: str, retrieved_chunks: list[dict]) -> list[dict]:
    citations = []
    if not answer or not retrieved_chunks:
        return citations
    sentences = split_answer_sentences(answer)
    if not sentences:
        return citations
    max_citations = max(1, settings.rag_max_citations)
    min_overlap = settings.rag_citation_min_overlap
    for sentence in sentences:
        claim_tokens = tokenize_text(sentence)
        if not claim_tokens:
            continue
        best_match = None
        best_overlap = 0.0
        for chunk in retrieved_chunks:
            chunk_tokens = tokenize_text(chunk.get("text", "") or "")
            if not chunk_tokens:
                continue
            overlap = len(claim_tokens.intersection(chunk_tokens)) / len(claim_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = chunk
        if not best_match or best_overlap < min_overlap:
            continue
        citations.append(
            {
                "claim": sentence,
                "filename": best_match.get("filename"),
                "chunk_index": best_match.get("chunk_index"),
                "overlap_score": round(best_overlap, 4),
                "evidence_excerpt": extract_evidence_excerpt(
                    text=best_match.get("text", "") or "",
                    claim_tokens=claim_tokens,
                ),
            }
        )
        if len(citations) >= max_citations:
            break
    if citations:
        return citations
    fallback_claim = sentences[0][:200] if sentences else (answer or "")[:200]
    for chunk in retrieved_chunks[:max_citations]:
        citations.append(
            {
                "claim": fallback_claim,
                "filename": chunk.get("filename"),
                "chunk_index": chunk.get("chunk_index"),
                "overlap_score": round(float(chunk.get("rerank_score") or 0.0), 4),
                "evidence_excerpt": extract_evidence_excerpt(
                    text=chunk.get("text", "") or "",
                    claim_tokens=tokenize_text(fallback_claim),
                ),
            }
        )
    return citations


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
    answer_confidence = compute_answer_confidence(reranked_chunks, context)
    top_rerank_score = float(reranked_chunks[0].get("rerank_score") or 0.0) if reranked_chunks else 0.0
    if not context.strip():
        return {
            "question": question,
            "filename": filename,
            "answer": "没有找到相关内容，我不知道。",
            "sources": build_sources(reranked_chunks),
            "retrieved_chunks": reranked_chunks,
            "citations": [],
            "is_refused": True,
            "refusal_reason": "no_relevant_context",
            "answer_confidence": 0.0,
        }
    refusal_reason = None
    if top_rerank_score < settings.rag_min_top_rerank_score:
        refusal_reason = "weak_retrieval_signal"
    elif answer_confidence < settings.rag_confidence_threshold:
        refusal_reason = "low_answer_confidence"
    if refusal_reason:
        return {
            "question": question,
            "filename": filename,
            "answer": "当前检索证据不足，我不确定。请提供更具体的问题或更多资料。",
            "sources": build_sources(reranked_chunks),
            "retrieved_chunks": reranked_chunks,
            "citations": [],
            "is_refused": True,
            "refusal_reason": refusal_reason,
            "answer_confidence": answer_confidence,
        }
    answer = generate_answer(question=question, context=context)
    sources = build_sources(reranked_chunks)
    if is_model_uncertain(answer) and answer_confidence >= settings.rag_confidence_threshold:
        answer = generate_answer(question=question, context=context)
    citations = build_evidence_citations(answer=answer, retrieved_chunks=reranked_chunks)
    if is_model_uncertain(answer):
        return {
            "question": question,
            "filename": filename,
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": reranked_chunks,
            "citations": citations,
            "is_refused": True,
            "refusal_reason": "model_uncertain",
            "answer_confidence": answer_confidence,
        }

    return {
        "question": question,
        "filename": filename,
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": reranked_chunks,
        "citations": citations,
        "is_refused": False,
        "refusal_reason": None,
        "answer_confidence": answer_confidence,
    }

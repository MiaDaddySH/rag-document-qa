import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

@dataclass
class EvalCase:
    case_id: str
    question: str
    filename: str | None
    top_k: int
    expected_keywords: list[str]
    min_source_count: int
    expected_source_filenames: list[str]


def load_cases(cases_file: Path) -> list[EvalCase]:
    raw = json.loads(cases_file.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Cases file must be a JSON array.")

    cases: list[EvalCase] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Case #{index} must be a JSON object.")
        case_id = str(item.get("id") or f"case_{index}")
        question = str(item.get("question") or "").strip()
        if not question:
            raise ValueError(f"Case '{case_id}' question is empty.")

        filename = item.get("filename")
        if filename is not None:
            filename = str(filename).strip() or None

        top_k = int(item.get("top_k", 3))
        expected_keywords = [str(keyword).strip() for keyword in item.get("expected_keywords", []) if str(keyword).strip()]
        min_source_count = int(item.get("min_source_count", 1))
        expected_source_filenames = [
            str(name).strip() for name in item.get("expected_source_filenames", []) if str(name).strip()
        ]

        cases.append(
            EvalCase(
                case_id=case_id,
                question=question,
                filename=filename,
                top_k=top_k,
                expected_keywords=expected_keywords,
                min_source_count=min_source_count,
                expected_source_filenames=expected_source_filenames,
            )
        )
    return cases


def keyword_hit_ratio(answer_text: str, expected_keywords: list[str]) -> float | None:
    if not expected_keywords:
        return None
    lowered_answer = answer_text.lower()
    hit_count = sum(1 for keyword in expected_keywords if keyword.lower() in lowered_answer)
    return hit_count / len(expected_keywords)


def evaluate_case(case: EvalCase, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return {
            "id": case.case_id,
            "executed": False,
            "passed": True,
            "reason": "dry_run",
        }

    from app.rag_pipeline import answer_question

    result = answer_question(
        question=case.question,
        top_k=case.top_k,
        filename=case.filename,
    )
    answer_text = str(result.get("answer") or "")
    sources = result.get("sources") or []

    has_answer = bool(answer_text.strip())
    source_count_ok = len(sources) >= case.min_source_count
    keyword_ratio = keyword_hit_ratio(answer_text, case.expected_keywords)
    keyword_ok = True if keyword_ratio is None else keyword_ratio > 0

    source_filename_ok = True
    if case.expected_source_filenames:
        actual_filenames = {str(source.get("filename") or "") for source in sources}
        source_filename_ok = any(expected in actual_filenames for expected in case.expected_source_filenames)

    passed = has_answer and source_count_ok and keyword_ok and source_filename_ok
    return {
        "id": case.case_id,
        "executed": True,
        "passed": passed,
        "question": case.question,
        "filename": case.filename,
        "has_answer": has_answer,
        "source_count": len(sources),
        "min_source_count": case.min_source_count,
        "source_count_ok": source_count_ok,
        "keyword_hit_ratio": keyword_ratio,
        "keyword_ok": keyword_ok,
        "source_filename_ok": source_filename_ok,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    executed_results = [item for item in results if item.get("executed")]
    passed_results = [item for item in executed_results if item.get("passed")]
    ratios = [item["keyword_hit_ratio"] for item in executed_results if item.get("keyword_hit_ratio") is not None]
    avg_keyword_ratio = (sum(ratios) / len(ratios)) if ratios else None

    pass_rate = 1.0
    if executed_results:
        pass_rate = len(passed_results) / len(executed_results)

    return {
        "total_cases": len(results),
        "executed_cases": len(executed_results),
        "passed_cases": len(passed_results),
        "pass_rate": pass_rate,
        "avg_keyword_hit_ratio": avg_keyword_ratio,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline RAG evaluation.")
    parser.add_argument(
        "--cases-file",
        default="app/eval_cases.sample.json",
        help="Path to evaluation cases JSON file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate case schema only, skip live inference calls.",
    )
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=0.6,
        help="Minimum required pass rate for executed cases.",
    )
    parser.add_argument(
        "--min-avg-keyword-hit",
        type=float,
        default=0.3,
        help="Minimum required average keyword hit ratio.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases_file = Path(args.cases_file)
    if not cases_file.exists():
        print(json.dumps({"error": f"Cases file not found: {cases_file}"}, ensure_ascii=False))
        return 1

    try:
        cases = load_cases(cases_file=cases_file)
    except Exception as error:
        print(json.dumps({"error": f"Failed to load cases: {error}"}, ensure_ascii=False))
        return 1

    results = [evaluate_case(case, dry_run=args.dry_run) for case in cases]
    summary = summarize(results)
    report = {
        "summary": summary,
        "results": results,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.dry_run:
        return 0

    pass_rate = float(summary["pass_rate"])
    avg_keyword_hit = summary["avg_keyword_hit_ratio"]
    if pass_rate < args.min_pass_rate:
        return 1
    if avg_keyword_hit is not None and avg_keyword_hit < args.min_avg_keyword_hit:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

import argparse
import json
import os
import subprocess
import sys
from itertools import product


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def run_evaluation(cases_file: str, env_overrides: dict[str, str]) -> dict:
    env = os.environ.copy()
    env.update(env_overrides)
    command = [
        sys.executable,
        "-m",
        "app.evaluation",
        "--cases-file",
        cases_file,
        "--min-pass-rate",
        "0.0",
        "--min-avg-keyword-hit",
        "0.0",
    ]
    result = subprocess.run(command, capture_output=True, text=True, env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "evaluation failed")
    return json.loads(result.stdout)


def build_trials(
    multipliers: list[int],
    vector_weights: list[float],
    min_scores: list[float],
) -> list[dict]:
    trials = []
    for multiplier, vector_weight, min_score in product(multipliers, vector_weights, min_scores):
        vector_weight = round(vector_weight, 3)
        keyword_weight = round(1.0 - vector_weight, 3)
        trial = {
            "RAG_RETRIEVAL_MULTIPLIER": str(multiplier),
            "RAG_RERANK_VECTOR_WEIGHT": str(vector_weight),
            "RAG_RERANK_KEYWORD_WEIGHT": str(keyword_weight),
            "RAG_MIN_SCORE": str(min_score),
        }
        trials.append(trial)
    return trials


def score_trial(report: dict) -> tuple[float, float]:
    summary = report.get("summary", {})
    return (
        float(summary.get("pass_rate", 0.0)),
        float(summary.get("avg_keyword_hit_ratio") or 0.0),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid search for RAG retrieval/rerank parameters.")
    parser.add_argument("--cases-file", default="app/eval_cases.sample.json")
    parser.add_argument("--multipliers", default="2,3,4")
    parser.add_argument("--vector-weights", default="0.6,0.7,0.8")
    parser.add_argument("--min-scores", default="0.15,0.2,0.25")
    parser.add_argument("--top-n", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trials = build_trials(
        multipliers=parse_int_list(args.multipliers),
        vector_weights=parse_float_list(args.vector_weights),
        min_scores=parse_float_list(args.min_scores),
    )
    results = []
    for trial in trials:
        try:
            report = run_evaluation(cases_file=args.cases_file, env_overrides=trial)
            pass_rate, avg_keyword_hit_ratio = score_trial(report)
            results.append(
                {
                    "params": trial,
                    "pass_rate": pass_rate,
                    "avg_keyword_hit_ratio": avg_keyword_hit_ratio,
                    "summary": report.get("summary", {}),
                }
            )
        except Exception as error:
            results.append(
                {
                    "params": trial,
                    "error": str(error),
                    "pass_rate": 0.0,
                    "avg_keyword_hit_ratio": 0.0,
                }
            )

    ranked = sorted(
        results,
        key=lambda item: (item.get("pass_rate", 0.0), item.get("avg_keyword_hit_ratio", 0.0)),
        reverse=True,
    )
    payload = {
        "cases_file": args.cases_file,
        "trial_count": len(results),
        "top_results": ranked[: max(1, args.top_n)],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

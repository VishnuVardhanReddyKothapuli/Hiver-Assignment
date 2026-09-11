import argparse
import json
from .agent import Agent
from .common import ARTIFACTS, DATA, read_jsonl

def main():
    parser = argparse.ArgumentParser(description="Hiver support lab: reproducible diagnostics and human evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--mode", choices=("diagnostic", "gold"), default="diagnostic")
    evaluate.add_argument("--threshold", type=float, default=.55)
    predict = sub.add_parser("predict")
    predict.add_argument("text")
    predict.add_argument("--system", choices=("trivial", "simple", "retrieval"), default="retrieval")
    predict.add_argument("--followup", action="store_true")
    predict.add_argument("--model", help="Optional explicit OpenAI model for an LLM draft; requires OPENAI_API_KEY")
    review = sub.add_parser("review")
    review.add_argument("--kind", choices=("intent", "reply"), default="intent")
    review.add_argument("--partition", choices=("dev", "test", "calibration", "audit"))
    prep = sub.add_parser("prepare-replies")
    prep.add_argument("--predictions", default=str(ARTIFACTS/"diagnostic_predictions.jsonl"))
    judge = sub.add_parser("judge")
    judge.add_argument("--model", required=True)
    sub.add_parser("agreement")
    sub.add_parser("status")
    args = parser.parse_args()
    try:
        if args.command == "evaluate":
            from .evaluation import run
            run(args.mode, args.threshold)
        elif args.command == "predict":
            prediction = Agent(read_jsonl(DATA/"train.jsonl")).predict(args.text, args.system, args.followup)
            if args.model:
                from .llm import draft_with_llm
                result = draft_with_llm(args.model, args.text, prediction)
            else:
                result = prediction.as_dict()
            print(json.dumps(result, indent=2, ensure_ascii=True))
        elif args.command == "review":
            from .review import annotate
            annotate(args.kind, args.partition)
        elif args.command == "prepare-replies":
            from .evaluation import make_review
            make_review(args.predictions)
        elif args.command == "judge":
            from .judge import run_judge
            run_judge(args.model)
        elif args.command == "agreement":
            from .judge import agreement
            agreement()
        elif args.command == "status":
            from .common import read_csv
            for filename in ("human_labels.csv", "reply_ratings.csv"):
                path = DATA/filename
                rows = read_csv(path) if path.exists() else []
                done = sum(r.get("label_source") == "human" for r in rows)
                print(f"{filename}: {done}/{len(rows)} marked human-reviewed (gold evaluation validates all fields)")
            print("LLM judge:", "recorded; run agreement to validate" if (ARTIFACTS/"judge_scores.jsonl").exists() else "not run")
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        parser.exit(2, f"Error: {error}\n")

if __name__ == "__main__":
    main()


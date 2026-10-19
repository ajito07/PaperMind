"""Evaluate retrieval (recall@k, MRR) per mode and, with --judge, answer quality.

Usage:  python eval/run_eval.py [--k 8] [--judge]
Needs papers from golden.json already ingested (see README).
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import agent, config, db, llm, prompts, retrieval  # noqa: E402

GOLDEN = Path(__file__).parent / "golden.json"


def ranked_ids(conn, query: str, mode: str, k: int) -> list[int]:
    if mode == "vector":
        return [cid for cid, _ in retrieval.vector_search(conn, query, k)]
    if mode == "bm25":
        return retrieval.bm25_search(conn, query, k)
    return [c["chunk_id"] for c in retrieval.hybrid_search(conn, query, k=k)["chunks"]]


def is_hit(conn, chunk_id: int, case: dict) -> bool:
    r = conn.execute(
        "SELECT c.text, p.title FROM chunks c JOIN papers p ON p.id=c.paper_id WHERE c.id=?",
        (chunk_id,)).fetchone()
    return (r and case["paper"].lower() in r["title"].lower()
            and any(s.lower() in r["text"].lower() for s in case["expect"]))


def eval_retrieval(conn, cases: list[dict], k: int) -> dict:
    modes = ("bm25", "vector", "hybrid")
    out = {}
    for mode in modes:
        recall1 = recall3 = recallk = mrr = 0.0
        for case in cases:
            ids = ranked_ids(conn, case["question"], mode, k)
            rank = next((i + 1 for i, cid in enumerate(ids) if is_hit(conn, cid, case)), None)
            if rank:
                recall1 += rank == 1
                recall3 += rank <= 3
                recallk += 1
                mrr += 1.0 / rank
        n = len(cases)
        out[mode] = {"recall@1": round(recall1 / n, 3), "recall@3": round(recall3 / n, 3),
                     f"recall@{k}": round(recallk / n, 3), "mrr": round(mrr / n, 3)}
    return out


def rag_answer(conn, question: str) -> str:
    res = retrieval.hybrid_search(conn, question)
    if not res["confident"]:
        return prompts.NOT_FOUND
    msgs = [{"role": "system",
             "content": prompts.ANSWER_SYSTEM.format(sources=agent.format_sources(res["chunks"]))},
            {"role": "user", "content": question}]
    return llm.chat(msgs)["content"]


def eval_answers(conn, cases: list[dict]) -> dict:
    faith = qual = judged = 0
    for case in cases:
        answer = rag_answer(conn, case["question"])
        msg = llm.chat([{"role": "user", "content": prompts.JUDGE.format(
            question=case["question"], reference=case["reference"], answer=answer)}],
            temperature=0)
        try:
            scores = json.loads(re.search(r"\{.*\}", msg["content"], re.S).group())
            faith += int(scores["faithfulness"])
            qual += int(scores["quality"])
            judged += 1
            print(f"  judged: {case['question'][:60]!r} → "
                  f"faith={scores['faithfulness']} quality={scores['quality']}")
        except (AttributeError, KeyError, ValueError):
            print(f"  judge output unparseable for {case['question'][:60]!r}, skipped")
    return {"faithfulness": round(faith / judged, 2) if judged else None,
            "quality": round(qual / judged, 2) if judged else None, "n_judged": judged}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=config.TOP_K_FUSED)
    ap.add_argument("--judge", action="store_true", help="also LLM-judge end-to-end answers (slow)")
    args = ap.parse_args()

    cases = json.loads(GOLDEN.read_text())
    conn = db.connect()
    n_ready = conn.execute("SELECT count(*) FROM papers WHERE status='ready'").fetchone()[0]
    if not n_ready:
        sys.exit("No ingested papers. Upload the golden-set papers first (see README).")

    print(f"# Eval: {len(cases)} questions, {n_ready} papers, k={args.k}\n")
    metrics = {"retrieval": eval_retrieval(conn, cases, args.k)}

    print(f"| mode | recall@1 | recall@3 | recall@{args.k} | MRR |")
    print("|---|---|---|---|---|")
    for mode, m in metrics["retrieval"].items():
        print(f"| {mode} | {m['recall@1']} | {m['recall@3']} | {m[f'recall@{args.k}']} | {m['mrr']} |")

    if args.judge:
        print("\nJudging end-to-end answers (this loads the 14B model)...")
        metrics["answers"] = eval_answers(conn, cases)
        print(f"\nfaithfulness: {metrics['answers']['faithfulness']}/5  "
              f"quality: {metrics['answers']['quality']}/5")

    conn.execute("INSERT INTO eval_runs(config, metrics) VALUES (?,?)",
                 (json.dumps({"k": args.k, "judge": args.judge, "n_cases": len(cases)}),
                  json.dumps(metrics)))
    conn.commit()
    conn.close()


if __name__ == "__main__":
    main()

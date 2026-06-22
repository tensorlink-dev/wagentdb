"""Example: an agent reviewing & summarizing past experiments.

Run `python examples/log_run.py` first to create some data, then:
    python examples/review_agent.py
"""

import wagentdb


def main() -> None:
    db = wagentdb.connect(project="demo")

    runs = db.runs(limit=20)
    print(f"Found {len(runs)} runs in 'demo':")
    for r in runs:
        best = (r.summary or {}).get("best_acc")
        print(f"  - {r.id}  {r.name:<16}  status={r.status:<9}  best_acc={best}")

    if not runs:
        print("No runs yet. Run examples/log_run.py first.")
        return

    # Pull a one-call digest of the most recent run (what an LLM agent would read).
    report = db.report(runs[0].id)
    print("\nReport for", report.run.name)
    print("  config:        ", report.run.config)
    print("  metric keys:   ", report.metric_keys)
    print("  latest metrics:", report.latest_metrics)
    print("  artifacts:     ", [(a.name, a.size_bytes) for a in report.artifacts])
    print("  log tail:")
    for line in report.log_tail:
        print("     ", line["timestamp"], line["message"])

    # An agent writes back its review so the next agent can read it.
    loss = report.latest_metrics.get("loss")
    verdict = "converged nicely" if loss and loss < 0.2 else "needs more steps"
    db.add_summary(
        report.run.id,
        f"Run {report.run.name}: final loss {loss}. Verdict: {verdict}.",
        kind="review",
        created_by="review-agent",
    )
    print("\nWrote review:", db.summaries(report.run.id)[-1].content)


if __name__ == "__main__":
    main()

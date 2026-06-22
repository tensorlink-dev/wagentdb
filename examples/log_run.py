"""Example: an agent logging a training run to wagentdb (embedded mode).

Run with:  python examples/log_run.py
This uses the local backend (a ./.wagentdb directory) so no R2 needed.
"""

import math
import random

import wagentdb


def main() -> None:
    run = wagentdb.init(
        project="demo",
        name="sgd-lr3e4",
        experiment="lr-sweep",
        config={"lr": 3e-4, "optimizer": "adamw", "batch_size": 64},
        tags=["demo", "baseline"],
        created_by="agent-001",
    )
    print("started", run)

    loss = 2.0
    for step in range(200):
        loss = max(0.05, loss * 0.97 + random.uniform(-0.01, 0.01))
        acc = 1.0 - math.exp(-step / 60)
        run.log({"loss": loss, "val/acc": acc}, step=step)
        if step % 50 == 0:
            run.log_text(f"step {step}: loss={loss:.3f} acc={acc:.3f}")

    # Store an artifact (here just bytes; in practice a checkpoint file path).
    run.log_artifact(b"PRETEND-MODEL-WEIGHTS", name="model.bin", type="model")
    run.summary["best_acc"] = acc
    run.finish()
    print("finished", run.id)


if __name__ == "__main__":
    main()

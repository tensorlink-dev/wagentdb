---
name: wagentdb
description: >-
  Log, track, link, and review ML training experiments with wagentdb (an
  open-source wandb for agents, backed by Cloudflare R2 + SQLite). Use when an
  agent runs a training job and should record config/metrics/artifacts, when
  linking related runs (forks, fine-tunes, resumes), or when reviewing /
  comparing / summarizing past experiments. Triggers: "log this run", "track
  the experiment", "what were the results of...", "compare runs", "summarize the
  experiment", anything involving wagentdb or experiment tracking.
---

# Using wagentdb

wagentdb is a tiny "wandb for agents". Two things you'll do: **log** a training
run, and later **review/summarize** runs. Everything is reachable from Python
(`import wagentdb`), an HTTP API, or the `wagentdb` CLI.

## Setup / where data goes

- `pip install -e ".[all]"` (or `pip install wagentdb[all]`).
- Backend is chosen by env. **Local** (default, no creds) writes to `./.wagentdb`.
  **R2** stores blobs + the SQLite index in a Cloudflare R2 bucket:
  ```bash
  export WAGENTDB_BACKEND=r2
  export WAGENTDB_R2_BUCKET=my-experiments
  export WAGENTDB_R2_ACCOUNT_ID=...          # derives the endpoint
  export WAGENTDB_R2_ACCESS_KEY_ID=...
  export WAGENTDB_R2_SECRET_ACCESS_KEY=...
  ```
- The same Python code runs in **embedded** mode (writes straight to storage) or
  **http** mode (talks to a server) — just add `url="https://wagentdb..."` to
  `init()`/`connect()`. Use http mode when multiple agents log concurrently
  (writes get serialized in one process); embedded is fine for a single agent.

## Logging a run (wandb-style)

```python
import wagentdb

run = wagentdb.init(
    project="ssm-ds",                 # auto-created if new
    name="designB-lr3e4",             # optional; defaults to the run id
    experiment="lr-sweep",            # optional grouping; auto-created
    config={"lr": 3e-4, "design": "B"},
    tags=["sweep"],
    created_by="agent-007",           # which agent owns this run
)
# init() auto-records env (python/torch/CUDA/GPU/lib versions) + git commit.

for step in range(1000):
    run.log({"loss": loss, "val/acc": acc}, step=step)   # numeric dict
run.log_text("lr warmup done", step=100)                 # free-form log line

run.log_artifact("model.pt", type="model")               # file -> streamed
run.log_artifact("checkpoints/ckpt-1000", type="checkpoint")  # dir -> tar.gz
run.summary["best_acc"] = 0.91                            # quick-scan summary
run.finish()                                              # or finish("failed")
```

Notes:
- `run.log(d, step=)` flattens nested dicts (`{"eval":{"loss":x}}` → `eval/loss`)
  and drops non-numeric / NaN values, so you can pass a framework's raw log dict.
- `log_artifact` accepts a **file path** (streamed, multipart on R2), a
  **directory** (archived to `.tar.gz`), or raw **bytes**.
- The run `summary` auto-tracks the latest value of each metric.

### HuggingFace transformers

Don't hand-roll the loop — use the callback:

```python
from wagentdb.integrations.transformers import WagentDBCallback
trainer = Trainer(..., callbacks=[
    WagentDBCallback(project="llm-sft", name="qwen-lora",
                     log_checkpoints=True, log_model=True)])
trainer.train()
```

## Linking experiments (the graph)

Record lineage so later agents can trace how runs relate:

```python
run.link(parent_run, relation="forked_from")       # or pass a run id string
```

Relations: `forked_from`, `finetuned_from`, `resumed_from`, `uses_artifact`,
`part_of`, `compared_with`, `produces`. (`part_of` to an experiment is added
automatically when you pass `experiment=`.)

You can also start a run already linked: `wagentdb.init(..., fork_from=parent_id)`.

## Reviewing / summarizing (the agent read path)

```python
db = wagentdb.connect(project="ssm-ds")     # add url=... for http mode

db.runs(status="finished", tag="sweep")     # filter runs
db.run(run_id)                              # one run
db.history(run_id, "loss")                  # full time-series for a metric
db.lineage(run_id)                          # {"in": [...], "out": [...]} edges
db.graph()                                  # nodes + edges for the project

report = db.report(run_id)                  # ONE-CALL digest — start here
# report has: run, project, experiment, metric_keys, latest_metrics,
#             artifacts, lineage, summaries, log_tail
```

When you've reviewed something, **write your conclusion back** so the next agent
benefits:

```python
db.add_summary(run_id,
               "Design B overfits after step 600; lr 3e-4 is too high.",
               kind="review", created_by="review-agent")
db.summaries(run_id)                        # read prior reviews/insights
```

`kind` ∈ `summary | review | insight | hypothesis`. Summaries can attach to a
run, experiment, or project (`subject_type=`).

## CLI

```bash
wagentdb serve [--port 8000]      # HTTP API + OpenAPI docs at /docs
wagentdb runs  [--project P]      # list runs
wagentdb report <run_id>          # JSON digest
wagentdb graph [--project P]      # JSON nodes+edges
```

## Guidance for agents

- **Always `finish()`** a run (use `finish("failed")` on error) so status and
  timestamps are correct.
- Put hyperparameters in `config`, scalar curves through `log(...)`, and final
  headline numbers in `summary` — that's what `report()` surfaces for scanning.
- Before starting related work, `db.runs(...)` + `db.report(...)` to see what's
  been tried, and read existing `db.summaries(...)` so you don't repeat it.
- After analyzing results, leave a `db.add_summary(...)` review — this is the
  shared memory across agents.
- The full API surface is in `wagentdb/store.py`; the schema is in
  `wagentdb/schema.sql`.

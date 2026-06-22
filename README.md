# wagentdb

**An open-source, agent-native experiment tracker — a "wandb for AI agents",
backed by Cloudflare R2 + SQLite.**

Agents that run training experiments need somewhere to log what they did, link
related experiments together, stash checkpoints, and — crucially — *read it all
back later* to review and summarize. `wagentdb` is that store, with three design
goals:

1. **Cheap, serverless storage.** Everything lives in a single Cloudflare R2
   bucket: large blobs (checkpoints, plots, datasets) as objects, and all the
   structured metadata in a SQLite file *that also lives in the bucket*.
2. **A graph, not just a list.** Runs and experiments are nodes; typed edges
   (`forked_from`, `finetuned_from`, `resumed_from`, `uses_artifact`, `part_of`,
   `compared_with`) let an agent trace lineage and group work.
3. **One tidy API for agents.** A `wandb`-style client for logging, and a
   one-call `report()` digest plus a full REST API so agents can review and
   summarize experiments.

```python
import wagentdb

run = wagentdb.init(project="ssm-ds", name="designB-lr3e4",
                    config={"lr": 3e-4, "design": "B"}, tags=["sweep"])
for step in range(1000):
    run.log({"loss": loss, "val/acc": acc}, step=step)
run.log_artifact("model.pt", type="model")
run.summary["best_acc"] = 0.91
run.finish()
```

```python
# Later, a (possibly different) agent reviews the work:
db = wagentdb.connect(project="ssm-ds")
for r in db.runs(status="finished"):
    print(r.name, r.summary)

report = db.report(run_id)                       # everything about a run in one call
db.add_summary(run_id, "Design B overfits after step 600.", kind="review")
```

---

## Why R2 + SQLite?

Object storage is cheap, durable, and S3-compatible, but it can't answer
"give me all finished runs in project X sorted by val/acc". A SQL database can,
but standing one up is friction. `wagentdb` keeps both and gets the best of each:

- **SQLite** is the index for all structured data (projects, experiments, runs,
  metrics, logs, the edge graph, summaries). It's a single self-contained file.
- That file is **stored in R2** under `db_key` (default `wagentdb.sqlite`). On
  open it's pulled into a local cache; after each write it's synced back up.
- **Artifacts** are written directly as R2 objects under
  `artifacts/<run_id>/<artifact_id>/<name>` and referenced from SQLite.

This is a **single-writer** model — ideal for one agent (or one server process)
writing at a time, which is exactly how training jobs log. For many concurrent
writers, run the server (below) and have agents talk to it over HTTP so writes
are serialized in one process.

For local development and tests there's a `local` backend that swaps R2 for a
directory on disk — no credentials needed, same code path.

---

## Install

```bash
pip install -e .            # core
pip install -e ".[all]"     # + R2 (boto3), server (fastapi/uvicorn), client (httpx)
```

Extras: `r2`, `server`, `client`, `all`, `dev`.

## Configure

By default `wagentdb` uses the local backend (`./.wagentdb`). To point it at R2,
set environment variables (or pass a `Settings` object):

```bash
export WAGENTDB_BACKEND=r2
export WAGENTDB_R2_BUCKET=my-experiments
export WAGENTDB_R2_ACCOUNT_ID=xxxxxxxxxxxxxxxx        # derives the endpoint
export WAGENTDB_R2_ACCESS_KEY_ID=...
export WAGENTDB_R2_SECRET_ACCESS_KEY=...
# optional: WAGENTDB_R2_ENDPOINT, WAGENTDB_DB_KEY, WAGENTDB_CACHE_DIR
```

```python
from wagentdb import Settings, init
settings = Settings(backend="r2", r2_bucket="my-experiments",
                    r2_account_id="...", r2_access_key_id="...",
                    r2_secret_access_key="...")
run = init(project="p", settings=settings)
```

---

## The two client modes

The `wandb`-style API is identical whether you log directly to storage
(**embedded**) or to a running server (**http**):

```python
# embedded — agent writes straight to R2 + sqlite
run = wagentdb.init(project="p", config={...})

# http — agent talks to a wagentdb server
run = wagentdb.init(project="p", config={...}, url="https://wagentdb.internal")
```

`Run` methods: `log(metrics, step=)`, `log_text(msg, level=, step=)`,
`log_artifact(path_or_bytes, name=, type=)`, `link(other_run, relation=)`,
`config_update(**kw)`, `run.summary[...] = ...`, `finish(status=)`.

`connect(...)` returns a read/review handle (`DB`): `projects()`,
`experiments()`, `runs(...)`, `run(id)`, `report(id)`, `history(id, key)`,
`lineage(id)`, `graph()`, `add_summary(...)`, `summaries(id)`.

---

## Works with any training stack

wagentdb is framework-agnostic by design — metrics are just dicts and the core
depends on none of these libraries. The pieces that make it slot into whatever
you're training (HF `transformers`, plain PyTorch loops, RL, JAX, ...):

- **Flexible `log()`** — pass a framework's raw log dict. Nested dicts are
  flattened (`{"eval": {"loss": x}}` → `eval/loss`) and non-numeric / NaN values
  are dropped, so `run.log(hf_logs, step=...)` just works.
- **Auto env + git capture** — each run records python/torch/CUDA/GPU and the
  versions of common libs (`transformers`, `datasets`, `accelerate`, `peft`,
  `trl`, `lightning`, `jax`, ...) plus the git commit/remote. Disable with
  `capture_env=False`.
- **Streaming artifacts for LLM-scale checkpoints** — `run.log_artifact(path)`
  streams a file (multipart on R2); pass a **directory** (e.g. a
  `save_pretrained` output) and it's archived to a single `.tar.gz` and
  streamed — checkpoints never get read fully into memory.

### HuggingFace `transformers`

A drop-in callback, the wagentdb equivalent of `WandbCallback`:

```python
from transformers import Trainer
from wagentdb.integrations.transformers import WagentDBCallback

trainer = Trainer(
    model=model, args=training_args, ...,
    callbacks=[WagentDBCallback(project="llm-sft", name="qwen-lora",
                                tags=["lora"], log_model=True)],
)
trainer.train()
```

It captures `TrainingArguments` + the model config, logs every metric the
Trainer emits, optionally uploads checkpoints (`log_checkpoints=True`) and the
final model (`log_model=True`), and finishes the run. Pass `run=` an existing
run to attach instead of creating one. Works the same for causal LMs, seq2seq,
classifiers, etc. — it forwards whatever the Trainer logs.

### Plain training loops

No integration needed — the `wandb`-style client covers any loop:

```python
run = wagentdb.init(project="rl-cartpole", config={"algo": "ppo", "lr": 3e-4})
for episode in range(n):
    run.log({"reward": r, "loss/policy": pl, "loss/value": vl}, step=episode)
run.log_artifact("policy.pt", type="model")
run.finish()
```

## Run the server

```bash
wagentdb serve --port 8000        # or: uvicorn wagentdb.server:app
```

Interactive, machine-readable API docs at `http://localhost:8000/docs`.

Key endpoints:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/runs` | start a run (auto-creates project/experiment) |
| `PATCH` | `/runs/{id}` | update status / summary / config |
| `POST` | `/runs/{id}/metrics` | log a batch of scalar metrics |
| `GET` | `/runs/{id}/metrics/{key}` | full time-series for a metric |
| `POST` | `/runs/{id}/logs` | append a log line |
| `POST` | `/runs/{id}/artifacts` | upload an artifact (raw bytes) |
| `GET` | `/artifacts/{id}/download` | presigned R2 URL (or streamed bytes) |
| `POST` | `/edges` | add a graph edge |
| `GET` | `/nodes/{id}/lineage` · `/ancestry` | trace links |
| `GET` | `/graph` | nodes + edges for visualization |
| `POST` | `/summaries` | an agent writes a review/insight |
| `GET` | **`/runs/{id}/report`** | **one-call digest of a run** |

---

## CLI

```bash
wagentdb serve [--host --port --reload]
wagentdb runs [--project P --status S]
wagentdb report <run_id>           # JSON digest
wagentdb graph [--project P]       # JSON nodes+edges
wagentdb init-db                   # create / sync the database
```

---

## Data model

```
project ──< experiment ──< run ──< metrics
                            │
                            ├──< logs
                            └──< artifacts (R2 objects)

edges:      any node ──relation──> any node   (the graph)
summaries:  agent-written notes attached to any node
```

See [`wagentdb/schema.sql`](wagentdb/schema.sql) for the full schema. Highlights:

- **runs** carry `config` (hyperparameters), `tags`, `summary` (auto-updated
  with the latest value of each metric for quick scanning), `git_commit`,
  `system` info, status and timestamps, and `created_by` (which agent owns it).
- **metrics** are one row per `(run, key, step)` — full history, not just the
  last value.
- **edges** are the graph: a single typed table linking any node to any node.
- **summaries** are how agents leave reviews/insights for the next agent.

---

## Examples

- [`examples/log_run.py`](examples/log_run.py) — an agent logging a training run.
- [`examples/review_agent.py`](examples/review_agent.py) — an agent reviewing
  and summarizing past runs.

```bash
python examples/log_run.py
python examples/review_agent.py
```

## Agent skill

[`.claude/skills/wagentdb/SKILL.md`](.claude/skills/wagentdb/SKILL.md) is a
Claude Code skill that teaches an agent how to log, link, and review experiments
with wagentdb. It's available automatically in sessions in this repo; copy the
`.claude/skills/wagentdb/` directory into your training repos (e.g. `dynaprior`,
`SSM-DS`) so agents there know how to use it too.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The suite runs fully offline against the local backend (storage, server via
`TestClient`, and the client in both embedded and HTTP modes).

## License

MIT

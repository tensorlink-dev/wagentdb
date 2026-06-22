"""wagentdb - an open-source, agent-native experiment tracker.

A small "wandb for AI agents": agents log training experiments to a Cloudflare
R2 bucket (large artifacts as objects, structured metadata in a SQLite file that
also lives in the bucket), link experiments together in a graph, and review /
summarize them through one tidy API.

Quick start::

    import wagentdb
    run = wagentdb.init(project="my-proj", config={"lr": 3e-4})
    run.log({"loss": 0.5}, step=1)
    run.finish()

    db = wagentdb.connect(project="my-proj")
    print(db.report(run.id))
"""

from .client import DB, Run, connect, init
from .config import Settings
from .store import Store

__version__ = "0.1.0"

__all__ = [
    "init",
    "connect",
    "Run",
    "DB",
    "Store",
    "Settings",
    "__version__",
]

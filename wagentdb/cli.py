"""Command-line interface for wagentdb.

    wagentdb serve [--host H --port P]     start the HTTP API server
    wagentdb runs  [--project P]           list runs as a table
    wagentdb report <run_id>               print a run digest (JSON)
    wagentdb graph [--project P]           dump the experiment graph (JSON)
    wagentdb init-db                       create / sync the database
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import Settings
from .store import Store


def _store() -> Store:
    return Store(Settings.from_env())


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("uvicorn is required: pip install wagentdb[server]", file=sys.stderr)
        return 1
    uvicorn.run("wagentdb.server:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def cmd_runs(args: argparse.Namespace) -> int:
    store = _store()
    runs = store.list_runs(project=args.project, status=args.status, limit=args.limit)
    if not runs:
        print("(no runs)")
        return 0
    width = max(len(r.id) for r in runs)
    print(f"{'ID'.ljust(width)}  {'STATUS'.ljust(9)}  NAME")
    for r in runs:
        print(f"{r.id.ljust(width)}  {r.status.ljust(9)}  {r.name}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    store = _store()
    report = store.run_report(args.run_id)
    print(json.dumps(report.model_dump(), indent=2, default=str))
    return 0


def cmd_graph(args: argparse.Namespace) -> int:
    store = _store()
    print(json.dumps(store.graph(project=args.project), indent=2, default=str))
    return 0


def cmd_init_db(args: argparse.Namespace) -> int:
    store = _store()
    store.db.sync_up()
    print(f"database ready (backend={store.settings.backend}) at {store.db.local_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wagentdb", description="agent-native experiment tracker")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("serve", help="run the HTTP API server")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(func=cmd_serve)

    r = sub.add_parser("runs", help="list runs")
    r.add_argument("--project")
    r.add_argument("--status")
    r.add_argument("--limit", type=int, default=50)
    r.set_defaults(func=cmd_runs)

    rep = sub.add_parser("report", help="print a run digest")
    rep.add_argument("run_id")
    rep.set_defaults(func=cmd_report)

    g = sub.add_parser("graph", help="dump the experiment graph")
    g.add_argument("--project")
    g.set_defaults(func=cmd_graph)

    i = sub.add_parser("init-db", help="create/sync the database")
    i.set_defaults(func=cmd_init_db)
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

import pytest
from fastapi.testclient import TestClient

import wagentdb
from wagentdb.server import create_app


@pytest.fixture
def client(store):
    app = create_app(store)
    return TestClient(app)


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_full_run_lifecycle_over_http(client):
    # create run
    r = client.post("/runs", json={"project": "p", "name": "r1", "config": {"lr": 1e-3}})
    assert r.status_code == 200
    run = r.json()
    rid = run["id"]

    # log metrics
    assert client.post(f"/runs/{rid}/metrics", json={"metrics": {"loss": 0.5}, "step": 0}).json()["logged"] == 1

    # log artifact (raw bytes)
    a = client.post(f"/runs/{rid}/artifacts", params={"name": "m.bin", "type": "model"},
                    content=b"weights")
    assert a.status_code == 200
    art_id = a.json()["id"]

    # download artifact (local backend streams bytes)
    dl = client.get(f"/artifacts/{art_id}/download")
    assert dl.status_code == 200
    assert dl.content == b"weights"

    # report
    rep = client.get(f"/runs/{rid}/report").json()
    assert rep["latest_metrics"]["loss"] == 0.5
    assert rep["project"]["name"] == "p"

    # finish
    upd = client.patch(f"/runs/{rid}", json={"status": "finished"})
    assert upd.json()["status"] == "finished"


def test_404_for_missing_run(client):
    assert client.get("/runs/run_does_not_exist").status_code == 404


def test_client_http_mode_against_testclient(client):
    # Drive the wandb-like client through the in-process HTTP server.
    run = wagentdb.init(project="hp", name="viaweb", url="http://testserver",
                        http_client=client, config={"x": 1})
    run.log({"loss": 0.3}, step=0)
    run.finish()

    db = wagentdb.connect(project="hp", url="http://testserver", http_client=client)
    runs = db.runs()
    assert any(r.name == "viaweb" for r in runs)
    report = db.report(run.id)
    assert report.latest_metrics["loss"] == 0.3

from wagentdb import models as m
from wagentdb.config import Settings
from wagentdb.store import Store


def test_create_run_autocreates_project_and_experiment(store):
    run = store.create_run(m.RunCreate(project="proj-a", name="r1", experiment="exp-1",
                                       config={"lr": 0.1}))
    assert run.id.startswith("run_")
    assert run.status == "running"
    assert run.config == {"lr": 0.1}

    proj = store.get_project("proj-a")
    exp = store.get_experiment("exp-1", project="proj-a")
    assert run.project_id == proj.id
    assert run.experiment_id == exp.id

    # part_of edge to the experiment was created
    lin = store.lineage(run.id)
    rels = {e.relation for e in lin["out"]}
    assert "part_of" in rels


def test_metrics_series_and_summary(store):
    run = store.create_run(m.RunCreate(project="p", name="r"))
    for step in range(5):
        store.log_metrics(run.id, {"loss": 1.0 - step * 0.1, "acc": step * 0.2}, step=step)

    keys = set(store.list_metric_keys(run.id))
    assert keys == {"loss", "acc"}

    series = store.get_metric_series(run.id, "loss")
    assert series.values[0] == 1.0
    assert len(series.values) == 5
    assert series.steps == [0, 1, 2, 3, 4]

    latest = store.latest_metrics(run.id)
    assert round(latest["loss"], 5) == round(0.6, 5)

    # summary auto-tracks latest values
    refreshed = store.get_run(run.id)
    assert round(refreshed.summary["acc"], 5) == round(0.8, 5)


def test_logs(store):
    run = store.create_run(m.RunCreate(project="p", name="r"))
    store.log_message(run.id, "starting", level="info", step=0)
    store.log_message(run.id, "uh oh", level="error", step=3)
    logs = store.get_logs(run.id)
    assert [l["message"] for l in logs] == ["starting", "uh oh"]
    errs = store.get_logs(run.id, level="error")
    assert len(errs) == 1


def test_artifact_roundtrip(store):
    run = store.create_run(m.RunCreate(project="p", name="r"))
    art = store.log_artifact(run.id, "model.bin", b"weights-bytes", type="model")
    assert art.size_bytes == len(b"weights-bytes")
    assert art.checksum
    fetched = store.get_artifact_bytes(art.id)
    assert fetched == b"weights-bytes"
    assert store.list_artifacts(run.id)[0].id == art.id


def test_graph_lineage_and_ancestry(store):
    base = store.create_run(m.RunCreate(project="p", name="base"))
    child = store.create_run(m.RunCreate(project="p", name="child"))
    grand = store.create_run(m.RunCreate(project="p", name="grand"))
    store.add_edge(m.EdgeCreate(src_type="run", src_id=child.id, dst_type="run",
                                dst_id=base.id, relation="forked_from"))
    store.add_edge(m.EdgeCreate(src_type="run", src_id=grand.id, dst_type="run",
                                dst_id=child.id, relation="finetuned_from"))

    anc = store.ancestry(grand.id)
    dst_ids = {e.dst_id for e in anc}
    assert base.id in dst_ids and child.id in dst_ids

    g = store.graph(project="p")
    assert len(g["nodes"]) >= 3


def test_summaries(store):
    run = store.create_run(m.RunCreate(project="p", name="r"))
    store.add_summary(m.SummaryCreate(subject_type="run", subject_id=run.id,
                                      content="overfits after 600 steps", kind="review",
                                      created_by="agent-7"))
    sums = store.list_summaries(run.id)
    assert len(sums) == 1
    assert sums[0].created_by == "agent-7"


def test_run_report(store):
    run = store.create_run(m.RunCreate(project="p", name="r", config={"lr": 1e-3}))
    store.log_metrics(run.id, {"loss": 0.5}, step=1)
    store.log_message(run.id, "hello")
    store.add_summary(m.SummaryCreate(subject_type="run", subject_id=run.id, content="ok"))
    report = store.run_report(run.id)
    assert report.run.id == run.id
    assert report.project.name == "p"
    assert "loss" in report.latest_metrics
    assert report.summaries[0].content == "ok"
    assert report.log_tail[-1]["message"] == "hello"


def test_persistence_reopen(tmp_path):
    settings = Settings(backend="local", local_dir=tmp_path / "wdb")
    s1 = Store(settings)
    run = s1.create_run(m.RunCreate(project="p", name="r"))
    s1.log_metrics(run.id, {"loss": 0.1}, step=0)
    s1.close()

    s2 = Store(settings)
    assert s2.get_run(run.id).name == "r"
    assert s2.latest_metrics(run.id)["loss"] == 0.1
    s2.close()


def test_db_synced_to_object_store(tmp_path):
    # The sqlite file should be written into the object store ("R2") too.
    settings = Settings(backend="local", local_dir=tmp_path / "wdb")
    s = Store(settings)
    s.create_run(m.RunCreate(project="p", name="r"))
    assert s.object_store.exists(settings.db_key)
    s.close()

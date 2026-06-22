import wagentdb


def test_embedded_init_log_finish(store):
    run = wagentdb.init(project="p", name="r1", config={"lr": 1e-3}, store=store)
    for step in range(3):
        run.log({"loss": 1.0 - step * 0.1}, step=step)
    run.log_text("done training")
    art = run.log_artifact(b"weights", name="model.bin", type="model")
    run.summary["best"] = 0.42
    run.finish()

    assert art.name == "model.bin"
    refreshed = store.get_run(run.id)
    assert refreshed.status == "finished"
    assert refreshed.summary["best"] == 0.42
    assert refreshed.summary["loss"] is not None


def test_embedded_review(store):
    a = wagentdb.init(project="p", name="base", store=store)
    a.log({"loss": 0.5}, step=0)
    a.finish()
    b = wagentdb.init(project="p", name="child", store=store, fork_from=a.id)
    b.finish()

    db = wagentdb.connect(project="p", store=store)
    runs = db.runs()
    assert {r.name for r in runs} == {"base", "child"}

    lineage = db.lineage(b.id)
    assert any(e.relation == "forked_from" for e in lineage["out"])

    db.add_summary(a.id, "baseline looks healthy", kind="review", created_by="agent-x")
    assert db.summaries(a.id)[0].content == "baseline looks healthy"

    report = db.report(a.id)
    assert report.run.id == a.id
    assert "loss" in report.latest_metrics

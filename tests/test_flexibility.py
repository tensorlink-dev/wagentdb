"""Tests for framework-agnostic flexibility: nested/tolerant logging, env
capture, streaming file/dir artifacts, and the transformers callback."""

import io
import tarfile
from types import SimpleNamespace

import wagentdb
from wagentdb.sysinfo import collect_system_info
from wagentdb.utils import flatten_numeric


def test_flatten_numeric_handles_nesting_and_non_numbers():
    flat = dict(flatten_numeric({
        "loss": 0.5,
        "eval": {"acc": 0.9, "name": "run"},   # nested dict + a string
        "ok": True,                             # bool -> 1.0
        "note": "hello",                        # dropped
        "nan": float("nan"),                    # dropped
        "items": [1, 2, 3],                     # dropped
    }))
    assert flat == {"loss": 0.5, "eval/acc": 0.9, "ok": 1.0}


def test_log_accepts_framework_style_dict(store):
    run = wagentdb.init(project="p", name="r", store=store, capture_env=False)
    # A dict like HF transformers' on_log payload, with mixed/nested values.
    run.log({"loss": 0.3, "eval": {"loss": 0.4}, "epoch": 1.0, "phase": "warmup"}, step=5)
    keys = set(store.list_metric_keys(run.id))
    assert keys == {"loss", "eval/loss", "epoch"}


def test_init_captures_env_and_git(store):
    run = wagentdb.init(project="p", name="r", store=store)  # capture_env defaults True
    fetched = store.get_run(run.id)
    assert fetched.system is not None
    assert "python" in fetched.system


def test_collect_system_info_keys():
    info = collect_system_info()
    assert "python" in info and "platform" in info


def test_log_artifact_file_streams(tmp_path, store):
    f = tmp_path / "weights.bin"
    f.write_bytes(b"x" * 5000)
    run = wagentdb.init(project="p", name="r", store=store, capture_env=False)
    art = run.log_artifact(str(f), type="model")
    assert art.size_bytes == 5000
    assert store.get_artifact_bytes(art.id) == b"x" * 5000


def test_log_artifact_directory_tars(tmp_path, store):
    d = tmp_path / "checkpoint"
    d.mkdir()
    (d / "config.json").write_text('{"hidden": 8}')
    (d / "model.bin").write_bytes(b"weights")
    run = wagentdb.init(project="p", name="r", store=store, capture_env=False)
    art = run.log_artifact(str(d), type="checkpoint")
    assert art.name.endswith(".tar.gz")

    # The archive round-trips and contains our files.
    raw = store.get_artifact_bytes(art.id)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        names = {m.name for m in tar.getmembers() if m.isfile()}
    assert any(n.endswith("config.json") for n in names)
    assert any(n.endswith("model.bin") for n in names)


def test_transformers_callback_without_transformers_installed(tmp_path, store):
    from wagentdb.integrations.transformers import WagentDBCallback

    cb = WagentDBCallback(project="hf", name="run1", store=store,
                          tags=["test"], log_model=True)

    args = SimpleNamespace(
        output_dir=str(tmp_path),
        to_dict=lambda: {"learning_rate": 1e-4, "num_train_epochs": 3},
    )
    state = SimpleNamespace(global_step=10)
    model = SimpleNamespace(config=SimpleNamespace(to_dict=lambda: {"hidden_size": 768}))

    cb.on_train_begin(args, state, None, model=model)
    cb.on_log(args, state, None, logs={"loss": 0.5, "epoch": 1.0, "stage": "warmup"})
    cb.on_train_end(args, state, None, model=model)

    run = cb.run
    assert run is not None
    fetched = store.get_run(run.id)
    assert fetched.status == "finished"
    # TrainingArguments + model config were captured.
    assert fetched.config["learning_rate"] == 1e-4
    assert fetched.config["model/hidden_size"] == 768
    # numeric metrics logged, string dropped.
    assert set(store.list_metric_keys(run.id)) == {"loss", "epoch"}

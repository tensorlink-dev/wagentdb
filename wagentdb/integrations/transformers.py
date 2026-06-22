"""HuggingFace ``transformers`` integration.

A drop-in :class:`~transformers.TrainerCallback` that logs a Trainer's metrics,
config, and (optionally) checkpoints to wagentdb — the same role
``WandbCallback`` plays for Weights & Biases.

Usage::

    from transformers import Trainer
    from wagentdb.integrations.transformers import WagentDBCallback

    trainer = Trainer(
        model=model,
        args=training_args,
        ...,
        callbacks=[WagentDBCallback(project="llm-sft", name="qwen-lora",
                                    tags=["lora"], log_model=True)],
    )
    trainer.train()

Or attach an existing run you started yourself::

    run = wagentdb.init(project="llm-sft", config={...})
    trainer.add_callback(WagentDBCallback(run=run))

The callback works regardless of model type (causal LM, seq2seq, classifier,
...): it just forwards whatever metrics the Trainer logs.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .. import init as _init
from ..client import Run

# Subclass the real TrainerCallback when transformers is installed; otherwise
# fall back to a plain base so the module still imports (and is testable).
try:  # pragma: no cover - depends on optional dependency
    from transformers import TrainerCallback as _Base
except Exception:  # pragma: no cover
    class _Base:  # type: ignore
        pass


class WagentDBCallback(_Base):
    def __init__(
        self,
        run: Optional[Run] = None,
        *,
        project: Optional[str] = None,
        name: Optional[str] = None,
        experiment: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        created_by: Optional[str] = None,
        url: Optional[str] = None,
        token: Optional[str] = None,
        settings: Any = None,
        store: Any = None,
        log_checkpoints: bool = False,
        log_model: bool = False,
    ):
        if run is None and project is None:
            raise ValueError("WagentDBCallback needs either run= or project=")
        self.run = run
        self._owns_run = run is None
        self._init_kwargs = dict(
            project=project, name=name, experiment=experiment, config=config or {},
            tags=tags, created_by=created_by, url=url, token=token,
            settings=settings, store=store,
        )
        self.log_checkpoints = log_checkpoints
        self.log_model = log_model

    # ------------------------------------------------------------------ hooks
    def on_train_begin(self, args=None, state=None, control=None, model=None, **kwargs):
        if self.run is None:
            config = dict(self._init_kwargs.get("config") or {})
            config.update(_args_to_dict(args))
            config.update(_model_config(model))
            kw = {**self._init_kwargs, "config": config}
            self.run = _init(**{k: v for k, v in kw.items() if v is not None or k == "config"})
        else:
            extra = {**_args_to_dict(args), **_model_config(model)}
            if extra:
                self.run.config_update(**extra)
        return control

    def on_log(self, args=None, state=None, control=None, logs=None, **kwargs):
        if self.run is not None and logs:
            step = getattr(state, "global_step", None)
            self.run.log(dict(logs), step=step)
        return control

    def on_save(self, args=None, state=None, control=None, **kwargs):
        if self.run is not None and self.log_checkpoints and args is not None:
            step = getattr(state, "global_step", None)
            ckpt = os.path.join(getattr(args, "output_dir", "."), f"checkpoint-{step}")
            if os.path.isdir(ckpt):
                self.run.log_artifact(ckpt, name=f"checkpoint-{step}", type="checkpoint")
        return control

    def on_train_end(self, args=None, state=None, control=None, model=None, **kwargs):
        if self.run is None:
            return control
        if self.log_model and args is not None:
            out = getattr(args, "output_dir", None)
            if out and os.path.isdir(out):
                self.run.log_artifact(out, name="model", type="model")
        if self._owns_run:
            self.run.finish()
        return control


def _args_to_dict(args: Any) -> Dict[str, Any]:
    if args is None:
        return {}
    for attr in ("to_sanitized_dict", "to_dict"):
        fn = getattr(args, attr, None)
        if callable(fn):
            try:
                return dict(fn())
            except Exception:
                pass
    return {}


def _model_config(model: Any) -> Dict[str, Any]:
    cfg = getattr(model, "config", None)
    to_dict = getattr(cfg, "to_dict", None)
    if callable(to_dict):
        try:
            return {f"model/{k}": v for k, v in to_dict().items()}
        except Exception:
            return {}
    return {}

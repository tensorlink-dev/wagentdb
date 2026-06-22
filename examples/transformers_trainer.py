"""Example: logging a HuggingFace transformers Trainer to wagentdb.

This mirrors how you'd wire it into a real LLM / model fine-tune. It needs
`transformers` installed (`pip install wagentdb[transformers] transformers`).
The only wagentdb-specific line is adding the callback.

    python examples/transformers_trainer.py
"""

# This is illustrative pseudo-real code; fill in your own model/dataset.
from wagentdb.integrations.transformers import WagentDBCallback


def main() -> None:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    args = TrainingArguments(
        output_dir="./out",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to=[],  # disable built-in integrations; we use wagentdb
    )

    trainer = Trainer(
        model=model,
        args=args,
        # train_dataset=..., eval_dataset=..., tokenizer=tokenizer,
        callbacks=[
            WagentDBCallback(
                project="llm-demo",
                name="distilbert-sst2",
                tags=["classification", "demo"],
                created_by="agent-trainer",
                log_checkpoints=True,   # upload each checkpoint dir as a .tar.gz
                log_model=True,         # upload the final model dir
                # url="https://wagentdb.internal",  # or run against a server
            )
        ],
    )
    trainer.train()  # metrics, config, env, git, and artifacts land in wagentdb


if __name__ == "__main__":
    main()

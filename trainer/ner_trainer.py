from transformers import TrainingArguments, Trainer

class NERTrainer:
    def __init__(self, model, tokenizer, train_ds, eval_ds, metrics_fn):
        self.training_args = TrainingArguments(
            output_dir="./results",
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_dir="./logs",
            report_to="tensorboard",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            load_best_model_at_end=True,
            fp16=True,
        )

        self.trainer = Trainer(
            model=model,
            args=self.training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=tokenizer,
            compute_metrics=metrics_fn
        )

    def train(self):
        self.trainer.train()

    def evaluate(self, test_ds):
        return self.trainer.evaluate(eval_dataset=test_ds)

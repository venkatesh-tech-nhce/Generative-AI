!pip install -q transformers==4.46.3 peft==0.13.2 datasets accelerate

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)

# Exp 6: PEFT with LoRA
class LoRAClassifier:
    def __init__(self):
        data = load_dataset("fancyzhx/ag_news")

        # Old:
        # self.train = data["train"].select(range(100))
        # self.test = data["test"].select(range(20))

        self.train = data["train"].select(range(1000))
        self.test = data["test"].select(range(100))

        self.tokenizer = AutoTokenizer.from_pretrained(
            "distilbert-base-uncased"
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=4
        )

        config = LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules=["q_lin", "v_lin"],
            task_type=TaskType.SEQ_CLS
        )

        self.model = get_peft_model(
            self.model,
            config
        )

    def tokenize(self, x):
        return self.tokenizer(
            x["text"],
            padding="max_length",
            truncation=True,
            max_length=64
        )

    def run(self):
        self.train = self.train.map(
            self.tokenize,
            batched=True
        )

        self.test = self.test.map(
            self.tokenize,
            batched=True
        )

        print("\nLoRA Parameters:")
        self.model.print_trainable_parameters()

        # Old:
        # args = TrainingArguments(
        #     output_dir="result",
        #     num_train_epochs=1,
        #     per_device_train_batch_size=16,
        #     learning_rate=2e-4,
        #     save_strategy="no",
        #     report_to="none"
        # )

        args = TrainingArguments(
            output_dir="result",
            num_train_epochs=3,
            per_device_train_batch_size=16,
            learning_rate=2e-4,
            save_strategy="no",
            report_to="none"
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.train,
            processing_class=self.tokenizer
        )

        trainer.train()

        result = trainer.predict(self.test)
        predicted = result.predictions.argmax(axis=1)

        classes = {
            0: "World",
            1: "Sports",
            2: "Business",
            3: "Technology"
        }

        print("\n========== PREDICTIONS ==========")

        for i in range(5):
            print("\nText:", self.test[i]["text"])
            print("Actual:", classes[self.test[i]["label"]])
            print("Predicted:", classes[predicted[i]])

        print("\nPEFT + LoRA Experiment Completed")


LoRAClassifier().run()
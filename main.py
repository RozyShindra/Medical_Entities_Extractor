from utils.label_utils import LabelUtils
from model.tokenizer_handler import TokenizerHandler
from model.model_handler import ModelHandler
from processing.data_loader import NERDataLoader
from processing.aligner import TokenAligner
from trainer.metrics import MetricsCalculator
from trainer.ner_trainer import NERTrainer

if __name__ == "__main__":
    labels = LabelUtils("data/labels.json")
    tokenizer = TokenizerHandler().get_tokenizer()
    model = ModelHandler("bert-base-uncased", len(labels.label2id)).get_model()

    train = NERDataLoader("data/train.json").load_data()
    dev = NERDataLoader("data/dev.json").load_data()
    test = NERDataLoader("data/test.json").load_data()

    aligner = TokenAligner(tokenizer, labels.label2id)
    train = train.map(aligner)
    dev = dev.map(aligner)
    test = test.map(aligner)

    metrics = MetricsCalculator(labels.id2label)
    trainer = NERTrainer(model, tokenizer, train, dev, metrics.compute)
    trainer.train()
    print(trainer.evaluate(test))

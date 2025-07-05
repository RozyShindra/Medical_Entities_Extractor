from transformers import AutoModelForTokenClassification

class ModelHandler:
    def __init__(self, model_id, num_labels):
        self.model = AutoModelForTokenClassification.from_pretrained(model_id, num_labels=num_labels)

    def get_model(self):
        return self.model

from transformers import AutoTokenizer

class TokenizerHandler:
    def __init__(self, model_id="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def get_tokenizer(self):
        return self.tokenizer

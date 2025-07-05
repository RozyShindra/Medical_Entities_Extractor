class TokenAligner:
    def __init__(self, tokenizer, label2id):
        self.tokenizer = tokenizer
        self.label2id = label2id

    def __call__(self, example):
        tokenized = self.tokenizer(example["tokens"], is_split_into_words=True, truncation=True, padding="max_length", max_length=128)
        word_ids = tokenized.word_ids()
        aligned_labels = []

        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)
            else:
                aligned_labels.append(self.label2id[example["labels"][word_id]])

        tokenized["labels"] = aligned_labels
        return tokenized

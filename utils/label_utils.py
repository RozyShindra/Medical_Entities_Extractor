import json

class LabelUtils:
    def __init__(self, label_path):
        with open(label_path, 'r') as f:
            self.label_list = json.load(f)

        self.unique_labels = ['O'] + [f'B-{l}' for l in self.label_list] + [f'I-{l}' for l in self.label_list]
        self.label2id = {label: i for i, label in enumerate(self.unique_labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}

import json
import pandas as pd
from datasets import Dataset

class NERDataLoader:
    def __init__(self, path):
        self.path = path

    def load_data(self):
        with open(self.path, 'r') as f:
            raw = json.load(f)

        data = []
        for entry in raw:
            text = entry["sentence"]
            tokens = list(text)
            labels = ["O"] * len(tokens)

            for ent in entry.get("entities", []):
                start, end = ent["pos"]
                if end > len(labels): continue
                labels[start] = f"B-{ent['type']}"
                for i in range(start + 1, end):
                    labels[i] = f"I-{ent['type']}"
            data.append({"tokens": tokens, "labels": labels})
        return Dataset.from_pandas(pd.DataFrame(data))

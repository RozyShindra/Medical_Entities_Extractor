from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score

class MetricsCalculator:
    def __init__(self, id2label):
        self.id2label = id2label

    def compute(self, p):
        pred, true = p.predictions.argmax(-1), p.label_ids
        pred_out, true_out = [], []

        for p_seq, t_seq in zip(pred, true):
            temp_pred, temp_true = [], []
            for p_id, t_id in zip(p_seq, t_seq):
                if t_id != -100:
                    temp_pred.append(self.id2label[p_id])
                    temp_true.append(self.id2label[t_id])
            pred_out.append(temp_pred)
            true_out.append(temp_true)

        return {
            "accuracy": accuracy_score(true_out, pred_out),
            "f1": f1_score(true_out, pred_out),
            "precision": precision_score(true_out, pred_out),
            "recall": recall_score(true_out, pred_out)
        }

class AccuracyMetrics:
    def __init__(self):
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0

    def update(self, matches, total_broadcast, total_tacticam):
        matched_ids = set(matches.keys())
        self.true_positives += len(matches)
        self.false_positives += total_tacticam - len(matches)
        self.false_negatives += total_broadcast - len(matches)

    def compute(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-6)
        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        return {
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'f1_score': round(f1, 3)
        }

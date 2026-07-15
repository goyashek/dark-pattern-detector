BENIGN = "Not a Dark Pattern"
ABSTAIN_THRESHOLD = 0.50


def result_status(label, top_score):
    if top_score < ABSTAIN_THRESHOLD:
        return "inconclusive"
    return "no_signal" if label == BENIGN else "signal"

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


# accuracy
def acc(predictions, target):
    return sum([predictions[i] == target[i] for i in range(len(predictions))]) / len(predictions)


# average per-digit accuracy
def mean_digit_acc(predictions, target):
    accum = 0
    for i in range(len(predictions)):
        pr = np.array(list(predictions[i]))
        tg = np.array(list(target[i]))
        l_min = min(pr.shape[0], tg.shape[0])
        l_max = max(pr.shape[0], tg.shape[0])
        if l_min != 0:
            accum += np.sum(pr[:l_min] == tg[:l_min]) / l_max

    return accum / len(predictions)


def evaluate(model, datasets, metrics, row_labels, col_labels):
    model.network.eval()
    t = pd.DataFrame(data=np.zeros([len(datasets), len(metrics)]),
                     index=row_labels,
                     columns=col_labels)

    for row in tqdm(range(len(datasets)), desc='Datasets', colour="#005500"):
        prediction = []
        target = []
        for (a, b), trg in tqdm(datasets[row], desc='Elements', colour="#00ff00"):
            prediction.append(model.generate(a, b))
            target.append(trg)
            for col in range(len(metrics)):
                metric = metrics[col](prediction, target)
                t.iloc[row][col] = metric
    return t

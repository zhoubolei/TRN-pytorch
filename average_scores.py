import numpy as np
import pdb
files_scores = ['output/TSN_moments_Flow_BNInception_avg_segment2.npz','output/TSN_moments_RGB_BNInception_avg_segment3_best.npz']


def compute_acc(labels, scores):

    pdb.set_trace()
    preds_max = np.argmax(scores)
    num_correct = np.sum(preds == labels)
    acc = num_correct * 1.0 / preds.shape[0]
    return acc

for filename in files_scores:
    data = np.load(filename)
    preds = data['predictions']
    labels = data['labels']
    scores = data['scores']
    acc_scores = compute_acc(labels, scores)
    num_correct = np.sum(preds == labels)
    acc = num_correct * 1.0 / preds.shape[0]
    print('acc=%.3f acc_score=%.3f %s' % (acc, acc_scores, filename))

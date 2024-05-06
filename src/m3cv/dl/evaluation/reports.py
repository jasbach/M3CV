import numpy as np

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score
)

def classification_report(eval_config, true, preds):
    report = {}
    bin_preds = np.copy(preds)
    bin_preds[bin_preds >= eval_config.binarize_threshold] = 1
    bin_preds[bin_preds < eval_config.binarize_threshold] = 0
    
    for metric in eval_config.metrics:
        if metric == 'accuracy':
            report['accuracy'] = accuracy_score(true, bin_preds)
        elif metric == 'precision':
            report['precision'] = precision_score(true, bin_preds)
        elif metric == 'recall':
            report['recall'] = recall_score(true, bin_preds)
        elif metric == 'AUC ROC':
            report['AUC ROC'] = roc_auc_score(true, preds)
        elif metric == 'confusion matrix':
            tn, fp, fn, tp = confusion_matrix(true, bin_preds).ravel()
            report['tn'] = int(tn)
            report['fp'] = int(fp)
            report['fn'] = int(fn)
            report['tp'] = int(tp)
    return report
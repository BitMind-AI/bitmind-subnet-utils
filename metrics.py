

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef


def compute_metrics(predictions, labels):
    """
    Compute multiclass and binary metrics from predictions and labels.

    Parameters:
    predictions (list): List of predicted class probabilities (softmax outputs)
    labels (list): List of true labels

    Returns:
    dict: A dictionary with multiclass and binary metrics
    """
    # Convert softmax predictions to class labels
    pred_labels = [p.index(max(p)) for p in predictions]
    
    # Multiclass metrics
    multi_accuracy = accuracy_score(labels, pred_labels)
    multi_precision = precision_score(labels, pred_labels, average='weighted', zero_division=0)
    multi_recall = recall_score(labels, pred_labels, average='weighted', zero_division=0)
    multi_f1 = f1_score(labels, pred_labels, average='weighted', zero_division=0)
    multi_mcc = matthews_corrcoef(labels, pred_labels)

    # Convert to binary by combining labels 1 and 2
    binary_labels = [1 if l > 0 else 0 for l in labels]
    binary_preds = [1 if p > 0 else 0 for p in pred_labels]
    
    # Sum probabilities for classes 1 and 2 for binary AUC
    binary_probs = [p[1] + p[2] for p in predictions]

    binary_accuracy = accuracy_score(binary_labels, binary_preds)
    binary_precision = precision_score(binary_labels, binary_preds, zero_division=0)
    binary_recall = recall_score(binary_labels, binary_preds, zero_division=0)
    binary_f1 = f1_score(binary_labels, binary_preds, zero_division=0)
    binary_auc = roc_auc_score(binary_labels, binary_probs)
    binary_mcc = matthews_corrcoef(binary_labels, binary_preds)
    
    return {
        "multiclass_accuracy": multi_accuracy,
        "multiclass_precision": multi_precision, 
        "multiclass_recall": multi_recall,
        "multiclass_f1": multi_f1,
        "multiclass_mcc": multi_mcc,
        "binary_accuracy": binary_accuracy,
        "binary_precision": binary_precision,
        "binary_recall": binary_recall,
        "binary_f1": binary_f1,
        "binary_auc": binary_auc,
        "binary_mcc": binary_mcc,
        "sample_size": len(predictions)
    }
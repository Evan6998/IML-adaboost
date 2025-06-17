import numpy as np


def read_data(args) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads in train and test datasets as numpy arrays. This function is already written for you.
    """
    train_data = np.loadtxt(args.train_data, dtype=np.float32, delimiter = ',')
    test_data = np.loadtxt(args.test_data, dtype=np.float32, delimiter = ',')
    train_data, train_labels = train_data[:, :-1], train_data[:, -1]
    test_data, test_labels = test_data[:, :-1], test_data[:, -1]
    return (
        train_data,
        train_labels,
        test_data,
        test_labels
    )


def classify(data: np.ndarray, dim: int, threshold: float, label: int) -> np.ndarray:
    '''
    Return predictions on data for a decision stump h(x) 
    parametrized by (dim, threshold, label):

    h(x) = {  label   if x[dim] > threshold
           { -label   otherwise
    '''
    return np.where(data[:, dim] > threshold, label, -1 * label)


def weak_classifier(
    data: np.ndarray, 
    labels: np.ndarray, 
    weights: np.ndarray,
) -> tuple[int, float, int, np.ndarray]:
    '''
    Finds weak classifier that minimizes error under current point weights.

    Returns parameters of best weak classifier, (dim, threshold, label) and predictions of
    weak classifier on dataset.
    '''
    _, D = data.shape
    
    epsilon = 0.000001
    min_error = np.Infinity
    params = None

    for d in range(D):
        thresholds = np.insert(np.unique(data[:, d]), 0, np.min(data[:, d]) - epsilon)
        for threshold in thresholds:
            for l in [-1, +1]:
                h = classify(data, d, threshold, l)
                weighted_error = np.sum(weights * (labels != h))
                if weighted_error < min_error:
                    min_error = weighted_error
                    params = d, threshold, l, h
    return params # type: ignore


def update_weights(
    current_weights: np.ndarray, 
    alpha: float,
    predictions: np.ndarray,
    labels: np.ndarray
) -> np.ndarray:
    '''
    Returns updated point weights.
    '''
    next_weights = current_weights * np.exp(-alpha * labels * predictions)
    return next_weights / np.sum(next_weights)


def train(
    num_iter: int,
    train_data: np.ndarray,
    train_labels: np.ndarray,
    test_data: np.ndarray,
    test_labels: np.ndarray,
) -> tuple[list, list, np.ndarray, list]:
    '''
    Trains adaboost model for num_iter iterations.

    Returns:
        weak_classifiers (list): list of weak classifiers over iterations -- (num_iters, 3)
        alphas (list): list of alphas over iterations -- (num_iters,)
        weights (np.ndarray): final weights over data points -- (len(train_data),)
        test_accs (list): list of test accuracies over iterations -- (num_iters,)
    '''
    N, _ = train_data.shape
    weights = np.ones(N) / N
    
    weak_classifiers = []
    alphas = []
    test_accs = []

    for _ in range(num_iter):
        dim, threshold, label, predictions = weak_classifier(train_data, train_labels, weights)
        weak_classifiers.append((dim, threshold, label))
        
        epsilon = np.sum(weights * (predictions != train_labels))
        
        alpha = np.log((1-epsilon) / epsilon) / 2
        alphas.append(alpha)

        acc, test_predictions = evaluate(weak_classifiers, alphas, test_data, test_labels)
        test_accs.append(acc)

        weights = update_weights(weights, alpha, predictions, train_labels)

    return weak_classifiers, alphas, weights, test_accs, test_predictions

def evaluate(
    weak_classifiers: list,
    alpha_list: list,
    test_data: np.ndarray,
    test_labels: np.ndarray,
) -> tuple[float, list]:
    '''
    Evaluates adaboost model on test data.
    Returns test accuracy and predictions on test data.
    '''
    predictions = np.zeros(len(test_data))
    for (dim, threshold, label), alpha in zip(weak_classifiers, alpha_list):
        predictions += classify(test_data, dim, threshold, label) * alpha
    sign_predictions = np.sign(predictions)
    return np.sum(sign_predictions == test_labels) / len(sign_predictions), sign_predictions


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data", type=str, help='Train data input')
    parser.add_argument("--test_data", type=str, help='Test data input')
    parser.add_argument("--num_iters", type=int, help='Number of iterations of adaboost')
    parser.add_argument("--metrics_out", type=str, help='Test accuracies over iterations')
    parser.add_argument("--test_out", type=str, help='FINAL predictions on test data')
    parser.add_argument("--weights_out", type=str, help='FINAL point weights over train data')

    args = parser.parse_args()

    train_data, train_labels, test_data, test_labels = read_data(args)

    # Your train/test code goes here
    weak_classifiers, alphas, weights, accs, preds = \
        train(args.num_iters, train_data, train_labels, test_data, test_labels)


    with open(args.metrics_out, "w") as f:
        for i in range(len(accs)):
            curr_epoch = i + 1
            curr_test_acc = accs[i]
            f.write("epoch={} accuracy(test): {}\n".format(
                curr_epoch, curr_test_acc))

    np.savetxt(args.test_out, preds, delimiter=',')

    np.savetxt(args.weights_out, weights, delimiter=',')
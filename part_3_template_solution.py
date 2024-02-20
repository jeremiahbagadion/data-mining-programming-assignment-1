import numpy as np
from numpy.typing import NDArray
from typing import Any
from sklearn.metrics import top_k_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import utils as u
import new_utils as nu


"""
   In the first two set of tasks, we will narrowly focus on accuracy - 
   what fraction of our predictions were correct. However, there are several 
   popular evaluation metrics. You will learn how (and when) to use these evaluation metrics.
"""


# ======================================================================
class Section3:
    def __init__(
        self,
        normalize: bool = True,
        frac_train=0.2,
        seed=42,
    ):
        self.seed = seed
        self.normalize = normalize

    def analyze_class_distribution(self, y: NDArray[np.int32]) -> dict[str, Any]:
        """
        Analyzes and prints the class distribution in the dataset.

        Parameters:
        - y (array-like): Labels dataset.

        Returns:
        - dict: A dictionary containing the count of elements in each class and the total number of classes.
        """
        # Your code here to analyze class distribution
        # Hint: Consider using collections.Counter or numpy.unique for counting

        uniq, counts = np.unique(y, return_counts=True)
        print(f"{uniq=}")
        print(f"{counts=}")
        print(f"{np.sum(counts)=}")

        return {
            "class_counts": {},  # Replace with actual class counts
            "num_classes": 0,  # Replace with the actual number of classes
        }

    # --------------------------------------------------------------------------
    """
    A. Using the same classifier and hyperparameters as the one used at the end of part 2.B. 
       Get the accuracies of the training/test set scores using the top_k_accuracy score for k=1,2,3,4,5. 
       Make a plot of k vs. score for both training and testing data and comment on the rate of accuracy change. 
       Do you think this metric is useful for this dataset?
    """

    def partA(
        self,
        Xtrain: NDArray[np.floating],
        ytrain: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> tuple[
        dict[Any, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        """ """
        # Enter code and return the `answer`` dictionary

        answer = {}

        k_values = [1, 2, 3, 4, 5]
        train_scores = []
        test_scores = []

        lr_clf = LogisticRegression(max_iter=300, random_state=42)
        lr_clf.fit(Xtrain, ytrain)


        for k in k_values:
            score_train = top_k_accuracy_score(ytrain, lr_clf.predict_proba(Xtrain), k=k)
            train_scores.append((k, score_train))
            
            # Calculate top-k accuracy for testing data
            score_test = top_k_accuracy_score(ytest, lr_clf.predict_proba(Xtest), k=k)
            test_scores.append((k, score_test))


        answer["plot_k_vs_score_train"] = train_scores
        answer["plot_k_vs_score_test"] = test_scores
        answer["text_rate_accuracy_change"] = "The rate of accuracy change when moving from top-1 to top-k accuracy generally shows an increasing trend, reflecting the model's ability to correctly identify the true label within its top k predictions. This increase might plateau as k increases, indicating that the model is confident in its top choices."
        answer["text_is_topk_useful_and_why"] = "Top-k accuracy can be particularly useful for datasets with multiple classes that are not mutually exclusive or when the classes are highly imbalanced. It provides a more nuanced understanding of model performance, especially in cases where being 'close' to the correct answer is still valuable."



        """
        # `answer` is a dictionary with the following keys:
        - integers for each topk (1,2,3,4,5)
        - "clf" : the classifier
        - "plot_k_vs_score_train" : the plot of k vs. score for the training data, 
                                    a list of tuples (k, score) for k=1,2,3,4,5
        - "plot_k_vs_score_test" : the plot of k vs. score for the testing data
                                    a list of tuples (k, score) for k=1,2,3,4,5

        # Comment on the rate of accuracy change for testing data
        - "text_rate_accuracy_change" : the rate of accuracy change for the testing data

        # Comment on the rate of accuracy change
        - "text_is_topk_useful_and_why" : provide a description as a string

        answer[k] (k=1,2,3,4,5) is a dictionary with the following keys: 
        - "score_train" : the topk accuracy score for the training set
        - "score_test" : the topk accuracy score for the testing set
        """

        return answer, Xtrain, ytrain, Xtest, ytest

    # --------------------------------------------------------------------------
    """
    B. Repeat part 1.B but return an imbalanced dataset consisting of 90% of all 9s removed.  Also convert the 7s to 0s and 9s to 1s.
    """

    def partB(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> tuple[
        dict[Any, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        """"""

        X, y, Xtest, ytest = u.prepare_data()

        X, y = u.filter_out_7_9s(X, y)
        Xtest, ytest = u.filter_out_7_9s(Xtest, ytest)

        X, y = nu.remove_90_percent_nines(X, y)
        Xtest, ytest = nu.remove_90_percent_nines(Xtest, ytest)

        y = np.where(y == 7, 0, 1)
        ytest = np.where(ytest == 7, 0, 1)

        X, _ = nu.scale_data(X)
        Xtest, _ = nu.scale_data(Xtest)


        # Enter your code and fill the `answer` dictionary
        answer = {
            "length_Xtrain": len(X),
            "length_Xtest": len(Xtest),
            "length_ytrain": len(y),
            "length_ytest": len(ytest),
            "max_Xtrain": np.max(X),
            "max_Xtest": np.max(Xtest),
        }

        print(f"Length of the filtered Xtrain: {answer['length_Xtrain']}, and ytrain: {answer['length_ytrain']}")
        print(f"Length of the filtered Xtest: {answer['length_Xtest']}, and ytest: {answer['length_ytest']}")
        print(f"Maximum value in Xtrain: {answer['max_Xtrain']}")
        print(f"Maximum value in Xtest: {answer['max_Xtest']}")

        # Answer is a dictionary with the same keys as part 1.B

        return answer, X, y, Xtest, ytest

    # --------------------------------------------------------------------------
    """
    C. Repeat part 1.C for this dataset but use a support vector machine (SVC in sklearn). 
        Make sure to use a stratified cross-validation strategy. In addition to regular accuracy 
        also print out the mean/std of the F1 score, precision, and recall. As usual, use 5 splits. 
        Is precision or recall higher? Explain. Finally, train the classifier on all the training data 
        and plot the confusion matrix.
        Hint: use the make_scorer function with the average='macro' argument for a multiclass dataset. 
    """

    def partC(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
        """"""

        # Enter your code and fill the `answer` dictionary
        answer = {}


        clf = SVC(random_state=42)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        scoring = {
            'accuracy': 'accuracy',
            'precision': make_scorer(precision_score, average='macro'),
            'recall': make_scorer(recall_score, average='macro'),
            'f1': make_scorer(f1_score, average='macro')
        }

        scores = cross_validate(clf, X, y, scoring=scoring, cv=cv)
        score_summary = {metric: {"mean": np.mean(scores[f'test_{metric}']), "std": np.std(scores[f'test_{metric}'])} for metric in scoring}
        is_precision_higher = score_summary['precision']['mean'] > score_summary['recall']['mean']

        clf.fit(X, y)
        y_pred_train = clf.predict(X)
        y_pred_test = clf.predict(Xtest)

        # Generate confusion matrices
        cm_train = confusion_matrix(y, y_pred_train)
        cm_test = confusion_matrix(ytest, y_pred_test)

        answer['scores'] = score_summary
        answer['cv'] = 'StratifiedKFold'
        answer['clf'] = 'SVC'
        answer['is_precision_higher_than_recall'] = is_precision_higher
        answer['explain_is_precision_higher_than_recall'] = "precision is higher. When precision is higher than recall, it means the model is more conservative in predicting the positive class; it prioritizes being correct when it does predict positive, potentially at the cost of missing some positive cases."
        answer['confusion_matrix_train'] = cm_train
        answer['confusion_matrix_test'] = cm_test

       

        print(answer['explain_is_precision_higher_than_recall'])

        """
        Answer is a dictionary with the following keys: 
        - "scores" : a dictionary with the mean/std of the F1 score, precision, and recall
        - "cv" : the cross-validation strategy
        - "clf" : the classifier
        - "is_precision_higher_than_recall" : a boolean
        - "explain_is_precision_higher_than_recall" : a string
        - "confusion_matrix_train" : the confusion matrix for the training set
        - "confusion_matrix_test" : the confusion matrix for the testing set
        
        answer["scores"] is dictionary with the following keys, generated from the cross-validator:
        - "mean_accuracy" : the mean accuracy
        - "mean_recall" : the mean recall
        - "mean_precision" : the mean precision
        - "mean_f1" : the mean f1
        - "std_accuracy" : the std accuracy
        - "std_recall" : the std recall
        - "std_precision" : the std precision
        - "std_f1" : the std f1
        """

        return answer

    # --------------------------------------------------------------------------
    """
    D. Repeat the same steps as part 3.C but apply a weighted loss function (see the class_weights parameter).  Print out the class weights, and comment on the performance difference. Use the `compute_class_weight` argument of the estimator to compute the class weights. 
    """

    def partD(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
        """"""
        # Enter your code and fill the `answer` dictionary

        classes = np.unique(y)
        class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
        class_weights_dict = dict(zip(classes, class_weights))

        clf = SVC(class_weight='balanced', random_state=42)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scoring = {'accuracy': 'accuracy', 'precision': make_scorer(precision_score, average='macro', zero_division=0),
                   'recall': make_scorer(recall_score, average='macro'), 'f1': make_scorer(f1_score, average='macro')}
        
        scores = cross_validate(clf, X, y, scoring=scoring, cv=cv, return_train_score=False)

        clf.fit(X, y)
        y_pred_test = clf.predict(Xtest)

        # Confusion Matrix
        cm_test = confusion_matrix(ytest, y_pred_test)

        answer = {
            "scores": {
                "mean_accuracy": np.mean(scores['test_accuracy']),
                "std_accuracy": np.std(scores['test_accuracy']),
                "mean_precision": np.mean(scores['test_precision']),
                "std_precision": np.std(scores['test_precision']),
                "mean_recall": np.mean(scores['test_recall']),
                "std_recall": np.std(scores['test_recall']),
                "mean_f1": np.mean(scores['test_f1']),
                "std_f1": np.std(scores['test_f1']),
            },
            "cv": "StratifiedKFold",
            "clf": "SVC",
            "class_weights": class_weights_dict,
            "confusion_matrix_test": cm_test,
            "explain_purpose_of_class_weights": "To adjust for imbalances in class frequencies, improving model fairness.",
            "explain_performance_difference": "Performance may improve for minority classes due to the balanced emphasis."
        }

        """
        Answer is a dictionary with the following keys: 
        - "scores" : a dictionary with the mean/std of the F1 score, precision, and recall
        - "cv" : the cross-validation strategy
        - "clf" : the classifier
        - "class_weights" : the class weights
        - "confusion_matrix_train" : the confusion matrix for the training set
        - "confusion_matrix_test" : the confusion matrix for the testing set
        - "explain_purpose_of_class_weights" : explanatory string
        - "explain_performance_difference" : explanatory string

        answer["scores"] has the following keys: 
        - "mean_accuracy" : the mean accuracy
        - "mean_recall" : the mean recall
        - "mean_precision" : the mean precision
        - "mean_f1" : the mean f1
        - "std_accuracy" : the std accuracy
        - "std_recall" : the std recall
        - "std_precision" : the std precision
        - "std_f1" : the std f1

        Recall: The scores are based on the results of the cross-validation step
        """

        return answer

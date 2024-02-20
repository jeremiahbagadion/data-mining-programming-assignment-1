# Inspired by GPT4

# Information on type hints
# https://peps.python.org/pep-0585/

# GPT on testing functions, mock functions, testing number of calls, and argument values
# https://chat.openai.com/share/b3fd7739-b691-48f2-bb5e-0d170be4428c


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    ShuffleSplit,
    cross_validate,
    KFold,
    GridSearchCV,
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

from typing import Any
from numpy.typing import NDArray
from utils import train_simple_classifier_with_cv
import numpy as np
import utils as u

# Initially empty. Use for reusable functions across
# Sections 1-3 of the homework
import new_utils as nu


# ======================================================================
class Section1:
    def __init__(
        self,
        normalize: bool = True,
        seed: int | None = None,
        frac_train: float = 0.2,
    ):
        """
        Initializes an instance of MyClass.

        Args:
            normalize (bool, optional): Indicates whether to normalize the data. Defaults to True.
            seed (int, optional): The seed value for randomization. If None, each call will be randomized.
                If an integer is provided, calls will be repeatable.

        Returns:
            None

        Notes: notice the argument `seed`. Make sure that any sklearn function that accepts
        `random_state` as an argument is initialized with this seed to allow reproducibility.
        You change the seed ONLY in the section of run_part_1.py, run_part2.py, run_part3.py
        below `if __name__ == "__main__"`
        """
        self.normalize = normalize
        self.frac_train = frac_train
        self.seed = seed

    # ----------------------------------------------------------------------
    """
    A. We will start by ensuring that your python environment is configured correctly and 
       that you have all the required packages installed. For information about setting up 
       Python please consult the following link: https://www.anaconda.com/products/individual. 
       To test that your environment is set up correctly, simply execute `starter_code` in 
       the `utils` module. This is done for you. 
    """

    def partA(self):
        # Return 0 (ran ok) or -1 (did not run ok)
        answer = u.starter_code()
        return answer

    # ----------------------------------------------------------------------
    """
    B. Load and prepare the mnist dataset, i.e., call the prepare_data and filter_out_7_9s 
       functions in utils.py, to obtain a data matrix X consisting of only the digits 7 and 9. Make sure that 
       every element in the data matrix is a floating point number and scaled between 0 and 1 (write
       a function `def scale() in new_utils.py` that returns a bool to achieve this. Checking is not sufficient.) 
       Also check that the labels are integers. Print out the length of the filtered 𝑋 and 𝑦, 
       and the maximum value of 𝑋 for both training and test sets. Use the routines provided in utils.
       When testing your code, I will be using matrices different than the ones you are using to make sure 
       the instructions are followed. 
    """

    def partB(
        self,
    ):
        X, y, Xtest, ytest = u.prepare_data()
        Xtrain, ytrain = u.filter_out_7_9s(X, y)
        Xtest, ytest = u.filter_out_7_9s(Xtest, ytest)
        Xtrain, scaled_correctly_train = nu.scale_data(Xtrain)
        Xtest, scaled_correctly_test = nu.scale_data(Xtest)

        # Validate if data is scaled correctly and labels are integers
        if not (scaled_correctly_train and scaled_correctly_test):
            raise ValueError("Data scaling error.")

        if not (ytrain.dtype == int and ytest.dtype == int):
            raise ValueError("Labels are not integers.")

        answer = {
            "length_Xtrain": len(Xtrain),  # Number of samples in the training set
            "length_Xtest": len(Xtest),  # Number of samples in the test set
            "length_ytrain": len(ytrain),  # Number of labels in the training set
            "length_ytest": len(ytest),  # Number of labels in the test set
            "max_Xtrain": Xtrain.max(),  # Maximum value in the training set
            "max_Xtest": Xtest.max(),
            }

        # Enter your code and fill the `answer` dictionary
        print(f"Length of Xtrain: {len(Xtrain)}")
        print(f"Length of Xtest: {len(Xtest)}")
        print(f"Length of ytrain: {len(ytrain)}")
        print(f"Length of ytest: {len(ytest)}")
        print(f"Maximum value in Xtrain: {Xtrain.max()}")
        print(f"Maximum value in Xtest: {Xtest.max()}")

        return answer, Xtrain, ytrain, Xtest, ytest

    """
    C. Train your first classifier using k-fold cross validation (see train_simple_classifier_with_cv 
       function). Use 5 splits and a Decision tree classifier. Print the mean and standard deviation 
       for the accuracy scores in each validation set in cross validation. (with k splits, cross_validate
       generates k accuracy scores.)  
       Remember to set the random_state in the classifier and cross-validator.
    """

    # ----------------------------------------------------------------------
    def partC(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
    ):
        # Enter your code and fill the `answer` dictionary
        clf = DecisionTreeClassifier(random_state=42)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = train_simple_classifier_with_cv(Xtrain=X, ytrain=y, clf=clf, cv=cv)

        answer = {
            "clf": clf,
            "cv": cv,
            "scores": {
                'mean_fit_time': np.mean(cv_results['fit_time']),
                'std_fit_time': np.std(cv_results['fit_time']),
                'mean_accuracy': np.mean(cv_results['test_score']),
                'std_accuracy': np.std(cv_results['test_score']),
            }
        }
        
        
        return answer

    # ---------------------------------------------------------
    """
    D. Repeat Part C with a random permutation (Shuffle-Split) 𝑘-fold cross-validator.
    Explain the pros and cons of using Shuffle-Split versus 𝑘-fold cross-validation.
    """

    def partD(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
    ):
        # Enter your code and fill the `answer` dictionary
        clf = DecisionTreeClassifier(random_state=42)
        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

        cv_results = train_simple_classifier_with_cv(Xtrain=X, ytrain=y, clf=clf, cv=cv)

        # Answer: same structure as partC, except for the key 'explain_kfold_vs_shuffle_split'

        answer = {
            "clf": clf,
            "cv": cv,
            "scores": {
                'mean_fit_time': np.mean(cv_results['fit_time']),
                'std_fit_time': np.std(cv_results['fit_time']),
                'mean_accuracy': np.mean(cv_results['test_score']),
                'std_accuracy': np.std(cv_results['test_score']),
            },
            "explain_kfold_vs_shuffle_split": "Shuffle-Split allows for more flexible train/test splits and can better handle imbalanced datasets by ensuring each split is representative. However, it may introduce more variance in evaluation metrics due to random sampling and may not use all data points for training or testing, unlike k-Fold which systematically uses all data in folds."
        }
        return answer

    # ----------------------------------------------------------------------
    """
    E. Repeat part D for 𝑘=2,5,8,16, but do not print the training time. 
       Note that this may take a long time (2–5 mins) to run. Do you notice 
       anything about the mean and/or standard deviation of the scores for each k?
    """

    def partE(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
    ):
        # Answer: built on the structure of partC
        # `answer` is a dictionary with keys set to each split, in this case: 2, 5, 8, 16
        # Therefore, `answer[k]` is a dictionary with keys: 'scores', 'cv', 'clf`

        answer = {}
        k_values = [2, 5, 8, 16]
        for k in k_values:
            cv = ShuffleSplit(n_splits=k, test_size=0.2, random_state=42)
            clf = DecisionTreeClassifier(random_state=42)
            cv_results = train_simple_classifier_with_cv(Xtrain=X, ytrain=y, clf=clf, cv=cv)

            mean_fit_time = np.mean(cv_results['fit_time'])
            std_fit_time = np.std(cv_results['fit_time'])
            mean_accuracy = np.mean(cv_results['test_score'])
            std_accuracy = np.std(cv_results['test_score'])

            answer[k] = {
                "scores": {
                    'mean_fit_time': mean_fit_time,
                    'std_fit_time': std_fit_time,
                    'mean_accuracy': mean_accuracy,
                    'std_accuracy': std_accuracy
                },
                "cv": cv,
                "clf": clf
            }

        # Enter your code, construct the `answer` dictionary, and return it.

        return answer

    # ----------------------------------------------------------------------
    """
    F. Repeat part D with a Random-Forest classifier with default parameters. 
       Make sure the train test splits are the same for both models when performing 
       cross-validation. (Hint: use the same cross-validator instance for both models.)
       Which model has the highest accuracy on average? 
       Which model has the lowest variance on average? Which model is faster 
       to train? (compare results of part D and part F)

       Make sure your answers are calculated and not copy/pasted. Otherwise, the automatic grading 
       will generate the wrong answers. 
       
       Use a Random Forest classifier (an ensemble of DecisionTrees). 
    """

    def partF(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
    ) -> dict[str, Any]:
        """ """

        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

        clf_DT = DecisionTreeClassifier(random_state=42)
        scores_DT = train_simple_classifier_with_cv(Xtrain=X, ytrain=y, clf=clf_DT, cv=cv)

        clf_RF = RandomForestClassifier(random_state=42)
        scores_RF = train_simple_classifier_with_cv(Xtrain=X, ytrain=y, clf=clf_RF, cv=cv)

        answer = {
            "clf_RF": clf_RF,
            "clf_DT": clf_DT,
            "cv": cv,
            "scores_RF": {
                'mean_fit_time': np.mean(scores_RF['fit_time']),
                'std_fit_time': np.std(scores_RF['fit_time']),
                'mean_accuracy': np.mean(scores_RF['test_score']),
                'std_accuracy': np.std(scores_RF['test_score']),
            },
            "scores_DT": {
                'mean_fit_time': np.mean(scores_DT['fit_time']),
                'std_fit_time': np.std(scores_DT['fit_time']),
                'mean_accuracy': np.mean(scores_DT['test_score']),
                'std_accuracy': np.std(scores_DT['test_score']),
            },
        }

        if answer["scores_RF"]["mean_accuracy"] > answer["scores_DT"]["mean_accuracy"]:
            answer["model_highest_accuracy"] = "Random Forest"
        else:
            answer["model_highest_accuracy"] = "Decision Tree"

        if answer["scores_RF"]["std_accuracy"] < answer["scores_DT"]["std_accuracy"]:
            answer["model_lowest_variance"] = "Random Forest"
        else:
            answer["model_lowest_variance"] = "Decision Tree"

    # Compare fit time to determine which model is faster
        if answer["scores_RF"]["mean_fit_time"] < answer["scores_DT"]["mean_fit_time"]:
            answer["model_fastest"] = "Random Forest"
        else:
            answer["model_fastest"] = "Decision Tree"

        print("Mean Accuracy for Random Forest:", answer["scores_RF"]["mean_accuracy"])
        print("Mean Accuracy for Decision Tree:", answer["scores_DT"]["mean_accuracy"])

        # Enter your code, construct the `answer` dictionary, and return it.

        """
         Answer is a dictionary with the following keys: 
            "clf_RF",  # Random Forest class instance
            "clf_DT",  # Decision Tree class instance
            "cv",  # Cross validator class instance
            "scores_RF",  Dictionary with keys: "mean_fit_time", "std_fit_time", "mean_accuracy", "std_accuracy"
            "scores_DT",  Dictionary with keys: "mean_fit_time", "std_fit_time", "mean_accuracy", "std_accuracy"
            "model_highest_accuracy" (string)
            "model_lowest_variance" (float)
            "model_fastest" (float)
        """

        return answer

    # ----------------------------------------------------------------------
    """
    G. For the Random Forest classifier trained in part F, manually (or systematically, 
       i.e., using grid search), modify hyperparameters, and see if you can get 
       a higher mean accuracy.  Finally train the classifier on all the training 
       data and get an accuracy score on the test set.  Print out the training 
       and testing accuracy and comment on how it relates to the mean accuracy 
       when performing cross validation. Is it higher, lower or about the same?

       Choose among the following hyperparameters: 
         1) criterion, 
         2) max_depth, 
         3) min_samples_split, 
         4) min_samples_leaf, 
         5) max_features 
    """

    def partG(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
        """
        Perform classification using the given classifier and cross validator.

        Parameters:
        - clf: The classifier instance to use for classification.
        - cv: The cross validator instance to use for cross validation.
        - X: The test data.
        - y: The test labels.
        - n_splits: The number of splits for cross validation. Default is 5.

        Returns:
        - y_pred: The predicted labels for the test data.

        Note:
        This function is not fully implemented yet.
        """

        # refit=True: fit with the best parameters when complete
        # A test should look at best_index_, best_score_ and best_params_
        """
        List of parameters you are allowed to vary. Choose among them.
         1) criterion,
         2) max_depth,
         3) min_samples_split, 
         4) min_samples_leaf,
         5) max_features 
         5) n_estimators
        """

        clf = RandomForestClassifier(random_state=42)
        default_parameters = clf.get_params()

        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2'],
            'n_estimators': [100, 200, 300]
        }

        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

        grid_search = GridSearchCV(clf, param_grid, cv=cv, n_jobs=1, verbose=1, scoring='accuracy')
        grid_search.fit(X, y)

        best_clf = grid_search.best_estimator_
        best_clf.fit(X, y)

        y_pred_train = best_clf.predict(X)
        y_pred_test = best_clf.predict(Xtest)

        answer = {
            "clf": clf,
            "default_parameters": default_parameters,
            "best_estimator": best_clf,
            "grid_search": grid_search,
            "mean_accuracy_cv": grid_search.best_score_,
            "confusion_matrix_train_orig": confusion_matrix(y, clf.fit(X, y).predict(X)),
            "confusion_matrix_train_best": confusion_matrix(y, y_pred_train),
            "confusion_matrix_test_orig": confusion_matrix(ytest, clf.predict(Xtest)),
            "confusion_matrix_test_best": confusion_matrix(ytest, y_pred_test),
            "accuracy_orig_full_training": accuracy_score(y, clf.predict(X)),
            "accuracy_best_full_training": accuracy_score(y, y_pred_train),
            "accuracy_orig_full_testing": accuracy_score(ytest, clf.predict(Xtest)),
            "accuracy_best_full_testing": accuracy_score(ytest, y_pred_test),
        }

        print("Training Accuracy (Original):", answer["accuracy_orig_full_training"])
        print("Training Accuracy (Best):", answer["accuracy_best_full_training"])
        print("Testing Accuracy (Original):", answer["accuracy_orig_full_testing"])
        print("Testing Accuracy (Best):", answer["accuracy_best_full_testing"])

        # Enter your code, construct the `answer` dictionary, and return it.

        """
           `answer`` is a dictionary with the following keys: 
            
            "clf", base estimator (classifier model) class instance
            "default_parameters",  dictionary with default parameters 
                                   of the base estimator
            "best_estimator",  classifier class instance with the best
                               parameters (read documentation)
            "grid_search",  class instance of GridSearchCV, 
                            used for hyperparameter search
            "mean_accuracy_cv",  mean accuracy score from cross 
                                 validation (which is used by GridSearchCV)
            "confusion_matrix_train_orig", confusion matrix of training 
                                           data with initial estimator 
                                (rows: true values, cols: predicted values)
            "confusion_matrix_train_best", confusion matrix of training data 
                                           with best estimator
            "confusion_matrix_test_orig", confusion matrix of test data
                                          with initial estimator
            "confusion_matrix_test_best", confusion matrix of test data
                                            with best estimator
            "accuracy_orig_full_training", accuracy computed from `confusion_matrix_train_orig'
            "accuracy_best_full_training"
            "accuracy_orig_full_testing"
            "accuracy_best_full_testing"
               
        """

        return answer

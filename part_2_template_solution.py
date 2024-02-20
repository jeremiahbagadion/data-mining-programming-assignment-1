# Add your imports here.
# Note: only sklearn, numpy, utils and new_utils are allowed.

import numpy as np
from numpy.typing import NDArray
from typing import Any
import new_utils as nu
import utils as u
from new_utils import print_scores
from utils import train_simple_classifier_with_cv, print_cv_result_dict
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, ShuffleSplit, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# ======================================================================

# I could make Section 2 a subclass of Section 1, which would facilitate code reuse.
# However, both classes have the same function names. Better to pass Section 1 instance
# as an argument to Section 2 class constructor.


class Section2:
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
        """
        self.normalize = normalize
        self.seed = seed
        self.frac_train = frac_train

    # ---------------------------------------------------------

    """
    A. Repeat part 1.B but make sure that your data matrix (and labels) consists of
        all 10 classes by also printing out the number of elements in each class y and 
        print out the number of classes for both training and testing datasets. 
    """

    def partA(
        self,
    ) -> tuple[
        dict[str, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        answer = {}

        Xtrain, ytrain, Xtest, ytest = u.prepare_data()

        Xtrain, _ = nu.scale_data(Xtrain)
        Xtest, _ = nu.scale_data(Xtest)

        nb_classes_train = len(np.unique(ytrain))
        nb_classes_test = len(np.unique(ytest))
        class_count_train = np.bincount(ytrain)
        class_count_test = np.bincount(ytest)

        answer['nb_classes_train'] = nb_classes_train
        answer['nb_classes_test'] = nb_classes_test
        answer['class_count_train'] = class_count_train
        answer['class_count_test'] = class_count_test
        answer['length_Xtrain'] = len(Xtrain)
        answer['length_Xtest'] = len(Xtest)
        answer['length_ytrain'] = len(ytrain)
        answer['length_ytest'] = len(ytest)
        answer['max_Xtrain'] = np.max(Xtrain)
        answer['max_Xtest'] = np.max(Xtest)

        print("Number of classes in the training set:", answer['nb_classes_train'])
        print("Number of classes in the testing set:", answer['nb_classes_test'])
        print("Number of elements in each class for the training set:", answer['class_count_train'])
        print("Number of elements in each class for the testing set:", answer['class_count_test'])
        # Enter your code and fill the `answer`` dictionary

        # `answer` is a dictionary with the following keys:
        # - nb_classes_train: number of classes in the training set
        # - nb_classes_test: number of classes in the testing set
        # - class_count_train: number of elements in each class in the training set
        # - class_count_test: number of elements in each class in the testing set
        # - length_Xtrain: number of elements in the training set
        # - length_Xtest: number of elements in the testing set
        # - length_ytrain: number of labels in the training set
        # - length_ytest: number of labels in the testing set
        # - max_Xtrain: maximum value in the training set
        # - max_Xtest: maximum value in the testing set

        # return values:
        # Xtrain, ytrain, Xtest, ytest: the data used to fill the `answer`` dictionary

        #Xtrain = Xtest = np.zeros([1, 1], dtype="float")
        #ytrain = ytest = np.zeros([1], dtype="int")

        return answer, Xtrain, ytrain, Xtest, ytest

    """
    B.  Repeat part 1.C, 1.D, and 1.F, for the multiclass problem. 
        Use the Logistic Regression for part F with 300 iterations. 
        Explain how multi-class logistic regression works (inherent, 
        one-vs-one, one-vs-the-rest, etc.).
        Repeat the experiment for ntrain=1000, 5000, 10000, ntest = 200, 1000, 2000.
        Comment on the results. Is the accuracy higher for the training or testing set?
        What is the scores as a function of ntrain.

        Given X, y from mnist, use:
        Xtrain = X[0:ntrain, :]
        ytrain = y[0:ntrain]
        Xtest = X[ntrain:ntrain+test]
        ytest = y[ntrain:ntrain+test]
    """

    def partB(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
        ntrain_list: list[int] = [1000, 5000, 10000],
        ntest_list: list[int] = [200, 1000, 2000],
    ) -> dict[int, dict[str, Any]]:
        """ """
        # Enter your code and fill the `answer`` dictionary
        answer = {}
        scaler = StandardScaler()
        #clf = DecisionTreeClassifier(random_state=42)
        #cv = KFold(n_splits=5, shuffle=True, random_state=42)
        #print("X shape:", X.shape)
        #print("y shape:", y.shape)
        #scores = train_simple_classifier_with_cv(Xtrain=X, ytrain=y, clf=clf, cv=cv)
        #print_cv_result_dict(scores)

        for ntrain in ntrain_list:
            for ntest in ntest_list:

                Xtrain, ytrain = X[:ntrain], y[:ntrain]
                Xtest, ytest = X[ntrain:ntrain+ntest], y[ntrain:ntrain+ntest]

                Xtrain_scaled = scaler.fit_transform(Xtrain)
                Xtest_scaled = scaler.transform(Xtest)

                # Part 1.C: Decision Tree with K-Fold Cross-Validation
                dt_clf = DecisionTreeClassifier(random_state=42)
                cv_kfold = KFold(n_splits=5, shuffle=True, random_state=42)
                dt_scores = cross_validate(dt_clf, Xtrain, ytrain, cv=cv_kfold)

                # Part 1.D: Shuffle-Split Cross-Validation
                cv_shuffle = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
                dt_shuffle_scores = cross_validate(dt_clf, Xtrain, ytrain, cv=cv_shuffle)

                lr_clf = LogisticRegression(max_iter=1000, random_state=42)
                lr_clf.fit(Xtrain_scaled, ytrain)
                lr_train_accuracy = lr_clf.score(Xtrain_scaled, ytrain)
                lr_test_accuracy = lr_clf.score(Xtest_scaled, ytest)

                answer[ntrain] = {
                    'partC': {'mean_accuracy': np.mean(dt_scores['test_score']), 'std_accuracy': np.std(dt_scores['test_score'])},
                    'partD': {'mean_accuracy': np.mean(dt_shuffle_scores['test_score']), 'std_accuracy': np.std(dt_shuffle_scores['test_score'])},
                    'partF': {'train_accuracy': lr_train_accuracy, 'test_accuracy': lr_test_accuracy},
                    'ntrain': ntrain,
                    'ntest': ntest,
                    'class_count_train': list(np.bincount(ytrain)),
                    'class_count_test': list(np.bincount(ytest)[1:]), 
                }

                print_scores(answer)


        """
        `answer` is a dictionary with the following keys:
           - 1000, 5000, 10000: each key is the number of training samples

           answer[k] is itself a dictionary with the following keys
            - "partC": dictionary returned by partC section 1
            - "partD": dictionary returned by partD section 1
            - "partF": dictionary returned by partF section 1
            - "ntrain": number of training samples
            - "ntest": number of test samples
            - "class_count_train": number of elements in each class in
                               the training set (a list, not a numpy array)
            - "class_count_test": number of elements in each class in
                               the training set (a list, not a numpy array)
        """

        return answer

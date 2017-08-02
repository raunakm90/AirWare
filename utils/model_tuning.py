from sklearn.externals import joblib
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV, StratifiedShuffleSplit, learning_curve
from .feature_engineering import remove_near_zero_var, MaskedPCA
from .generate_report import write_results
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def gridSearch_clf(x, y, groups, param_grid, clf, file_path='./baseline_models/'):
    # Define Leave one subject out CV object
    cv_obj = LeaveOneGroupOut()

    pipe = Pipeline([
        ('normalize', StandardScaler()),
        ('reduce_dim', MaskedPCA()),
        ('classify', clf)
    ])

    grid = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=cv_obj, n_jobs=-1,
                        scoring='accuracy', verbose=1)
    grid.fit(x, y, groups=groups)

    print("Best Score")
    print(grid.best_score_)

    print("Grid Search based Best Estimator")
    print(grid.best_estimator_)

    print("Saving best estimator to disk")
    joblib.dump(grid.best_params_, filename=file_path + "clf_gridsearch.pkl")

    return grid.best_estimator_


def eval_model(clf_pipe, x, y, file_path):
    # Delete near zero variance columns
    nz_var_ind = remove_near_zero_var(x, thresh=20)
    x = np.delete(x, nz_var_ind, axis=1)

    cv_obj = StratifiedShuffleSplit(n_splits=5, random_state=234, test_size=0.3)
    train_sizes = np.linspace(.1, 1.0, 5)
    train_sizes, train_scores, test_scores = learning_curve(clf_pipe, x, y, cv=cv_obj,
                                                            n_jobs=-1, train_sizes=train_sizes)
    write_results(train_sizes, train_scores, test_scores, file_path)
    # plot_learning_curve(train_sizes, train_scores, test_scores, "SVM_Learning_Curve",
    #                     file_path)

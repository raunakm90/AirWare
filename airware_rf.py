from data import Read_Data
from sklearn.ensemble import RandomForestClassifier
from utils.baseline_model_helper import *
from utils.generate_report import *
import argparse

MODEL_PATH = "./baseline_models/rf/"
CV_FOLDS = 5


def run_gridSearch_rf(cv_strategy=None):
    start = time.time()
    # Number of principle components for Masked PCA
    n_components_range = [100, 200]
    # Number of trees in the forest
    n_estimators = [10, 50, 100, 500, 1000]
    # Number of features to consider when looking for best split
    max_features = ["sqrt", "log2"]

    # Min Samples at leaf node
    min_samples_leaf = [1, 10, 50]

    x, y, user, lab_enc = airware_baseline_data()
    # Delete near zero variance columns
    nz_var_ind = remove_near_zero_var(x, thresh=20)
    x = np.delete(x, nz_var_ind, axis=1)

    # Create a mask for PCA only on doppler signature
    mask = np.arange(x.shape[1]) < x.shape[1] - 2
    param_grid = [
        {
            'reduce_dim__n_components': n_components_range,
            'reduce_dim__mask': [mask],
            'classify__n_estimators': n_estimators,
            'classify__max_features': max_features,
            'classify__class_weight': ['balanced'],
            'classify__bootstrap': [True],
            'classify__criterion': ["gini", "entropy"],
            'classify__min_samples_leaf': min_samples_leaf,
            'classify__random_state': [2346]
        }
    ]
    clf_obj = RandomForestClassifier()
    grid_search_best_estimator = gridSearch_clf(x=x, y=y, groups=user, clf=clf_obj, param_grid=param_grid,
                                                file_path=MODEL_PATH)
    print('It took ', time.time() - start, ' seconds.')
    return grid_search_best_estimator


def eval_rf(cv_strategy='loso'):
    start = time.time()
    rf_clf_params = joblib.load(MODEL_PATH + "clf_gridsearch.pkl")
    pipe = Pipeline([
        ('normalize', StandardScaler()),
        ('reduce_dim', MaskedPCA()),
        ('classify', RandomForestClassifier())
    ])
    if cv_strategy == 'loso':
        print("Random Forest with Leave One Subject CV")
        train_clf_loso(pipe, rf_clf_params, MODEL_PATH + "leave_one_subject/rf")
        print('It took ', time.time() - start, ' seconds.')
    elif cv_strategy == 'personalized':
        print("Random Forest with Personalized CV")
        train_clf_personalized(pipe, rf_clf_params, MODEL_PATH + "personalized/rf")
        print('It took ', time.time() - start, ' seconds.')
    elif cv_strategy == 'user_calibrated':
        print("Random Forest with User Calibrated CV")
        train_size_percent = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        for train_size in train_size_percent:
            train_clf_user_calibrated(pipe, rf_clf_params, train_size, MODEL_PATH + "user_calibrated/rf"+str(train_size))
        print('It took ', time.time() - start, ' seconds.')
    else:
        raise ValueError("Cross-validation strategy not defined")


if __name__ == '__main__':
    function_map = {'gridSearch': run_gridSearch_rf,
                    'eval_rf': eval_rf}
    parser = argparse.ArgumentParser(
        description="AirWare Random forest grid search and train model using different CV strategies")
    # "?" one argument consumed from the command line and produced as a single item
    # Positional arguments
    parser.add_argument('-model_strategy',
                        help="Define function to run for Random Forest",
                        choices=['gridSearch', 'eval_rf'],
                        default='eval_rf')
    parser.add_argument('-cv_strategy',
                        help="Define cross-validation strategy",
                        choices=['loso', 'personalized', 'user_calibrated'],
                        default=None)

    args = parser.parse_args()
    function = function_map[args.model_strategy]
    function(args.cv_strategy)

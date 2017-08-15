from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from utils.baseline_model_helper import *
from utils.generate_report import *
import argparse

MODEL_PATH = "./baseline_models/mlp/"


def run_gridSearch_mlp(cv_strategy=None):
    start = time.time()
    x, y, user, lab_enc = airware_baseline_data()
    # Delete near zero variance columns
    nz_var_ind = remove_near_zero_var(x, thresh=20)
    x = np.delete(x, nz_var_ind, axis=1)

    # Create a mask for PCA only on doppler signature
    mask = np.arange(x.shape[1]) < x.shape[1] - 2

    # Number of principle components for Masked PCA
    n_components_range = [100, 200]

    param_grid = {
        'reduce_dim__n_components': n_components_range,
        'reduce_dim__mask': [mask],
        'classify__hidden_layer_sizes': [(50,), (100,), (200,), (500,), (50, 50,), (100, 100,), (50, 100,), (100, 500,),
                                         (500, 100,), (100, 200), (200, 100),
                                         (500, 250,), (250, 100,)],
        'classify__alpha': [0.0001, 0.01, 1, 10, 100],
        'classify__activation': ['relu', 'tanh', 'logistic'],
        'classify__max_iter': [1000],
        'classify__random_state': [576],
        'classify__solver': ['adam'],
        'classify__early_stopping': [True],
        'classify__beta_1': [0.9],
        'classify__beta_2': [0.999]}

    clf_obj = MLPClassifier()
    grid_search_best_estimator = gridSearch_clf(x=x, y=y, groups=user, clf=clf_obj, param_grid=param_grid,
                                                file_path=MODEL_PATH)

    print('It took ', time.time() - start, ' seconds.')
    return grid_search_best_estimator


def eval_mlp(cv_strategy='loso'):
    start = time.time()
    mlp_clf_params = joblib.load(MODEL_PATH + "clf_gridsearch.pkl")
    pipe = Pipeline([
        ('normalize', StandardScaler()),
        ('reduce_dim', MaskedPCA()),
        ('classify', MLPClassifier())
    ])
    if cv_strategy == 'loso':
        print("MLP with Leave One Subject CV")
        train_clf_loso(pipe, mlp_clf_params, MODEL_PATH + "leave_one_subject/mlp")
        print('It took ', time.time() - start, ' seconds.')
    elif cv_strategy == 'personalized':
        print("MLP with Personalized CV")
        train_clf_personalized(pipe, mlp_clf_params, MODEL_PATH + "personalized/mlp")
        print('It took ', time.time() - start, ' seconds.')
    elif cv_strategy == 'user_calibrated':
        print("MLP with User Calibrated CV")
        train_size_percent = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        for train_size in train_size_percent:
            train_clf_user_calibrated(pipe, mlp_clf_params, train_size,
                                      MODEL_PATH + "user_calibrated/mlp" + str(train_size))
        print('It took ', time.time() - start, ' seconds.')
    else:
        raise ValueError("Cross-validation strategy not defined")


def eval_mlp_doppler(cv_strategy='loso'):
    start = time.time()
    mlp_clf_params = {'classify__activation': 'tanh',
                      'classify__alpha': 0.1,
                      'classify__beta_1': 0.9,
                      'classify__beta_2': 0.999,
                      'classify__early_stopping': True,
                      'classify__hidden_layer_sizes': (500, 250),
                      'classify__max_iter': 1000,
                      'classify__solver': 'adam',
                      'classify__random_state': 576,
                      'reduce_dim__n_components': 100}
    pipe = Pipeline([
        ('normalize', StandardScaler()),
        ('reduce_dim', PCA()),
        ('classify', MLPClassifier())
    ])
    if cv_strategy == 'loso':
        print("MLP with Leave One Subject CV - Doppler")
        train_clf_doppler(pipe, mlp_clf_params, MODEL_PATH + "/doppler/")
        print('It took ', time.time() - start, ' seconds.')
    else:
        raise ValueError("Cross-validation strategy not defined")


def eval_mlp_ir(cv_strategy='loso'):
    start = time.time()
    mlp_clf_params = {'classify__activation': 'tanh',
                      'classify__alpha': 0.1,
                      'classify__beta_1': 0.9,
                      'classify__beta_2': 0.999,
                      'classify__early_stopping': True,
                      'classify__hidden_layer_sizes': (500, 250),
                      'classify__max_iter': 1000,
                      'classify__solver': 'adam',
                      'classify__random_state': 576}
    pipe = Pipeline([
        ('normalize', StandardScaler()),
        ('classify', MLPClassifier())
    ])
    if cv_strategy == 'loso':
        print("MLP with Leave One Subject CV - IR")
        train_clf_ir(pipe, mlp_clf_params, MODEL_PATH + "/ir/")
        print('It took ', time.time() - start, ' seconds.')
    else:
        raise ValueError("Cross-validation strategy not defined")


if __name__ == '__main__':
    function_map = {'gridSearch': run_gridSearch_mlp,
                    'eval_mlp': eval_mlp,
                    'eval_mlp_doppler': eval_mlp_doppler,
                    'eval_mlp_ir': eval_mlp_ir}
    parser = argparse.ArgumentParser(
        description="AirWare MLP grid search and train model using different CV strategies")
    # "?" one argument consumed from the command line and produced as a single item
    # Positional arguments
    parser.add_argument('-model_strategy',
                        help="Define function to run for MLP",
                        choices=['gridSearch', 'eval_mlp', 'eval_mlp_doppler', 'eval_mlp_ir'],
                        default='eval_mlp')
    parser.add_argument('-cv_strategy',
                        help="Define cross-validation strategy",
                        choices=['loso', 'personalized', 'user_calibrated'],
                        default=None)

    args = parser.parse_args()
    function = function_map[args.model_strategy]
    function(args.cv_strategy)

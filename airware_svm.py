from sklearn.svm import SVC
from sklearn.decomposition import PCA
from utils.baseline_model_helper import *
from utils.generate_report import *
import argparse

MODEL_PATH = "./baseline_models/svm/"
CV_FOLDS = 5


def run_gridSearch_svm(cv_strategy=None):
    start = time.time()
    # Number of principle components for Masked PCA
    n_components_range = [100, 200]
    # C trades off misclassification of training examples against simplicity of the decision surface.
    # Higher C selects more samples as support vectors
    c_range = np.logspace(-3, 3, 7)
    # gamma defines how far the influence of a single training example reaches; low==far.
    # Inverse of the radius of influence of samples selected by the model as support vectors
    gamma_range = np.logspace(-3, 3, 7)
    kernel_options = ['rbf', 'linear']

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
            'classify__C': c_range,
            'classify__kernel': kernel_options,
            'classify__gamma': gamma_range,
            'classify__class_weight': ['balanced']
        }
    ]
    clf_obj = SVC()
    grid_search_best_estimator = gridSearch_clf(x=x, y=y, groups=user, clf=clf_obj, param_grid=param_grid,
                                                file_path=MODEL_PATH)

    print('It took ', time.time() - start, ' seconds.')

    return grid_search_best_estimator


def eval_svm_doppler(cv_strategy='loso'):
    start = time.time()
    svm_clf_params = {'classify__gamma': 1.0,
                      'classify__C': 10,
                      'classify__class_weight': 'balanced',
                      'reduce_dim__n_components': 100}
    pipe = Pipeline([
        ('normalize', StandardScaler()),
        ('reduce_dim', PCA()),
        ('classify', SVC())
    ])
    if cv_strategy == 'loso':
        print("SVM with Leave One Subject CV - Doppler")
        train_clf_doppler(pipe, svm_clf_params, MODEL_PATH + "/doppler/")
        print('It took ', time.time() - start, ' seconds.')
    else:
        raise ValueError("Cross-validation strategy not defined")


def eval_svm_ir(cv_strategy='loso'):
    start = time.time()
    svm_clf_params = {'classify__gamma': 1.0,
                      'classify__C': 10.0,
                      'classify__class_weight': 'balanced'}
    pipe = Pipeline([
        ('normalize', StandardScaler()),
        ('classify', SVC())
    ])
    if cv_strategy == 'loso':
        print("SVM with Leave One Subject CV - IR")
        train_clf_ir(pipe, svm_clf_params, MODEL_PATH + "/ir/")
        print('It took ', time.time() - start, ' seconds.')
    else:
        raise ValueError("Cross-validation strategy not defined")


def eval_svm(cv_strategy='loso'):
    start = time.time()
    svm_clf_params = joblib.load(MODEL_PATH + "clf_gridsearch.pkl")
    pipe = Pipeline([
        ('normalize', StandardScaler()),
        ('reduce_dim', MaskedPCA()),
        ('classify', SVC())
    ])
    if cv_strategy == 'loso':
        print("SVM with Leave One Subject CV")
        train_clf_loso(pipe, svm_clf_params, MODEL_PATH + "leave_one_subject/svm")
        print('It took ', time.time() - start, ' seconds.')
    elif cv_strategy == 'personalized':
        print("SVM with Personalized CV")
        train_clf_personalized(pipe, svm_clf_params, MODEL_PATH + "personalized/svm")
        print('It took ', time.time() - start, ' seconds.')
    elif cv_strategy == 'user_calibrated':
        print("SVM with User Calibrated CV")
        train_size_percent = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        for train_size in train_size_percent:
            train_clf_user_calibrated(pipe, svm_clf_params, train_size,
                                      MODEL_PATH + "user_calibrated/svm" + str(train_size))
        print('It took ', time.time() - start, ' seconds.')
    else:
        raise ValueError("Cross-validation strategy not defined")
if __name__ == '__main__':
    function_map = {'gridSearch': run_gridSearch_svm,
                    'eval_svm': eval_svm,
                    'eval_svm_doppler':eval_svm_doppler,
                    'eval_svm_ir':eval_svm_ir}
    parser = argparse.ArgumentParser(
        description="AirWare SVM grid search and train model using different CV strategies")
    # "?" one argument consumed from the command line and produced as a single item
    # Positional arguments
    parser.add_argument('-model_strategy',
                        help="Define function to run for SVM",
                        choices=['gridSearch', 'eval_svm', 'eval_svm_doppler', 'eval_svm_ir'],
                        default='eval_svm')
    parser.add_argument('-cv_strategy',
                        help="Define cross-validation strategy",
                        choices=['loso', 'personalized', 'user_calibrated'],
                        default=None)

    args = parser.parse_args()
    function = function_map[args.model_strategy]
    function(args.cv_strategy)

import fastsae.sklearn
import sklearn
import sklearn.utils.estimator_checks

if __name__ == "__main__":
    sklearn.utils.estimator_checks.check_estimator(fastsae.sklearn.SAE())

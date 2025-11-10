
# The ``mlght-ml`` framework

## Aim and scope
The aim of this framework is to set up a personal tool to have a "façade" interface to automate common machine learning pipelines, mainly based on the [scikit-learn](https://scikit-learn.org/stable/) library. Its inception was the moment in which I realized I was using the same design patterns across different projects. The natural extension seemed to be the abstraction of these patterns and the construction of a reusable tool. The main source of inspiration for the abstraction patterns in the [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) library.

### The name
`mlght` is a contraction of **m**achine **l**earnin**g** orc**h**estration **t**ool. It *might* also recall a "machine learning *light* automation tool".

## Functionalities
The framework wraps the functionalities of the ``scikit-learn`` API. The objective is to give the user a smooth experience and automatically handle the typical machine learning end-to-end pipeline

1. Data Preprocessing (normalization, scaling, ...).
2. K-Fold cross validation.
3. Model training. 
4. Model testing via trained model inference.
5. Predictions denormalization.

The user does not have to manually normalize and denormalize data. The framework contains all the boilerplate code to automate the steps outlined above. All of these steps are optional. The user might not want to normalize the data or perform K-Fold cross validation. The framework API provides the means to let the user organizing the training and testing procedure according to its needs, with little boilerplate code addition. Note however that project-specific coding is still required. The following essential documentation highlights where these steps might be most needed (and encouraged).

### Consistency with `sklearn.pipeline.Pipeline`
The `mlght`y functionalities are compatible with the flexibility and composability offered by scikit-learn's `Pipeline` object. For my usual work, I prefer to have full control on the processing steps.

# Examples
Thematic notebook examples are available in the [companion repository](https://github.com/MatteoZambra/mlghtml-examples).

# Installation
Do

```bash
git clone https://github.com/MatteoZambra/mlghtml
cd mlghtml
pip -e install .
```

See the needed dependencies in ``requirements.txt``. However, the ``pyproject.toml`` manages the dependencies itself.

# Example
## Code
```python
# main
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.model_selection import train_test_split

from mlght.core.training import Trainer
from mlght.core.scalers import ResettableStandardScaler

from project.src.utils.data import get_data
from project.src.utils import evaluate

# --- Import preprocessed data
# Preprocessing: data cleaning, feature engineering.
# These steps are preminently project-dependent.
# As such, they can hardly be abstracted
X, y = get_data()
Xtrain, ytrain, Xtest, ytest = train_test_split(X, y, test_size = 0.25)

# --- Instantiate the trainer
# `mvreg.training.Trainer` is the core class of the framework.
trainer = Trainer(
    base_estimator    = LinearRegression,            # Regressor, NOT instantiated!
    estimator_name    = "LinearRegression",          # Name of the estimator
    estimator_kwargs  = dict(fit_intercept = True),  # Estimator initialization keywords
    n_cv_splits       = 5,                           # Number of k-fold splits
    n_estimators      = 10,                          # Number of estimators. If > 1, then ensemble learning
    eval_metrics      = {"RMSE": rmse},              # Evaluation metrics
    input_normalizer  = ResettableStandardScaler,    # Input scaler, NOT instantiated!
    target_normalizer = ResettableStandardScaler     # Output scaler, NOT instantiated!
)

# --- Fit
trainer.fit(Xtrain, ytrain)

# --- Predict
# Columns and indices are passed to make `ypred` and `ytest` fully compatile
ypred = trainer.predict(Xtest, ytest.columns, ytest.index)

# --- Case-specific evaluation framework
evaluate(ytest, ypred)
```

That's it.

## Remarks

### Fit and predict methods
The `fit` and `predict` methods fully manage the normalization and denormalization pipeline. The user only has to think about data preparation and features engineering.

The pipeline of the `fit` method is
```python
class Trainer:
    ...

    def fit(self, X, y):
        # Normalize
        X_ = self.normalize(X, self.xscaler, fit = True)
        y_ = self.normalize(y, self.yscaler, fit = True)

        # Fit the model
        # (ensemble automatically managed)
        self.Model.fit_ensemble(X_, y_)
```

The `prediction` pipeline is
```python
class Trainer:
    ...

    def predict(self, X, y, columns, indices):
        # Obtain the denormalized predictions
        y = self.inference(X)

        # Apply any postprocessing needed
        # It could simply be as in the default case
        # That is, assigning columns and indices of 
        # the test set, to ensure consistency
        y = self.postprocessing(y, columns, indices)

        # Return the results
        return y
    
    def inference(self, X):
        # Normalize test set
        X_ = self.normalize(X, self.xscaler, fit = False)

        # Predict (ensemble automatically managed)
        y = self.Model.ensemble_predict(X_)

        # Denormalize
        y_ = self.denormalize(y, self.yscaler)

        # Return the result
        return y_
```

The rationale is:

* `inference` manages the computations: normalization, prediction, denormalization
* `prediction` wraps the computations of `inference` and further applies postprocessing

### Trainer
Almost sure, the ``Trainer`` as illustrated in the minimal example above needs case-specific adjustments. The better alternative is to write a project-specific trainer.

```python
# project.src.models
from mlght.core.training import Trainer as BaseTrainer

class Trainer(BaseTrainer):
    def __init__(
            self,
            ... , # Standard mandatory arguments
            case_specific_args,
            case_specific_kwargs
        ):
        super().__init__(
            ... # Standard mandatory arguments
        )
        ...

    def case_specific_method(self, ...):
        """
        Case-specific method
        """
        ...
```

This gives full control on how the trainer would behave on specific problems. The most notable example is the standard method ``Trainer.postproces``. This method acts on the underlying model output. In the standard case, it simply prepares a ``pandas.DataFrame`` of the predictions. The specific case could require to further elaborate the predictions before returning them for the final evaluation.

In this case, the import statement in ``main.py`` becomes

```python
# main
from project.src.models import Trainer

trainer = Trainer(
    ... # Standard plus custom arguments and kwargs
)
```

### Model
The framework accepts any object that imitates the ``scikit-learn`` API. This object must therefore be endowed with ``fit`` and ``predict`` methods. The trainer orchestrates the training and predicting pipeline. 

Such model can be defined as

```python
# project.src.models
class Model:
    def __init__(self, model, *args, **kwargs):
        self.model = model
        ...

    def fit(self, X, y):
        """
        Fitting logic.
        """
        ...

    def predict(self, X):
        """
        Predict logic.
        """
        ...
        return y

```

### Ensemble learning
Ensemble learning is the default. Under the hood, the trainer class encapsulates the ``Model`` articulated above in a ``mlght.core.bagging_regressor.BaggingRegressor`` object. This object still inherits the ``fit`` and ``predict`` methods as any suitable model would. The bagging regressor object fits a models ensemble and then produces an aggregated result.

To disable ensemble learning, the user needs to set the argument ``n_estimators = 1`` when instantiating the trainer.

### Scalers
The framework passess as default input and output scalers the ``mlght.core.scalers.ResettableStandardScaler`` class. This is crucial since this class

* Inherits from ``sklearn.preprocessing.StandardScaler``, retaining all of its functionalities
* In addition, it implements a ``reset`` method that is used in cross validation and multiple runs methods to avoid any data leakage between runs and multiple normalization, that would invalidate the results.

As now, only the resettable derivatives of ``StandardScaler`` and ``MinMaxScalers`` is implemented. Others to come. The current structure can be used for all the scalers that center and scale data. Robust scaler follows the same rules.

### Data
The most sensitive part of the framework is data preprocessing and postprocessing. It is not an accident that these parts are largely (exclusively) left for the user to write. in the example above, at lines

```python
# main
from project.src.utils.data import get_data
...
X, y = get_data()
```

the `get_data` method is supposed to encapsulate all the logic responsible for

* Data loading
* Type validation, if applicable
* Data cleaning, deduplication, preparation

The `X` and `y` objects are assumed to be ready-to-use data structures. The user is also responsible for ensuring index consistency upon train/test split and columns names management. Should anything in the trainer operations need to know the data columns names, is up to the user save these as trainer's class attributes or relevant methods.

An example could be the implementation of the ``Trainer.postprocess`` method. 

```python
# main
from project.src.utils.data import get_data
from project.src.models import Trainer

X, y = get_data()
trainer = Trainer(
    ... ,
    input_columns = list(X.columns),
    output_columns = list(y.columns)
)
```

```python
# project.src.models
from mlght.core.training import Trainer as BaseTrainer

class Trainer(BaseTrainer):
    def __init__(
            ... , # Same as above
            input_columns,
            output_columns
        ):
        # Same as above
        super().__init__(...)
        self.input_columns = input_columns
        self.output_columns = output_columns

    def postprocessing(self, y)
        """
        Prediction post-processing logic.
        Note that it might also depend on input X for certain cases.
        Example: If the prediction is a input - output difference. In this case one might have to perform 

            output = input - prediction
        """
        input_columns = self.input_columns
        ...
        return y
```

# Contribution
This project is still a draft. Contributions not provisioned in the near future. However, the interested user may still contact the authors for any clarification and/or suggestion.

# Further work
- [ ] Units testing.
- [ ] Extend the ``mlght.core.scalers`` classes to include, at least, the resettable version of ``sklearn.preprocessing.RobustScaler``, and others.
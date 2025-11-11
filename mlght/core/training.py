
import os
# import pickle
import joblib
from tabulate import tabulate
from tqdm import tqdm

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import root_mean_squared_error

from mlght.core import bagging_regressor

__all__ = [
    "Trainer"
]


class Trainer:
    def __init__(
            self,
            base_estimator,
            estimator_name,
            estimator_kwargs,
            n_cv_splits = None,
            n_estimators = 1,
            eval_metrics = {"RMSE": root_mean_squared_error},
            input_normalizer = None,
            target_normalizer = None,
            pbar_module = tqdm,
            verbose = False
        ):
        """
        Initialize the trainer "façade" class. Under the hood, this constructor
        initializes the ``mlght.core.bagging_regressor`` class. This latter
        hosts the ensemble of base estimators selected and performs ensemble
        learning and prediction. The ``Trainer`` interface is endowed with its
        interface methods, so the user simply calls the methods of ``Trainer``
        as wrappers for the underlying ``bagging_regressor`` functionalities.
        
        **NOTE**: If the user does *not* want to perform ensemble learning, him
        sufficies to set the keyword ``n_estimators`` to 1.
        
        Parameters
        ----------
        base_estimator : object
            The base estimator. It can be any legitimate that inherits, or overrides,
            the methods of the ``sklearn.base.RegressorMixin`` class. Note: it
            can even be a PyTorch ``torch.nn.Module`` object, provided that it
            has ``fit`` and ``predict`` methods that mimick the scikit-learn
            interface methods.
        estimator_name : str
            Name of the estimator.
        estimator_kwargs : dict, optional
            Dictionary of keywords used to initialize the estimator. Note that
            the types are heterogeneous, since each model can be initialized
            by simple int, str, bool keywords as well as types derived from
            the PyTorch ``torch.nn`` or ``sklearn.gaussian_process.Kernel`` types.
            The default is ``dict()``.
        n_cv_splits : int, optional
            The number of cross-validation folds to use for the k-folds 
            cross validation protocol. The default is None. If None, then 
            the ``run_kfold_cv`` is skipped.
        n_estimators : int, optional
            Number of estimators to use in the ensemble learning protocol.
            The default is 1.
        eval_metrics : dict[str, callable], default
            Provides the evaluation metrics used to assess the model performance.
            The dictionary structure allows to inform the model about which metric
            to use and the name of each metric, to display a tidy metrics table.
            The default is ``{"RMSE": sklearn.metrics.root_mean_squared_error}``.
        input_normalizer : object, optional
            Scaler to preprocess input data. This object must be one of the custom
            resettable scalers implemented in ``mlght.core.scalers``.
            The default is None. In this case, data are not normalized.
        target_normalizer : object, optional
            Scaler to preprocess the output data. The default is None.
            The same considerations as for ``input_normalizer`` apply.
        pbar_module : tqdm.std.tqdm, optional
            The progress bar. In the notebook, it would be ``tqdm.notebook``.
            The default is ``tqdm.std.tqdm``.
        verbose : bool, optional
            Whether to display progress bars and output messages. The default
            is False.
        
        Returns
        -------
        None.
        
        """
        # Model
        self.n_cv_splits = n_cv_splits
        self.estimator_name = estimator_name
        self._Model = bagging_regressor.BaggingRegressor(
            base_estimator,
            estimator_name,
            estimator_kwargs,
            n_estimators,
            pbar_module
        )
        
        # Aesthetics
        self.pbar_module = pbar_module
        self.verbose = verbose

        # Evaluation
        self.eval_metrics = eval_metrics
        
        # Normalization pipeline
        self._init_normalizers(input_normalizer, target_normalizer)

        # Class attributes
        self.can_be_saved = False
    #end

    def _logging(self, msg):
        """
        Logger method. If the ``verbose`` attribute is set to True,
        then this method prints on screen messages for the user.

        Parameters
        ----------
        msg : str
            The message to display.
        
        Returns
        -------
        None.

        """
        if self.verbose:
            print(msg)

    def get_models(self, ensemble = True):
        """
        Get the trained models ensemble.

        Parameters
        ----------
        ensemble : bool, optional
            If True, a list containing the instances of 
            trained models is returned. Otherwise, the 
            method returns the ``mlght.core.bagging_regressor.BaggingRegressor``
            instance. The default is True.
        
        Returns
        -------
        any
            If ``ensemble`` is True, the list of trained
            ``base_estimator`` instances. Otherwise, the 
            ``mlght.core.bagging_regressor.BaggingRegressor``
            instance itself.
        """
        if ensemble:
            return self._Model.get_models_ensemble()
        else:
            return self._Model

    def _init_normalizers(
            self,
            input_normalizer,
            target_normalizer
        ):
        """
        Initialize the normalization operators for either input and/or target.
        
        Parameters
        ----------
        input_normalizer : object
            The ``input_normalizer`` as provided to the constructor.
        target_normalizer : object
            The ``target_normalizer`` as provided to the constructor.
        
        Returns
        -------
        None.
        
        """
        if not input_normalizer:
            self._logging("No normalization selected for input. Not recommended!")
            self.xscaler = None
        else:
            self._logging(f"Input normalizer: {input_normalizer.__qualname__}")
            self.xscaler = input_normalizer()
        
        if not target_normalizer:
            self._logging("No normalization selected for target. Not recommended!")
            self.yscaler = None
        else:
            self._logging(f"Target normalizer: {target_normalizer.__qualname__}")
            self.yscaler = target_normalizer()
    #end
    
    def _reset_normalizers(self):
        """
        Some custom scalers implement a reset method, to flush the scaling 
        parameters learnt as of ``scaler.fit(X)``. This is useful in the 
        ensemble learning and in the cross-validation protocols, where a loop
        is performed over multiple dataset folds and/or base estimators. 
        This is done to prevent information leaking.
        
        Parameters
        ----------
        None.
        
        Returns
        -------
        None.
        
        """
        try:
            self.xscaler.reset()
        except AttributeError:
            self._logging("`xscaler` is probabilty set to `None`. Skipping scaler reset ...")
            pass
        
        try:
            self.yscaler.reset()
        except AttributeError:
            self._logging("`xscaler` is probabilty set to `None`. Skipping scaler reset ...")
            pass
    #end
    
    def _normalize(self, data, scaler, fit, set_index = True, return_data = True):
        """
        Normalize data. That ``set_index`` is crucial. If it is set to 
        ``False``, there may be inconsistencies in dataframes comparisons.
        
        Parameters
        ----------
        data : numpy.ndarray | pandas.DataFrame
            The input data. Note that according to the type of ``data``, this
            method behaves differently. For arrays, only the computation is 
            performed. For data frames, the computation is completed by the
            creation of a new data frame, with the original columns **and**
            indices. Note how important the indices are. Read the synopsis of 
            this method.
        scaler : object
            The transformer (one of the two, since this method is called for
            inputs and outputs separately) set in the constructor initialization.
        fit : bool
            Whether to fit the transformer or to transform the data using the 
            pretrained scaler.
        set_index : bool, optional
            This can be neglected for downstream inference, postprocessing or
            analyses. Within the scope of (ensemble) learning in the ``Trainer``
            interface, the index information may be prominent.
        return data : bool, optional
            Whether to return data after method call of not.
        
        Returns
        -------
        data_scaled : numpy.ndarray | pandas.DataFrame
            As for the input, an inpuy array is simply processed by computation
            while a data frame is subsequently re-wrapped by a data frame.
        None
            If ``return_data`` is set to ``False``.
        
        """
        # If the user did not specify a scaler - that is, no normalization -,
        # then he's warned, and data are returned as provided as argument
        if not scaler:
            self._logging("Warning in `normalize`: scaler is `None`. Skipping ...")
            return data
        
        # In scaler fitting stage, we point at the fit_transform method
        # If we are to normalize the test set, we only transform it, with
        # the scaling rules learnt from the training set, as Christ commands
        transform_fn = (
            scaler.fit_transform if fit else scaler.transform
        )
        if fit:
            scaler.is_fit = True
        
        # Normalization
        # NOTE: The input may be either be numpy.ndarray or pandas.DataFrame
        data_scaled = transform_fn(data)
        if isinstance(data, pd.DataFrame):
            data_scaled = pd.DataFrame(
                data_scaled,
                columns = data.columns
            )
            
            # Setting dataframe index
            if set_index:
                data_scaled.index = data.index
        
        # Set a flag to notify that data is scaled
        data_scaled.is_scaled = True

        if return_data:
            return data_scaled
    #end
    
    def _denormalize(self, data, scaler):
        """
        Denormalize data. Setting the index of denormalized data to 
        the index of input data is mandatory in this case. Note that if
        the ``Trainer`` has not been initialized with a scaler, the 
        data return as they were given. The user is warned with a message.
        
        Parameters
        ----------
        data : numpy.ndarray | pandas.DataFrame
            The data to denormalize according to the scaler passed.
        scaler : object
            The scaler selected by the user at initialization time.
        
        Returns
        -------
        data_denormalized : numpy.ndarray | pandas.DataFrame
            The unscaled data. According to the type of the input parameter,
            so it will be the type of the output.
        
        """
        # Warn non-normalized data
        if not scaler:
            self._logging("Warning in `denormalize`: scaler is `None`. Skipping ... ")
            return data
        
        # Check that scaler is fitter. Raises error if not
        scaler.check_fitted()
        
        # Otherwise, denormalize
        if len(data.shape) == 1:
            data = data[..., np.newaxis]
        data_denormalized = scaler.inverse_transform(data)
        
        if isinstance(data, pd.DataFrame):
            data_denormalized = pd.DataFrame(
                data = data_denormalized,
                columns = data.columns,
                index = data.index
            )

        # Set a flag to signal that data is unscaled
        # data_denormalized.is_scaled = False

        return data_denormalized
    #end

    def _make_cv_splits(self, X, y):
        r"""
        Split the dataset in train and validation folds.
        
        Note: This `Trainer` class is supposed to manage only the train
        partition. The train/test partition must have been performed 
        before this class is even instantiated. Or at least, the `run`
        entry method expects only the train
        :math:`\left\{(X_i, \; y_i)\right\}_{i = 1}^{N_{\mathrm{train}}}` pairs.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Inputs.
        y : pandas.DataFrame
            Outputs.
        
        Returns
        -------
        splits : generator
            The iterator with the k cross-validation folds.
        
        """
        # Check that data are unscaled
        self._check_dataframe_is_scaled(X, "X", "make_cv_splits")
        self._check_dataframe_is_scaled(y, "y", "make_cv_splits")

        if not self.n_cv_splits:
            self._logging("Number of cv splits unset. K-Fold cross-validation skipped ...")
            return None
        
        if self.n_cv_splits == 1:
            self._logging(
                "Set number of cv splits equals 1. Casting to 2 for safety"
                "Note: Run a cross-validation training (splits = 5 to 10) at least once"
                "in order to see whether the model setup is acceptable."
            )
            n_splits = 2
        else:
            n_splits = self.n_cv_splits
        
        X = X.reset_index(drop = True)
        y = y.reset_index(drop = True)

        cv_splits = KFold(n_splits = n_splits).split(X, y)
        splits = [
            (
                (X.loc[split[0]], y.loc[split[0]]), # Training split
                (X.loc[split[1]], y.loc[split[1]])  # Validation split
            )
            for split in cv_splits
        ]
        
        return splits
    #end
    
    def run_kfold_cv(self, X, y):
        r"""
        Run the main KFold cross-validation loop. In the stage of hyperparameters
        tuning, we may want to test on the k-folded dataset. This is not necessary 
        once we found the optimal set of hyperparameters. In this case, we train
        the model on the whole training set.
        
        Note: The :math:`\left\{(X_i, y_i)\right\}_{i = 1}^{N_{\mathrm{train}}}`
        couple is the train set. The train/test partition is supposed to have
        been performed before the instantiation of the `Trainer` class.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Input data.
        y : pandas.DataFrame
            Output data.
        
        Returns
        -------
        None.
        
        """
        # Initialize evaluation metrics list
        cv_foldwise_metrics = []

        # Instantiate the progress bar
        pbar_runs = self.pbar_module(
            total = self.n_cv_splits, 
            colour = "green",
            position = 0, leave = True
        )
        
        # Loop over folds
        cv_splits = self._make_cv_splits(X, y)
        if not cv_splits:
            return None
        
        for run, ((X_train, y_train), (X_val, y_val)) in enumerate(cv_splits):
            pbar_runs.set_description(f"Fold {run+1}")
            pbar_runs.update()
            
            # Train the ensemble model
            # (Normalization inside fit)
            self.fit(X_train, y_train)
            
            # Get predictions and denormalize the aggregated prediction
            # (Normalization inside inference)
            y_pred = self.predict(
                X_val,
                columns = y_val.columns,
                index = y_val.index
            )
            
            # Postprocess y_val
            # (Train/test split reset indices!)
            y_val = self.postprocess(
                X_val, y_val,
                y_val.columns,
                y_val.index
            )
            
            # Record performance of this fold
            metrics = self.evaluate(y_val, y_pred, print_metrics = False)
            cv_foldwise_metrics.append(metrics)
            
            # Reset scalers, will be re-fit in the next loop
            self._reset_normalizers()
        #end
        
        # Evaluate average performance level across the folds
        # Delete kfold-associated state afterwards
        self.cross_valid_evaluation(cv_foldwise_metrics, print_metrics = True)
        cv_foldwise_metrics[:] = []
        
        # Flush models ensemble. In the next stage, that is, proper fitting,
        # we will re-train the models ensemble and we'll get proper scaling params.        
        self._Model.flush_models()
    #end

    def split_data(self, X, y, test_size = 0.25):
        """
        Split input data in train/[test, val].

        Parameters
        ----------
        X : pandas.DataFrame
            Input data.
        y : pandas.DataFrame
            Output targets.
        test_size : float
            The fraction of data to hold out for testing.
        
        Returns
        -------
        X : pandas.DataFrame
            Training input data.
        y : pandas.DataFrame
            Training output target.
        Xtest : pandas.DataFrame
            Testing input data.
        ytest : pandas.DataFrame
            Testing output targets.
        
        """
        # Check that data are unscaled
        self._check_dataframe_is_scaled(X, "X", "split_data")
        self._check_dataframe_is_scaled(y, "y", "split_data")

        # Data split with the scikit-learn utility
        X, Xtest, y, ytest = train_test_split(
            X, y, test_size = test_size, shuffle = True
        )
        
        # IMPORTANT: To reset indices!
        ## Training set
        X = X.reset_index(drop = True)
        y = y.reset_index(drop = True)

        # Testing set
        Xtest = Xtest.reset_index(drop = True)
        ytest = ytest.reset_index(drop = True)
        
        return X, y, Xtest, ytest
    #end
    
    def n_runs_assessment(
            self,
            X,
            y,
            runs = 10
        ):
        r"""
        After the model is fit, the hyperparameters have been chosen, the 
        model can be tested in two ways:
            1. With ``self.inference`` and then to visualize the results
               by means of the associated functions in ``src.utils.beauty``
            2. Do ``n`` runs to test random subsampling and compute the average
               plus/minus std. This is a more robust estimate
        
        **NOTE**: ``X`` and ``y`` must NOT be normalized.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Input data.
        y : pandas.DataFrame
            Output data.
        runs : int, optional
            Number of assessment runs to perform. Default is 10.
        
        Returns
        -------
        None.
        
        """
        # Initialize performances data structure
        n_assessments_metrics = []
        
        # Perform the runs
        for run in self.pbar_module(range(runs), desc = "Runs"):
            # Split data, if the case
            Xtrain, ytrain, Xtest, ytest = (
                self
                .split_data(X, y)
            )
            
            # Fit model. Normalization happens inside the fit method
            self.fit(Xtrain, ytrain)
            
            # Inference on test set. Normalization inside the method
            ypred = self.predict(
                Xtest,
                columns = ytest.columns,
                index = ytest.index
            )
            
            # Postprocess: Re-assign columns and indices
            # (Train/test split reset indices!)
            ytest = self.postprocess(
                Xtest, ytest,
                columns = ytest.columns, index = ytest.index
            )

            # Save the metrics
            n_assessments_metrics.append(self.evaluate(ytest, ypred, print_metrics = False))
            
            # Flush the models and scalers params
            self._Model.flush_models()
            self._reset_normalizers()
        #end
        
        # Aggregate metrics and display the performance statistics
        self.cross_valid_evaluation(n_assessments_metrics, print_metrics = True)
    #end

    def _check_scaler_is_fit(self, scaler, method):
        """
        Check whether the passed scaler is fitted yet.

        Parameters
        ----------
        scaler : object
            The scaler used.
        method : str
            The method that calls this function. For logging.
        
        Raises
        ------
        RuntimeError
            If the scaler is fitted yet. This is determined
            based on the state of the scaler object.
        
        Returns
        -------
        None.

        """
        if scaler and scaler.is_fit:
            raise RuntimeError(f"In `{method}`: {scaler} is fitted yet!")
    #end
    
    def _check_dataframe_is_scaled(self, dataframe, name, method):
        """
        Check if the data is scaled yet. Note: this method relies 
        on a flag attribute manually attached to the dataframe.
        While brittle, it is effective in the context of this
        framework implementation.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            The dataframe to test.
        name : str
            The name of the dataframe. For logging.
        method : str
            The function that calls this method. For logging.
        
        Raises
        ------
        RuntimeError
            Thrown if the dataset is scaled yet. In this case, 
            the program is stopped.
        AttributeError
            Thrown if the dataframe does not have the flag
            attribute signaling that is has been scaled. In
            this case, data are indeed unscaled, so the method
            does not stop the program.
        
        Returns
        -------
        None.

        """
        try:
            if dataframe.is_scaled:
                raise RuntimeError(f"Object {name} is scaled!\n"
                                   f"Pass unscaled data to `{method}`")
        except AttributeError:
            # If an AttributeError occurs, it is because the flag is_scaled
            # has not been set to the dataframe. Hence no need to worry.
            # The data in the dataframe is for sure not scaled
            pass
    #end
    
    def fit(self, X, y):
        """
        Method to call the underlying bagging regressor ensemble fit function.
        
        **IMPORTANT**: ``X`` and ``y`` must **not** be normalized.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Input data. Training set.
        y : pandas.DataFrame
            Output data (ground truths). Training set.
        
        Returns
        -------
        None.
        
        """
        # Check that scalers are not fit
        # Otherwise, data are normalized twice
        self._check_scaler_is_fit(self.xscaler, "fit")
        self._check_scaler_is_fit(self.yscaler, "fit")

        # Check that dataframes (if so) do not have 
        # the flag that signals they are scaled
        self._check_dataframe_is_scaled(X, "X", "fit")
        self._check_dataframe_is_scaled(y, "y", "fit")

        # Normalize
        X = self._normalize(X, self.xscaler, fit = True)
        y = self._normalize(y, self.yscaler, fit = True)
        
        # Fit the model ensemble
        self._Model.fit_ensemble(X, y)
    #end
    
    def inference(self, X):
        """
        Method to call the underlying bagging regressor
        ensemble prediction function.
        
        **IMPORTANT**: ``X`` must **not** be normalized!

        ```python
        # Pipeline
        X = normalize(X)
        y = model(X)
        y = denormalize(y)
        ```
        
        Parameters
        ----------
        X : pandas.DataFrame
            Input data. Test set.
        
        Returns
        -------
        y_pred : pandas.DataFrame
            Predicted data.
        
        """
        # Check that dataframe X (if so) is not scaled
        self._check_dataframe_is_scaled(X, "X", "inference")

        # Normalize
        X_ = self._normalize(X, self.xscaler, fit = False)

        # Ensemble prediction
        y_pred = self._Model.ensemble_predict(X_)

        # Prediction denormalization
        y_pred = self._denormalize(y_pred, self.yscaler)

        # Return
        return y_pred
    #end
    
    def postprocess(self, X, y, columns, index, *args, **kwargs):
        """
        Generic postprocessing. It is to be further implemented and detailed in
        the specific application case.
        
        Parameters
        ----------
        X : numpy.ndarray | pandas.DataFrame
            Input data. Likely the test set.
        y : numpy.ndarray | pandas.DataFrame
            The intermediately-elaborated predictions.
        columns : list of string
            The columns to set to the final data frame.
        index : pandas.Index
            The indices to set to the final data frame.
        args : iterable
            Additional positional arguments. Useful as
            this method will be overridden by derivate
            classes in case-specific context.
        kwargs : iterable
            Additional keyword arguments. Useful at
            derivate instantiation stage.
        
        Returns
        -------
        y_ : pandas.DataFrame
            The final data frame produced. This is the prediction ready to
            be used in the downstream analyses.
        
        """
        # Prepare the dataframe with columns and indices
        y_ = pd.DataFrame(
            data = y,
            columns = columns,
            index = index
        )
        
        # Return
        return y_
    #end
    
    def predict(self, X, columns, index):
        """
        Method that wraps prediction and postprocessing in one place. This
        is an interface method that exposes to the user a simple command to
        perform both the prediction and final data frame preparation operations.
        
        **IMPORTANT**: ``X`` must **not** be normalized. Normalization done here.

        ```python
        # Pipeline
        y = inference(X)
        y = postprocess(X)
        ```
        
        Parameters
        ----------
        X : numpy.ndarray | pandas.DataFrame
            Input data. Likely a test set.
        columns : list of string
            Columns to set to the final data frame.
        index : pandas.Index
            Indices to set to the final data frame.
        
        Returns
        -------
        y_pred : pandas.DataFrame
            The final data frame. Amenable to downstream analyses.
        
        """
        # Check that the dataframe X (if so) is not scaled
        self._check_dataframe_is_scaled(X, "X", "predict")
        
        # The full inference pipeline
        y_pred = self.inference(X)

        # Postprocess (After denormalization occurred in predict!)
        y_pred = self.postprocess(X, y_pred, columns, index)

        # Ship the results
        return y_pred
    #end
    
    def prepare_for_shipping(self, X, y):
        """
        Once cross-validation is done, n-runs assessment to check that the
        statistics of performance are stationary, then it is needed to prepare
        a definitive compact version of the model. 
        
        This method saves the model ensemble as well as the scaling rules 
        parameters, in order to prepare it for shipping.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Input data. Complete data set.
        y : pandas.DataFrame
            Output data (ground truths). Complete set.
        
        Returns
        -------
        None.
        
        """
        # 0. First: Flush anything related to scalers and models
        self.xscaler.reset()
        self.yscaler.reset()
        self._Model.flush_models()

        self.data_details = {
            "input_columns"  : list(X.columns),
            "target_columns" : list(y.columns)
        }

        # 1. Normalization: extrapolate scaling rules parameters
        X_ = self._normalize(X, self.xscaler, fit = True)
        y_ = self._normalize(y, self.yscaler, fit = True)

        # 2. Fit the ensemble
        self._Model.fit_ensemble(X_, y_)

        # Done!
        self._logging("\n✅ Scalers and Model trained!"
                      "\n🛠️ Preparing for shipping ...")
        
        # 3. The trainer is ready
        # Set the flag informing that the trainer can be saved.
        # At this point, the models are trained, the scalers are fit
        # and the data details have been stored as attributes.
        self.can_be_saved = True
    #end

    def save_trainer(
            self,
            path,
            trainer_name = "trainer_artifacts.joblib",
            overwrite = False
        ):
        """
        Save the complete trainer class as pickle-serialized object.
        Note: this strategy is rather brittle. Provisional: use
        the ``joblib`` library instead.

        Parameters
        ----------
        path : str
            Path to save the trainer.
        trainer_name : str, optional
            The name to call the saved trainer with. The
            default is ``trainer.pkl``.
        
        Returns
        -------
        None.

        """
        # Sanity check
        if not self.can_be_saved:
            raise RuntimeError("The trainer can not be saved. "
                               "Test it with cross-val, perform multiple runs. "
                               "Once that done, run `trainer.prepare_for_shipping(*)`. "
                               "Then, you can save the model.")

        # Prepare the file system, if needed
        path = os.path.join(path, self.estimator_name.replace(" ", ""))
        if not os.path.exists(path):
            os.makedirs(path)
            self._logging(f"Created path to save the trainer:\n{path}")

        # Prepare the file, file path details
        if not trainer_name.endswith(".joblib"):
            trainer_name += ".joblib"
        file_path = os.path.join(path, trainer_name)

        # Check if overwrite
        if os.path.exists(file_path) and not overwrite:
            self._logging("No overwrite set for saving. Returing ...")
            return

        # And save the model
        with open(file_path, "wb") as f:
            joblib.dump(self, f)
        #end

        self._logging(f"\nSaved artifacts to {file_path}")
    #end

    def evaluate(self, ytest, ypred, print_metrics, eval_metrics = None, phase = None):
        r"""
        Intra-fold performance evaluation. 
        
        Parameters
        ----------
        ytest : pandas.DataFrame
            Ground truth.
        ypred : pandas.DataFrame
            Predictions.
        print_metrics : bool
            Whether to print metrics table.
        eval_metrics : dict[str, callable], default
            The evaluation metrics to use. If not
            passed, then the method uses the metrics
            passed at class instantiation. 
            The default is None.
        phase : str, optional
            The evaluation phase. Default is None.
        
        Returns
        -------
        metrics : pandas.DataFrame
            The metrics table.
        
        """
        # Get metrics
        if not eval_metrics:
            eval_metrics = self.eval_metrics
        
        # Prepare columns and test/pred vectors
        columns = list(ytest.columns)
        ytest_ = np.asarray(ytest).copy()
        ypred_ = np.asarray(ypred).copy()

        # Create the evaluation table
        metrics = (
            pd
            .DataFrame(
                {
                    kmetric : [
                        vmetric(ytest_[:,i], ypred_[:,i])
                        for i in range(len(columns))
                    ]
                    for kmetric, vmetric
                    in eval_metrics.items()
                },
                index = columns
            )
            .T
        )

        # Metrics printout
        if print_metrics:
            print(f"\n------------\nEvaluation: {'' if not phase else phase}\n")
            print(
                tabulate(
                    metrics,
                    headers = "keys",
                    tablefmt = "grid"
                )
            )
        
        return metrics
    #end
    
    def cross_valid_evaluation(self, metrics, print_metrics = True):
        """
        The same. This function serves the purpose of collecting all 
        the folds-related metrics and to produce a table with the 
        inter-folds averages.
        
        Parameters
        ----------
        metrics : list of pandas.DataFrame
            A list with the metrics tables for each run.
        print_metrics : bool, optional
            Whether to print the aggregated metrics.
        
        Returns
        -------
        None.
        
        """
        # Aggregate metrics
        columns = list(metrics[0].columns)

        metrics_stats = []
        for metric in self.eval_metrics.keys():
            # Append the average of the metric
            metrics_stats.append(
                pd.concat(
                    [
                        df.loc[metric].to_frame().T
                        for df in metrics
                    ]
                )
                .mean(axis = 0)
                .to_frame()
                .T
                .values
                .flatten()
            )
            # Append the standard deviation of the metric
            metrics_stats.append(
                pd.concat(
                    [
                        df.loc[metric].to_frame().T
                        for df in metrics
                    ]
                )
                .std(axis = 0)
                .to_frame()
                .T
                .values
                .flatten()
            )
        
        metrics_table = pd.DataFrame(
            metrics_stats,
            columns = columns,
            index = [
                f"{stat} {metric}"
                for metric in self.eval_metrics.keys()
                for stat in ["Avg", "Std"]
            ]
        )
        
        # Disarticolate the statistics (avg or std) and the metric (RMSE, MAE)
        metrics_melted = metrics_table.copy()
        metrics_melted[["Stat", "Metric"]] = (
            metrics_melted
            .reset_index()
            .rename(
                columns = {"index" : "Metric"}
            )["Metric"]
            .str
            .split(" ")
            .str[:2]
            .to_list()
        )
        
        # Pivot table: associate to each coordinate (L, a, b) its statistics
        # To have: {Coord: avg \pm std}
        metrics_pivot = pd.pivot(
            metrics_melted,
            values = columns,
            columns = ["Stat"],
            index = ["Metric"]
        )
        metrics_pivot = metrics_pivot.reset_index(level = 0)
        metrics_pivot.columns = [
            (c[0] + " " + c[1]).strip() for c in metrics_pivot.columns
        ]
        metrics_pivot = metrics_pivot.set_index("Metric")
        
        metrics_table_print = (
            metrics_pivot
            .assign(
                **{
                    c: metrics_pivot[c + " Avg"].apply(lambda x: f"{x:.4f}") + " ± " +
                       metrics_pivot[c + " Std"].apply(lambda x: f"{x:.4f}")
                       for c in columns
                }
            )
        )
        metrics_table_print = metrics_table_print[columns]
        
        # Print metrics
        if print_metrics:
            print(
                tabulate(
                    metrics_table_print,
                    headers = "keys",
                    tablefmt = "grid"
                )
            )
        
        # Save tables
        self.average_metrics = metrics
        self.average_std_metrics = metrics_pivot
    #end
#end

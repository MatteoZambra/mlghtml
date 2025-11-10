
import numpy as np
import sklearn

__all__ = [
    "BaggingRegressor"
]


class BaggingRegressor:
    def __init__(
            self,
            base_estimator,
            estimator_name,
            estimator_kwargs,
            n_estimators,
            pbar_module,
            test_split_size = 0.33
        ) -> None:
        """
        Initialize the bagging regressor class.

        Parameters
        ----------
        base_estimator : object
            The uninstantiated estimator class. It **must**
            inherit the ``fit`` and ``predict`` methods to
            mimick the ``sklearn.base.BaseEstimator`` class.
        estimator_name : str
            The name of the estimator. Informal, to keep track
            of the experiment details. Almost useless, can be 
            removed.
        estimator_kwargs : dict[str, object]
            the keyword arguments to initialize the estimator.
        n_estimators : int
            The number of estimators of the ensemble.
        pbar_module : object
            The module to display the progress bars.
        test_split_size : float
            The test set size. Used to split data during 
            ensemble learning.
        
        Returns
        -------
        None.
        
        """
        self.base_estimator = base_estimator
        self.estimator_name = estimator_name
        
        self.pbar_module = pbar_module
        self.estimator_kwargs = estimator_kwargs
        self.n_estimators = n_estimators
        self.test_split_size = test_split_size
        self.models_ensemble = []
    #end

    def get_models_ensemble(self):
        """
        Returns the ensemble of trained models.

        Raises
        ------
        AttributeError
            If the model ensemble is not initialized or trained.
        
        Returns
        -------
        self.models_ensemble : list[object]
            The list of trainer ensemble members.
        
        """
        if self.models_ensemble:
            return self.models_ensemble
        else:
            raise AttributeError("Models ensemble not trained. "
                                 "Cannot provide them. Train beforehand!")
    
    def set_models_ensemble(self, trained_ensemble):
        """
        Sets a list of trained models.

        Parameters
        ----------
        trained_ensemble : list[obj]
            The list of fitted estimators.
        
        Returns
        ------
        None.

        """
        self.models_ensemble = trained_ensemble
    
    def save_trained_ensemble(self):
        """
        Deprecated for now.
        """
        raise NotImplementedError("This method is temporarily deprecated.")
    
    def load_trained_ensemble(self):
        """
        Deprecated for now.
        """
        raise NotImplementedError("This method is temporarily deprecated.")
    
    def get_train_test_sets(self, X, y):
        """
        If we do not perform ensemble learning (``n_estimators == 1``), then
        the whole training set is used. Otherwise, each estimator is trained
        on a randomly subsample of the training set. These subsets can 
        overlap, indeed no random seed is set in the function
        ``skearn.model_selection.train_test_split``.

        Parameters
        ----------
        X : pandas.DataFrame
            Input data.
        y : pandas.DataFrame
            Targets.
        
        Returns
        -------
        Xtrain : pandas.DataFrame
            Training set.
        ytrain : pandas.DataFrame
            Training targets.
        
        """
        if self.n_estimators > 1:
            X_train, _, y_train, _ = sklearn.model_selection.train_test_split(
                X, y, test_size = self.test_split_size
            )
        else:
            X_train, y_train = X.copy(), y.copy()
        
        return X_train, y_train
    #end
    
    def flush_models(self):
        """
        If there are previously trained models in the ensemble, 
        this method flushes the list.

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        self.models_ensemble[:] = []
    #end
    
    def save_ensemble_member(self, model):
        """
        Save a trained ensemble member in the ensemble models list.
        
        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        self.models_ensemble.append(model)
    #end
    
    def fit_ensemble(self, X, y):
        """
        Fit the ``self.n_estimators`` ensemble.
        
        NOTE: :math:`X` and :math:`y` are the training set input and targets
        associated with the :math:`k`-the cross validation split, if the
        training scheme accounts for it. Otherwise, it is the training set,
        complete as obtained by the data preparation procedure.
        
        **NOTE (Important)**: This class assumes that both X and y have been
        *normalized* already! Which, if the case, should have been happened
        in the ``mlght.core.training.Trainer`` class

        Parameters
        ----------
        X : pandas.DataFrame
            Input data.
        y : pandas.DataFrame
            Targets.
        
        Returns
        -------
        None.

        """
        # Initialize progress bar
        pbar_estimators = self.pbar_module(
            total = self.n_estimators, 
            colour = "white",
            position = 1, leave = False
        )
        for n_member in range(self.n_estimators):
            pbar_estimators.set_description(f"Estimator: {n_member+1}")
            pbar_estimators.update()
            
            # Instantiate the ensemble member
            this_model = self.base_estimator(**self.estimator_kwargs)
            
            # Sample part of the training dataset
            X_train, y_train = self.get_train_test_sets(X, y)
            
            # Fit the model and save it
            this_model.fit(X_train, y_train)
            self.save_ensemble_member(this_model)
            
            # Delete current model - for memory efficiency
            del this_model
        #end
    #end
    
    def ensemble_predict(self, X, aggregation_func = np.median):
        """
        Predict. The ensemble members have been fitted yet.
        Note that, if the user does not want to learn an ensemble, 
        and passes n_estimators = 1, or does not pass anything at all, 
        then the ``self.models_ensemble`` only contains one member.

        Parameters
        ----------
        X : pandas.DataFrame
            Input data.
        aggregation_func : callable, optional
            The aggregation function to aggregate the model outputs.
            The default is ``numpy.median``.
        
        Returns
        -------
        predictions : numpy.ndarray
            The aggregated predictions.
        
        """
        predictions = []
        for model in self.models_ensemble:
            y_pred = model.predict(X)
            predictions.append(y_pred)
        
        predictions = aggregation_func(np.array(predictions), axis = 0)
        return predictions
    #end

    def get_model_params(self, asarray = False):
        """
        Method to retrieve the models ensemble trained parameters.

        Parameters
        ----------
        asarray : bool, optional
            Whether to make an array of the outputs. In this case, the
            parameters are returned as lists of ``numpy.ndarrays``, which
            are the individual coeffcients. The default is
            False. In this case, the parameters are returned as list 
            of ``numpy.ndarray``. That is, the coefficients are returned
            as vector and not as list.
        
        Returns
        -------
        coefficients, intercepts : tuple[list[numpy.ndarray] | numpy.ndarray]
            The ensemble model parameters.
        """
        if self.models_ensemble:
            coefficients = [m.coef_ for m in self.models_ensemble]
            intercepts = [m.intercept_ for m in self.models_ensemble]

            if asarray:
                # Do this to ensure that coefficients and intercept
                # are returned as arrays of shape
                #   * (n_estimators, n_features) for coefficients
                #   * (n_estimators, n_target) for the intercept
                coefficients = (
                    np.array(coefficients)
                    .reshape(len(coefficients), -1)
                )
                intercepts = (
                    np.array([intercepts])
                    .reshape(len(intercepts), -1)
                )
            
            return coefficients, intercepts
    #end
#end




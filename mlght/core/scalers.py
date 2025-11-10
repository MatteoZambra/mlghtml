
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

__all__ = [
    "ResettableStandardScaler",
    "ResettableMinMaxScaler"
]


class BaseResettableScaler:
    """
    Base class for the ResettableScaler derivates.

    This class is set to be the parent, alongside with 
    the technique-specific scaler from ```sklearn.preprocessing``
    classes, for the resettable scalers defined below. 

    Rationale
    ---------
    In the cases of 

    * K-fold cross validation
    * Multiple run evaluation

    the scaler is called to preprocess diverse data split. E.g.
    the first fold is normalized and a model is fit on it. The
    second fold can not use the same scaling rules as the previous one.
    This would result is a severe data leakage, thus compromising 
    the reliability of the results. The resettable scaler allows
    to flush the scaling rules (scaler's attributes) as the fold,
    or the model run, is over. In this way, the next fold (or run)
    can start with a fresh scaler. This prevents data leakage.

    """
    def __init__(self):
        """
        Initialize.

        Returns
        -------
        None.

        """
        super().__init__()

    def check_fitted(self):
        """
        Check if the scaler is fit.

        Raises
        ------
        TypeError
            If the scaler has not been instantiated.
        NotFittedError
            If the scaler is asked to reset but it has not been fit.

        Returns
        -------
        None.

        """
        try:
            check_is_fitted(self)
            return True
        except TypeError as e:
            print("Scaler is not instantiated.")
            raise e
        except NotFittedError as e:
            print("Scaler not fitted. Fit it beforehand.")
            raise e

    def reset(self):
        """
        Base class method to reset the scaler parameters.

        Returns
        -------
        None.

        """
        for attr in self._scaler_specific_attributes:
            if hasattr(self, attr):
                setattr(self, attr, None)

    @property
    def scaling_family(self):
        return self._scaling_family

class IdentityScaler(BaseResettableScaler):
    """
    Identity scaler.

    Provisional
    -----------
    In ``mlght.core.training.Trainer`` the decision logic for scaler
    instantiation (if None, else ...) might be made simpler
    if the default (the user did not specify any scaler) is
    an identity scaler. That is, it does nothing. When asked
    to fit, transform etc it returns data as they are.

    **Pros**

    * Seamless integration with the ``mlght.core.training.Trainer`` logic
    * No loss of generality

    **Cons**

    * It might confuse the user and/or developer. Implement
      it only if sure that nobody shall put hands on the 
      training orchestration logic!
    """
    def __init__(self):
        super().__init__()
        self._scaling_family = None

    def fit(self, data):
        return data

    def fit_transform(self, data):
        return data

    def transform(self, data):
        return data

    def check_fitted(self):
        return

    def reset(self):
        pass


class ResettableStandardScaler(StandardScaler, BaseResettableScaler):
    """
    Resettable standard scaler.
    """
    def __init__(self):
        """
        Initialize.

        Returns
        -------
        None.

        """
        super().__init__()
        self.deletable = True
        self.is_fit    = False
        self._scaler_specific_attributes = [
            "n_features_in_",
            "n_samples_seen_",
            "mean_",
            "var_",
            "scale_",
            "features_names_in",
        ]
        self._scaling_family = "centering"

    @property
    def data_center(self):
        return self.mean_

    @property
    def data_scale(self):
        return self.scale_

    def check_fitted(self):
        """
        Check if the scaler is fit.

        Returns
        -------
        None.

        """
        return super().check_fitted()

    def reset(self):
        """
        Resets the scaler parameters.

        Returns
        -------
        None.

        """
        super().reset()
        self.is_fit = False


class ResettableMinMaxScaler(MinMaxScaler, BaseResettableScaler):
    """
    Resettable min-max scaler.
    """
    def __init__(self):
        """
        Initialize.

        Returns
        -------
        None.

        """
        super().__init__()
        self.deletable = True
        self.is_fit = False
        self._scaler_specific_attributes = [
            "n_features_in_",
            "n_samples_seen_",
            "min_",
            "data_min_",
            "data_max_",
            "data_range_",
            "features_names_in_"
        ]
        self._scaling_family = "centering"

    @property
    def data_center(self):
        return self.data_min_

    @property
    def data_scale(self):
        return self.data_range_

    def check_fitted(self):
        """
        Check if the scaler is fit.

        Returns
        -------
        None.

        """
        return super().check_fitted()

    def reset(self):
        """
        Resets the scaler parameters.

        Returns
        -------
        None.

        """
        super().reset()
        self.is_fit = False



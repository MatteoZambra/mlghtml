
import numpy as np

__all__ = [
    "ParametersDenormalization"
]

class ParametersDenormalization:
    """
    This class contains utilities to denormalize linear model
    parameters. This serves the purpose of model interpretability.
    """

    @staticmethod
    def parameters_denormalization(
            parameters,
            xscaler,
            yscaler,
            fit_intercept,
            absolute_importance = True
        ):
        r"""
        Gate function. Calls the proper denormalization procedure according to
        the scaling rules of the x- and y-scalers used.

        Parameters
        ----------
        parameters : numpy.ndarray
            Linear model parameters in the scaled space.
        xscaler : might.scalers.ResettableScaler
            The features scaler.
        yscaler : might.scalers.ResettableScaler
            The target scaler.
        fit_intercept : bool
            If the intercept was to fit.
        absolute_importance : bool, optional
            Whether to return unscaled or scaledparameters. The default is True.
            But it is redundant, because of the way in which center and scale
            variables are prepared by the function ``_get_scaler_params``.

        Raises
        ------
        NotImplementedError
            If the scaling rules are not implemented or if the two scalers have
            different scaling rules. For now, let us use the same scaler for 
            both the input and the output variables.

        Returns
        -------
        coeffs : numpy.ndarray
            Coefficients :math:`\left\{ \beta_j \right\}_{j = 1, \dots, p}`.
        intercept : numpy.ndarray
            Intercept :math:`\beta_0`.

        """
        # Preliminary check
        if not xscaler and not yscaler:
            coeffs, intercept = parameters
            return coeffs, intercept
        
        # Get a list with the `scaling_attribute` family for x and y scalers
        scalers_families = [xscaler.scaling_family, yscaler.scaling_family]
        
        # Denormalize according to the scalers families belongingness
        if all([sf == "centering" for sf in scalers_families]):
            coeffs, intercept = ParametersDenormalization.denormalize_center_scaling(
                parameters,
                xscaler,
                yscaler,
                fit_intercept = fit_intercept,
                absolute_importance = absolute_importance
            )
        else:
            raise NotImplementedError("Scaling family not implemented. To come.")
        
        return coeffs, intercept
    
    @staticmethod
    def denormalize_center_scaling(
            parameters,
            xscaler,
            yscaler,
            fit_intercept,
            absolute_importance = True
        ):
        r"""
        Denormalize model parameters.
        
        Assume that data are scaled with the scaling rule
        
        .. math::
            \tilde{\mathbf{x}}_j = \frac{\mathbf{x}_j - C(\mathbf{x}_j)}{S(\mathbf{x}_j)}
        
        The objects :math:`C` and :math:`S` are generic "Center" and "Scale"
        functions.
        
        .. note::
            For the ``sklearn.preprocessing.MinMaxScaler`` object
            
                * :math:`C(\mathbf{x}_j) = \min(\mathbf{x}_j)`
                * :math:`S(\mathbf{x}_j) = \max(\mathbf{x}_j) - \min(\mathbf{x}_j`)
        
            For the ``sklearn.preprocessing.StandardScaler`` object
            
                * :math:`C(\mathbf{x}_j) = \mathrm{mean}(\mathbf{x}_j)`
                * :math:`S(\mathbf{x}_j) = \mathrm{std}(\mathbf{x}_j)`
            
            For the ``sklearn.preprocessing.RobustScaler`` object (not yet
            implemented in the ``might``'s scalers module)
            
                * :math:`C(\mathbf{x}_j) = \mathrm{median}(\mathbf{x}_j)`
                * :math:`S(\mathbf{x}_j) = \mathrm{IQR}(\mathbf{x}_j)`
        
        Then, once the model parameters are fit on scaled data (often recommended),
        the parameters reflect the scaled data space. For **model interpretability**,
        however, one might want the parameters to be expressed in the data 
        natural scale. To undo the parameters scaling, given the scaling rule
        above, one rewrites the expression
        
        .. math::
            \tilde{\mathbf{y}} = \tilde{\beta}_0 + \sum_{j = 1}^{p}
                                            \tilde{\beta}_j \tilde{\mathbf{x}}_j
        
        in terms of the unscaled variables. After rearraging, one obtains
        
        .. math::
            \mathbf{y} = \underbrace{
                    S(\mathbf{y}) \left( \tilde{\beta}_0 - \sum_{j = 1}^{p}
                                         \frac{C(\mathbf{x}_j)}{S(\mathbf{x}_j)}
                                         \mathbf{x}_j \right) + C(\mathbf{y})
                }_{\beta_0} + \sum_{j = 1}^{p} \underbrace{
                    \left( \tilde{\beta}_j \frac{S(\mathbf{y})}{S(\mathbf{x}_j)} \right)
                }_{\beta_i} \mathbf{x}_j
        
        This method performs this unscaling.

        Parameters
        ----------
        As in ``parameters_denormalization``.

        Returns
        -------
        As in ``parameters_denormalization``.

        """
        def _get_scaler_params(scaler, param_shape):
            if scaler:
                scale = scaler.data_scale
                center = scaler.data_center
            else:
                scale = np.ones(param_shape)
                center = np.zeros(param_shape)
            return (
                np.asarray(center),
                np.asarray(scale)
            )

        # TBA: Check that it is a linear model! Either sklearn or custom!
        # For now, we'll simply assume that the user fed the Trainer a linear model
        coefficients, intercepts = parameters

        center_x, scale_x = _get_scaler_params(xscaler, coefficients.shape)
        center_y, scale_y = _get_scaler_params(yscaler, intercepts.shape)

        if absolute_importance:
            # NOTE: in principle, this absolute_importance 
            # statement is not even necessary. If scalers are None,
            # then the _get_scaler_params should return dummy 
            # parameters that have no effect on the coefficients
            
            # Denormalize x
            weights_coeffs = (scale_y / scale_x)
            unscaled_coefficients = coefficients * weights_coeffs

            # Denormalize y
            weights_intercept = center_x / scale_x
            weighted_sum = (coefficients * weights_intercept).sum(axis = 1)
            unscaled_intercepts = scale_y * (intercepts.flatten() - weighted_sum) + center_y
            if not fit_intercept:
                unscaled_intercepts = np.array([[0.]])
        
        return unscaled_coefficients, unscaled_intercepts
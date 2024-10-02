import numpy as np

class Objective:
    pass

class MeanSquaredError(Objective):
    pass

class MeanAbsoluteError(Objective):
    pass

class PinballLoss(Objective):
    def __init__(self, quantiles):
        """
        Initialize with multiple quantiles.
        :param quantiles: array-like, the quantiles for which the loss is calculated. Each must be between 0 and 1.
        """
        self.quantiles = np.array(quantiles)
        if np.any((self.quantiles <= 0) | (self.quantiles >= 1)):
            raise ValueError("Quantile values must be between 0 and 1.")

    def score(self, y_true, y_preds, mean=True):
        """
        Compute the pinball loss between true values and multiple sets of predictions.
        Each set of predictions corresponds to a specific quantile.
        :param y_true: array-like, true values.
        :param y_preds: 2D array-like, predicted values for each quantile. Shape: (n_samples, n_quantiles).
        :return: numpy array, the pinball losses for each quantile.
        """
        y_true = np.array(y_true)
        y_preds = np.array(y_preds)

        assert len(y_true) == y_preds.shape[0], "Number of true values must match the number of predictions."
        assert y_preds.shape[1] == len(self.quantiles), "Number of prediction sets must match the number of quantiles."

        errors = y_true - y_preds
        losses = np.where(errors > 0, self.quantiles * errors, (self.quantiles - 1) * errors)

        if mean:
            return np.nanmean(losses)
        else:
            return losses
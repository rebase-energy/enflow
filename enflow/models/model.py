class Model:
    """
    Base class for all models.

    This class provides common functionality and shared attributes for
    all derived models: Simulator, Predictor, Optimizer, and Agent.

    Attributes
    ----------
    name : str
        Name of the model.
    """
    def __init__(self, name):
        self.name = name
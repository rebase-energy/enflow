from dataclasses import dataclass, fields
import numpy as np

from enflow.spaces import BaseSpace


@dataclass
class OutputSpace(BaseSpace):
    """
    The output space for the energy model.
    """

ActionSpace = OutputSpace
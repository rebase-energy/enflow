from dataclasses import dataclass, fields
import numpy as np

from enerflow.spaces import BaseSpace


@dataclass
class Action(BaseSpace):
    """
    The action vector.

    """

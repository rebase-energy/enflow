from dataclasses import dataclass
import typing as t

import energydatamodel as edm
import emflow as ef

@dataclass
class Problem:
    """
    A problem to be solved by the energy model.
    """
    name: str
    description: str
    energy_system: edm.EnergySystem
    dataset: tuple


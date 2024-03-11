from dataclasses import dataclass
import typing as t

import energydatamodel as edm

@dataclass
class Dataset:
    name: str
    description: str
    energy_system: edm.EnergySystem
    data: t.Optional[edm.TimeSeries] = None
    
    pass
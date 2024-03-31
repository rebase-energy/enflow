from dataclasses import dataclass
import typing as t
import pandas as pd
import xarray as xr

import energydatamodel as edm

@dataclass
class Dataset:
    name: str
    description: t.Optional[str] = None
    energy_system: t.Optional[edm.EnergySystem] = None
    data: t.Optional[t.Dict[str, pd.DataFrame]] = None
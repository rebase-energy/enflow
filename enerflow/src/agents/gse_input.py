import pandas as pd
import numpy as np
from pandas.tseries.frequencies import to_offset


def create_input(load_data,generation_data,grid_data,battery_data):    

    periods = load_data.index

    # Calculate data frequency
    resolution = {None: pd.to_timedelta(to_offset(pd.infer_freq(load_data.index))).total_seconds()/3600}

    # Set grid values
    market_rate  = dict(zip(periods, grid_data['market rate'].values))
    feedin_tariff = dict(zip(periods, grid_data['feedin tariff'].values))
    community_fee =  dict(zip(periods, grid_data['community fee'].values))
    grid_fee = dict(zip(periods, grid_data['grid fee'].values))

    # Set load values
    demand = load_data.stack().to_dict()
    
    # Set generation values
    generation = generation_data.stack().to_dict()
      

    # Set battery values
    battery_min_soc = dict()
    battery_capacity = dict()
    battery_charge_max = dict()
    battery_discharge_max = dict()
    battery_efficiency_charge = dict()
    battery_efficiency_discharge = dict()
    battery_soc_ini = dict()
    battery_soc_fin = dict()
    for c in battery_data.columns:
        
        battery_capacity.update({c: battery_data.loc['capacity',c]}) if 'capacity' in battery_data.index else battery_capacity.update({c: 0.0})
        battery_charge_max.update({c: battery_data.loc['charging power',c]}) if 'charging power' in battery_data.index else battery_charge_max.update({c: 0.0})
        battery_discharge_max.update({c: battery_data.loc['discharging power',c]}) if 'discharging power' in battery_data.index else battery_discharge_max.update({c: 0.0})
        battery_efficiency_charge.update({c: battery_data.loc['charging efficiency',c]}) if 'charging efficiency' in battery_data.index else battery_efficiency_charge.update({c: 1.0})
        battery_efficiency_discharge.update({c: battery_data.loc['discharging efficiency',c]}) if 'discharging efficiency' in battery_data.index else battery_efficiency_discharge.update({c: 1.0})
        battery_min_soc.update({c: battery_data.loc['min soc',c]}) if 'min soc' in battery_data.index else battery_min_soc.update({c: 0.0})
        battery_soc_ini.update({c: battery_data.loc['soc initial',c]}) if 'soc initial' in battery_data.index else battery_soc_ini.update({c: battery_data.loc['min soc',c]})
        battery_soc_fin.update({c: battery_data.loc['soc final',c]}) if 'soc final' in battery_data.index else battery_soc_fin.update({c: battery_data.loc['min soc',c]})
  


    # Create sets
    demand_members = {b for a, b in list(demand.keys())}
    generation_members = {b for a, b in list(generation.keys())}
    battery_members = battery_capacity.keys()
    all_members = demand_members | generation_members | battery_members

    demand_members = list(demand_members)
    generation_members = list(generation_members)
    battery_members = list(battery_members)
    all_members = list(all_members)


    # Create model data input dictionary
    model_data = {None: {
        'T': periods,
        'H': all_members,
        'D': demand_members,
        'G': generation_members,
        'B': battery_members,

        
        'generation': generation,
        'demand': demand,
        
        'battery_min_soc': battery_min_soc,
        'battery_capacity': battery_capacity,
        'battery_charge_max': battery_charge_max,
        'battery_discharge_max': battery_discharge_max,
        'battery_efficiency_charge': battery_efficiency_charge,
        'battery_efficiency_discharge': battery_efficiency_discharge,
        'battery_soc_ini': battery_soc_ini,
        'battery_soc_fin': battery_soc_fin,
        
        'marketmakerrate': market_rate,
        'feedintariff': feedin_tariff,
        'community_fee': community_fee,
        'grid_fee': grid_fee,
        
        'dt': resolution,
    }}  

    return model_data
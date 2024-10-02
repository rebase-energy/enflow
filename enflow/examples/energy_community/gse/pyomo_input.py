import pandas as pd
import numpy as np
from pandas.tseries.frequencies import to_offset


def create_static_input(house):    
    
    # Create model data input dictionary
    model_data = {
        'H': [house.name],
        'D': [house.name] if house.has_demand() else [],
        'G': [house.name] if house.has_pvsystem() else [],
        'B': [house.name] if house.has_battery() else []
        }
    
    
    battery_data = {
        'battery_min_soc': {house.name: house.get_batteries()[0].min_soc_kwh} if house.has_battery() else {},
        'battery_capacity': {house.name: house.get_batteries()[0].capacity_kwh} if house.has_battery() else {},
        'battery_charge_max': {house.name: house.get_batteries()[0].max_charge_kw} if house.has_battery() else {},
        'battery_discharge_max': {house.name: house.get_batteries()[0].max_discharge_kw} if house.has_battery() else {},
        'battery_efficiency_charge': {house.name: house.get_batteries()[0].charge_efficiency} if house.has_battery() else {},
        'battery_efficiency_discharge': {house.name: house.get_batteries()[0].discharge_efficiency} if house.has_battery() else {},                
        }  

    model_data.update(battery_data)    

    return model_data

def create_dynamic_input(state, house):   
    periods = state.time_range_index
    resolution = {None: pd.to_timedelta(to_offset(pd.infer_freq(state.time_range_index))).total_seconds()/3600}

    model_data = {
        'T': periods,

        'demand': state.demand_timeseries.df[[state.demand_timeseries.column_name]].stack().to_dict(),
        'generation': state.pv_timeseries.df[[state.pv_timeseries.column_name]].stack().to_dict(),
 
        'battery_soc_ini': {house.name: state.initial_battery_soc} if house.has_battery() else {},
        'battery_soc_fin': {house.name: state.final_battery_soc} if house.has_battery() else {},

        'marketmakerrate': dict(zip(periods, state.price_timeseries['market rate'].values)),
        'feedintariff': dict(zip(periods, state.price_timeseries['feedin tariff'].values)),
        'community_fee': dict(zip(periods, state.price_timeseries['community fee'].values)),
        'grid_fee': dict(zip(periods, state.price_timeseries['grid fee'].values)),
        
        'dt': resolution,
    }  

    return model_data
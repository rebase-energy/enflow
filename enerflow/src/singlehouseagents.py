import pandas as pd
import numpy as np
from pyomo.environ import AbstractModel,Set,Param,Var,Objective,Constraint,SolverFactory
from pyomo.environ import NonNegativeReals, PositiveReals, inequality
from pyomo.environ import value
from enerflow.src.agents.gse import create_model, instantiate_model, solve_model, get_results
from enerflow.src.agents.pyomo_input import create_static_input, create_dynamic_input
from enerflow.src.singlehouse import Action

solver_path = "C:/Solvers/Cbc-releases.2.10.10-w64-msvc17-md/bin/cbc.exe"

class SingleHouseAgent:
    def __init__(self, house):
        self.house = house

        self.abstract_model = create_model()
        self.static_data = create_static_input(self.house)

    def action(self, state):
        
        # Create model data
        dynamic_data = create_dynamic_input(state, self.house)
        model_data = {None: {**self.static_data, **dynamic_data}}
        # Instantiate model with data
        model_instance = self.abstract_model.create_instance(model_data)
        # Solve model
        optimizer = SolverFactory("cbc", executable=solver_path)
        optimizer.solve(model_instance, tee=True, keepfiles=False)
        # Convert results to action
        power_buy = value(model_instance.PL1_BUY[:,self.house.name])
        power_sell = value(model_instance.PL1_SELL[:,self.house.name])
        battery_charge = value(model_instance.B_IN[:,self.house.name])
        battery_discharge = value(model_instance.B_OUT[:,self.house.name])
        battery_soc = value(model_instance.B_SOC[:,self.house.name])

        action = Action(time_range_index=state.time_range_index,
                        power_buy=power_buy, 
                        power_sell=power_sell,
                        battery_charge=battery_charge, 
                        battery_discharge=battery_discharge, 
                        battery_soc=battery_soc)

        return action    
    
    
class SingleHouseRandomAgent:
    def __init__(self, house):
        self.house = house

    def action(self, state):

        timedelta = 0.25 # To derive it based on data freq
        power_sell = np.zeros(len(state.time_range_index), dtype = float)
        power_buy = np.zeros(len(state.time_range_index), dtype = float)
        battery_charge = np.zeros(len(state.time_range_index), dtype = float)
        battery_discharge = np.zeros(len(state.time_range_index), dtype = float)
        load = np.zeros(len(state.time_range_index), dtype = float)
        pv = np.zeros(len(state.time_range_index), dtype = float)  
        battery_soc = np.zeros(len(state.time_range_index), dtype = float)

        np.random.default_rng().uniform(-1,1,1)
        
        if self.house.has_demand():
            load = self.house.timeseries.get_data().squeeze().to_numpy()
        if self.house.has_pvsystem():
            pv = self.house.get_pvsystems()[0].timeseries.get_data().squeeze().to_numpy() 
        if self.house.has_battery():
        
        
            for i in range(len(state.time_range_index)):
    
                if i == 0:
                    previous_soc = state.initial_battery_soc
                else:
                    previous_soc = battery_soc[i-1]
                    
                batterycharge = np.random.default_rng().uniform(0,1,1)
                batterydischarge = np.random.default_rng().uniform(0,1,1)
                
                battery_charge[i] = max(batterycharge-batterydischarge,0)*self.house.get_batteries()[0].max_charge_kw
                battery_discharge[i] = max(batterydischarge-batterycharge,0)*self.house.get_batteries()[0].max_discharge_kw
                  
                if battery_charge[i] > self.house.get_batteries()[0].max_charge_kw:
                    battery_charge[i] = self.house.get_batteries()[0].max_charge_kw
                    print(f'Battery charge in step {i} is limited to max_charge_kw')
                if battery_discharge[i] > self.house.get_batteries()[0].max_discharge_kw:
                    battery_discharge[i] = self.house.get_batteries()[0].max_discharge_kw 
                    print(f'Battery discharge in step {i} is limited to max_discharge_kw')
                
                battery_soc[i] = previous_soc + self.house.get_batteries()[0].charge_efficiency*battery_charge[i]*timedelta - (1/self.house.get_batteries()[0].discharge_efficiency)*battery_discharge[i]*timedelta
    
                
                # Check battery soc limits
                if battery_soc[i] > self.house.get_batteries()[0].capacity_kwh:
                    battery_soc[i] = self.house.get_batteries()[0].capacity_kwh
                    battery_charge[i] = (battery_soc[i] - previous_soc)/(self.house.get_batteries()[0].charge_efficiency*timedelta)
                    
                if battery_soc[i] < self.house.get_batteries()[0].min_soc_kwh:
                    battery_soc[i] = self.house.get_batteries()[0].min_soc_kwh
                    battery_discharge[i] = (previous_soc - battery_soc[i])*self.house.get_batteries()[0].discharge_efficiency/timedelta   
        
        

        action = Action(time_range_index=state.time_range_index,
                        power_buy=power_buy, 
                        power_sell=power_sell,
                        battery_charge=battery_charge, 
                        battery_discharge=battery_discharge, 
                        battery_soc=battery_soc)

        return action
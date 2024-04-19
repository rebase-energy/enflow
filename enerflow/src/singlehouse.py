import pandas as pd
import numpy as np
from dataclasses import dataclass
import typing as t 
import gymnasium as gym
import energydatamodel as edm
import matplotlib.pyplot as plt


#%%
@dataclass
class SingleHouseState: 
    time_range_index: pd.DatetimeIndex
    demand_timeseries: np.ndarray
    price_timeseries: np.ndarray
    pv_timeseries: t.Optional[np.ndarray] = None
    initial_battery_soc: t.Optional[float] = None
    final_battery_soc: t.Optional[float] = None

#%% 
class SingleHouseEnvironment(gym.Env):

    def __init__(self, house, df_price, initial_battery_soc=None, final_battery_soc=None):
        self.house = house
        self.df_price = df_price
        self.initial_battery_soc = initial_battery_soc
        self.final_battery_soc = final_battery_soc

        self.state = None
        self.action = None
        self.reward = None
        self.done = False
        self.info = {}

    def step(self, action):
        # In this case, we will not calculate a next state.
        # We should check that the solution is feasible. Or should it be an Action method to check for feasibility?
        # Then we should return some sort of flag telling if the solution is feasible or not.
        
        timedelta = 0.25 # To derive it based on data freq
        psell = np.zeros(len(self.state.time_range_index), dtype = float)
        pbuy = np.zeros(len(self.state.time_range_index), dtype = float)
        soc = np.zeros(len(self.state.time_range_index), dtype = float)
        battery_charge = np.zeros(len(self.state.time_range_index), dtype = float)
        battery_discharge = np.zeros(len(self.state.time_range_index), dtype = float)
        load = np.zeros(len(self.state.time_range_index), dtype = float)
        pv = np.zeros(len(self.state.time_range_index), dtype = float)
       
        if self.house.has_demand():
            load = self.house.timeseries.get_data().squeeze().to_numpy()
        if self.house.has_pvsystem():
            pv = self.house.get_pvsystems()[0].timeseries.get_data().squeeze().to_numpy() 
        if self.house.has_battery():
            # dir(self.house)
            
            # Check action length
            # if (len(self.state.time_range_index) != len(action.battery_charge))or(len(self.state.time_range_index) != len(action.battery_discharge)):
            #     print('Not Feasible - Different Length')

            # # Check battery charging power feasibility
            # if any(action.battery_charge > self.house.get_batteries()[0].max_charge_kw):
            #     print('Not Feasible - battery_charge')
 
            # # Check battery discharging power feasibility
            # if any(action.battery_discharge > self.house.get_batteries()[0].max_discharge_kw):
            #     print('Not Feasible - battery_discharge')    
            
            for i in range(len(self.state.time_range_index)):

                if i == 0:
                    previous_soc = self.state.initial_battery_soc
                else:
                    previous_soc = soc[i-1]
                    
                battery_charge[i] = action.battery_charge[i]
                battery_discharge[i] = action.battery_discharge[i]
                  
                if battery_charge[i] > self.house.get_batteries()[0].max_charge_kw:
                    battery_charge[i] = self.house.get_batteries()[0].max_charge_kw
                    print(f'Battery charge in step {i} is limited to max_charge_kw')
                if battery_discharge[i] > self.house.get_batteries()[0].max_discharge_kw:
                    battery_discharge[i] = self.house.get_batteries()[0].max_discharge_kw 
                    print(f'Battery discharge in step {i} is limited to max_discharge_kw')
                
                soc[i] = previous_soc + self.house.get_batteries()[0].charge_efficiency*battery_charge[i]*timedelta - (1/self.house.get_batteries()[0].discharge_efficiency)*battery_discharge[i]*timedelta

                
                # Check battery soc limits
                if soc[i] > self.house.get_batteries()[0].capacity_kwh:
                    soc[i] = self.house.get_batteries()[0].capacity_kwh
                    battery_charge[i] = (soc[i] - previous_soc)/(self.house.get_batteries()[0].charge_efficiency*timedelta)
                    
                if soc[i] < self.house.get_batteries()[0].min_soc_kwh:
                    soc[i] = self.house.get_batteries()[0].min_soc_kwh
                    battery_discharge[i] = (previous_soc - soc[i])*self.house.get_batteries()[0].discharge_efficiency/timedelta   
        
        # Calculate power trading quantities
        trade = pv-load+battery_discharge-battery_charge
        psell[trade>0] = trade[trade>0]
        pbuy[trade<0] = -trade[trade<0]
        
        # Calculate cost
        cost_grid = (np.array(self.df_price['community fee'])+np.array(self.df_price['grid fee']))*(pbuy + psell)*timedelta
        cost_energy = np.array(self.df_price['market rate'])*pbuy*timedelta - np.array(self.df_price['feedin tariff'])*psell*timedelta
    
        return psell, pbuy, soc, battery_charge, battery_discharge, cost_grid, cost_energy

    def reset(self):
        self.state = SingleHouseState(time_range_index=self.house.timeseries.df.index,
                                      demand_timeseries=self.house.timeseries,
                                      price_timeseries=self.df_price)
        
        if self.house.has_pvsystem():
            pvsystem = self.house.get_pvsystems()[0]
            self.state.pv_timeseries = pvsystem.timeseries
        
        if self.house.has_battery():
            self.state.initial_battery_soc = self.initial_battery_soc
            self.state.final_battery_soc = self.final_battery_soc

        return self.state

    def render(self, psell, pbuy, soc, battery_charge, battery_discharge, cost_grid, cost_energy): # Include quantities into state

        load = np.zeros(len(self.state.time_range_index), dtype = float)
        pv = np.zeros(len(self.state.time_range_index), dtype = float)

        if self.house.has_demand():
            load = self.house.timeseries.get_data().squeeze().to_numpy()
        if self.house.has_pvsystem():
            pv = self.house.get_pvsystems()[0].timeseries.get_data().squeeze().to_numpy()

        
        linewidth = 1.5
        fontsize = 12

        # Share both X and Y axes with all subplots
        fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, sharex='all', sharey='none', figsize=(15, 12))
        ax1.plot(self.state.time_range_index, load, linestyle='solid', color='blue', linewidth=linewidth)
        ax1.set_ylabel('Load [kW]', fontsize=fontsize)
        ax1.set_title(f'House {self.house.assets[0].timeseries.column_name}')
        # ax1.set_yticks(list(range(0,10,1)))
        ax1.grid(color='gray', axis = 'both', visible = True)
 
        ax2.plot(self.state.time_range_index, pv, linestyle='solid', color='orange', linewidth=linewidth)
        ax2.set_ylabel('PV [kW]', fontsize=fontsize)
        ax2.grid(color='gray', axis = 'both', visible = True)
        
        ax3.plot(self.state.time_range_index, soc, linestyle='solid', color='red', linewidth=linewidth)
        ax3.set_ylabel('Battery SOC [kWk]', fontsize=fontsize)
        ax3.grid(color='gray', axis = 'both', visible = True)
        
        ax4.plot(self.state.time_range_index, pbuy, linestyle='solid', color='brown', linewidth=linewidth)
        ax4.set_ylabel('Grid import [kW]', fontsize=fontsize)
        ax4.grid(color='gray', axis = 'both', visible = True)
        
        ax5.plot(self.state.time_range_index, psell, linestyle='solid', color='green', linewidth=linewidth)
        ax5.set_ylabel('Grid export [kW]', fontsize=fontsize)
        ax5.grid(color='gray', axis = 'both', visible = True)
        
        ax6.plot(self.state.time_range_index, cost_grid+cost_energy, linestyle='solid', color='black', linewidth=linewidth)
        ax6.set_ylabel('Cost [EUR]', fontsize=fontsize)
        ax6.grid(color='gray', axis = 'both', visible = True)
        
        plt.tick_params(axis='x', rotation=0)
        fig.align_labels()
        
        return fig

    def close(self):
        # Close the environment
        pass

#%%
@dataclass
class Action:
    # Somehow we want to constriain the action space using bounds. 
    # Maybe use the gym spaces class? Box then add attribute for index datatime
    # Should action include a method to check for feasibility?
    time_range_index: pd.DatetimeIndex
    power_buy: np.ndarray
    power_sell: np.ndarray
    battery_soc: t.Optional[np.ndarray] = None
    battery_charge: t.Optional[np.ndarray] = None
    battery_discharge: t.Optional[np.ndarray] = None  



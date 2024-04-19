import pandas as pd
import energydatamodel as edm
from enerflow.src.singlehouse import SingleHouseEnvironment
from enerflow.src.singlehouseagents import SingleHouseAgent, SingleHouseRandomAgent


#%% Import data
df_load = pd.read_csv("./enerflow/examples/energy_community/data/load.csv", index_col=0, parse_dates=True)
df_pv = pd.read_csv("./enerflow/examples/energy_community/data/pv.csv", index_col=0, parse_dates=True)
df_price = pd.read_csv("./enerflow/examples/energy_community/data/grid.csv", index_col=0, parse_dates=True)
df_battery = pd.read_csv("./enerflow/examples/energy_community/data/battery.csv", index_col=0)


#%% Create the energy community assets
all_houses = df_load.columns
houses_with_pv = df_pv.columns
houses_with_battery = df_battery.columns

houses = []
for house_name in all_houses:
    house = edm.House(name=house_name, timeseries=edm.TimeSeries(df=df_load, column_name=house_name, filename="load.csv"))

    if house_name in houses_with_pv:
        pvsystem = edm.PVSystem(timeseries=edm.TimeSeries(df=df_pv, column_name=house_name, filename="pv.csv"))
        house.add_assets(pvsystem)

    if house_name in houses_with_battery:
        battery = edm.Battery(capacity_kwh=df_battery.loc["capacity", house_name],
                              min_soc_kwh=df_battery.loc["min soc", house_name],
                              max_charge_kw=df_battery.loc["charging power", house_name],
                              max_discharge_kw=df_battery.loc["discharging power", house_name],
                              charge_efficiency=df_battery.loc["charging efficiency", house_name],
                              discharge_efficiency=df_battery.loc["discharging efficiency", house_name])
        house.add_assets(battery)

    houses.append(house)

energycommunity = edm.EnergyCommunity(assets=houses)


#%% Create the single house environment
house = energycommunity.assets[0]
env = SingleHouseEnvironment(house=house, df_price=df_price, initial_battery_soc=0, final_battery_soc=0)


#%% Random action agent

# Initialize the environment
initial_state = env.reset()

# Initialize the agent
agent = SingleHouseRandomAgent(house=house)

# Run the agent
action = agent.action(initial_state)

# Get the next state
psell, pbuy, soc, battery_charge, battery_discharge, cost_grid, cost_energy = env.step(action)

# Visualize
fig = env.render(psell, pbuy, soc, battery_charge, battery_discharge, cost_grid, cost_energy)

total_cost = sum(cost_grid) + sum(cost_energy)



#%% Optimized action agent

# Initialize the environment
initial_state = env.reset()

# Initialize the agent
agent = SingleHouseAgent(house=house)

# Run the agent
action = agent.action(initial_state)

# Get the next state
psell, pbuy, soc, battery_charge, battery_discharge, cost_grid, cost_energy = env.step(action)

# Visualize
fig = env.render(psell, pbuy, soc, battery_charge, battery_discharge, cost_grid, cost_energy)

total_cost = sum(cost_grid) + sum(cost_energy)
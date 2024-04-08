<div align="center">
	<img width="300" src="https://github.com/rebase-energy/enerflow/blob/main/assets/enerflow-logo.png?raw=true" alt="enerflow">
<h2 style="margin-top: 0px;">
    ⚡ Open-source Python framework for modelling sequential decision problems in the energy sector
</h2>
</div>

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/enerflow.svg)](https://badge.fury.io/py/enerflow) 
[![All Contributors](https://img.shields.io/badge/all_contributors-2-orange.svg?style=flat-square)](#contributors)
[![Join us on Slack](https://img.shields.io/badge/Join%20us%20on%20Slack-%2362BEAF?style=flat&logo=slack&logoColor=white)](https://join.slack.com/t/rebase-community/shared_invite/zt-1dtd0tdo6-sXuCEy~zPnvJw4uUe~tKeA) 
[![GitHub Repo stars](https://img.shields.io/github/stars/rebase-energy/enerflow?style=social)](https://github.com/rebase-energy/enerflow)

**enerflow** is a Python framework for modeling sequential decision problems in the energy sector. It is based on both OpenAI Gym (now [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)) and [Warran Powell's decision framework](https://castle.princeton.edu/rlso/). enerflow enables you to: 

* 🛤️ Structure your code as modular and reusable components and adopt the "model first, then solve"-mantra. 
* 🏗️ Build agents, predictors, optimizers and simulators to solve sequential decision problems
* 🌱 Forumate your problems as environments and evalute pre-existing solutions
* 🧪 Run parametrized experiments that generate reproducible results (code, data and parameters)
* ➿ Run sweeps for benchmarking, scenario analysis and parameter tuning

**📖 [Documentation](https://docs.energydatamodel.org/en/latest/)**
&ensp;|&ensp;
**🚀 [Quickstart - Try out now in Colab](-)**

As energy systems world-wide continue to decarbonise through deployment of renewable and fossil-free energy, they are also becoming more intermittent and complex. We need better data-driven methods to handle this complexity, with `python` rapidly becoming the tool-of-choice for energy modellers. 

## Framework approach and example
**enerflow** is about adopting a problem-centric approach that follows the "model first, then solve"-mantra. Concretely, this means that problems are solved through the following steps: 

1. Define the considered energy system
2. Define state, action and exogenous spaces
3. Create the environment and the transition function
4. Define the cost (or contribution)
5. Create the agent to operate in environment
6. Run the agent and evaluate performance

Following is an example pseudo-code outlining the steps of the framework: 

```bash
import pandas as pd
import gymnasium as gym
import enerflow as ef

# Load data
df = pd.read_csv("data.csv")

# 1) Define the energy system
pvsystem = ef.PVSystem(capacity=2400, surface_azimuth=180, surface_tilt=25, timeseries=ef.TimeSeries(df=df, column="pv_power"))
windturbine = ef.WindTurbine(capacity=3200, hub_height=120, rotor_diameter=100, timeseries=ef.TimeSeries(df=df, column="wind_power"))
demand = ef.ElectricityDemand(df=df, column="demand")
battery = ef.Battery(storage_capacity=1000, min_soc=150, max_charge=500, max_discharge=500)

microgrid = ef.MicroGrid(assets=[pvsystem, windturbine, demand, battery], latitude=46, longitude=64)

dataset = ef.Dataset(data={"data": df}, energysystem=microgrid)

# 2) Define state, action and exogenous spaces
state_space = gym.spaces.Dict(...)
action_space = gym.spaces.Dict(...)
exogenous_space = gym.spaces.Dict(...)

# 3) Define the environment
class MicroGridEnv(gym.Env):
    def __init__():
        self.state_space = state_space
        self.action = action
        self.exogenous = exogenous
    def reset():  
        ...
    def step():  
        ...
env = MicroGridEnv(dataset=dataset)

# 4) Define the cost (or contribution)
class Cost(ef.Objective):
    ...
cost = Cost()

# 5) Define the agent to operate in environment
class Agent(ef.Agent):
    ... 
agent = Agent()

state = env.reset()
while done is not True:
    action = agent.act(state)
    state, exogeneous, done, info = env.step(action)
    revenue = scorer.calculate(state, action, exogeneous)

env.close()
```

## Installation

Install the **stable** release: 
```bash
pip install enerflow
```

Install the **latest** release: 
```bash
pip install git+https://github.com/rebase-energy/enerflow
```

Install in editable mode for **development**: 
```bash
git clone https://github.com/rebase-energy/enerflow.git
cd enerflow
pip install -e . 
```

## Contributors

* test
* test 2



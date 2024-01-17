"""This example showcases how to use emflow for the solar forecasting task of GEFCom2014."""

import gym
from gym import spaces
from dataclasses import dataclass
import numpy as np

@dataclass
class State:
    """The state vector.

    Args:
        pv_production [kWh/h]: Power production from solar wind used in the microgrid.
    """

    pv_production: float #: testing to write something here
    grid_import_peak: float
    spot_market_price: float

    @property
    def vector(self) -> np.ndarray:
        return np.array(
            [
                self.consumption,
                self.pv_production,
                self.wind_production,
                self.battery_storage,
                self.hydrogen_storage,
                self.grid_import,
                self.grid_import_peak,
                self.spot_market_price,
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_vector(cls, state: np.ndarray) -> "State":
        return cls(
            consumption=state.item(0),
            pv_production=state.item(1),
            wind_production=state.item(2),
            battery_storage=state.item(3),
            hydrogen_storage=state.item(4),
            grid_import=state.item(5),
            grid_import_peak=state.item(6),
            spot_market_price=state.item(7),
        )


class GEFCom2014SolarEnv(gym.Env):
    """An Environment for GEFCom2014 Solar Competition."""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(GEFCom2014SolarEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example: self.action_space = spaces.Discrete(2)
        # Example: self.observation_space = spaces.Box(low=0, high=255, shape=(...), dtype=np.float32)
        pass

    def step(self, action):
        # Execute one time step within the environment
        # return observation, reward, done, info
        return 0

    def reset(self):
        # Reset the state of the environment to an initial state
        pass

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass

    def close(self):
        # Clean up resources
        pass

# Test your environment
env = GEFCom2014SolarEnv()

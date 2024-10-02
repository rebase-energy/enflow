import gymnasium as gym
import pandas as pd
import numpy as np
from gymnasium.spaces import Box, Dict

class DataFrameSpace(gym.Space):
    def __init__(self, column_spaces, n_rows=None):
        """
        Custom space for a pandas DataFrame where each column has its own bounds defined by Box spaces.
        
        The column_spaces can be either:
        - A dictionary where the keys are column names and values are gym.spaces.Box,
        - A gym.spaces.Dict, where the keys are column names and values are gym.spaces.Box.

        :param column_spaces: Either a dictionary or gym.spaces.Dict with column names and their corresponding Box spaces.
        :param n_rows: Number of rows in the DataFrame. If None, the number of rows is flexible.
        """
        # Check if column_spaces is a Dict space from gym or a normal dictionary
        if isinstance(column_spaces, Dict):
            column_spaces = column_spaces.spaces  # Extract the internal dictionary
        
        assert isinstance(column_spaces, dict), "column_spaces must be a dictionary or gym.spaces.Dict with column names as keys and Box spaces as values"
        
        # Validate that each value is a gym.spaces.Box
        for col, box_space in column_spaces.items():
            assert isinstance(box_space, Box), f"Box space for column '{col}' must be a gym.spaces.Box instance"
        
        self.column_spaces = column_spaces
        self.columns = list(column_spaces.keys())
        self.n_rows = n_rows
        
        # Define the shape of the dataframe
        if n_rows is not None:
            shape = (n_rows, len(self.columns))
        else:
            shape = (None, len(self.columns))  # Flexible row dimension

        super().__init__(shape=shape, dtype=None)

    def sample(self, n_rows=None):
        """
        Generate a random sample DataFrame where each column adheres to its respective Box space constraints.
        """

        if n_rows is None and self.n_rows is None:
            raise ValueError("Parameter 'n_rows' is required because 'self.n_rows' is None.")
        if n_rows is None:
            n_rows = self.n_rows       

        # Create a DataFrame by sampling from each column's Box space
        data = {col: np.random.uniform(
                    low=self.column_spaces[col].low,
                    high=self.column_spaces[col].high,
                    size=(n_rows,)
                ).astype(self.column_spaces[col].dtype)
                for col in self.columns}

        return pd.DataFrame(data)

    def contains(self, x):
        """
        Check if a given DataFrame is contained in the space, i.e., if each column's values lie within its respective Box space.
        """
        if not isinstance(x, pd.DataFrame):
            return False
        if len(x.columns) != len(self.columns):
            return False
        if set(x.columns) != set(self.columns):
            return False
        if self.n_rows is not None and len(x) != self.n_rows:
            return False
        
        # Check each column individually
        for col in self.columns:
            values = x[col].values
            if not np.all((values >= self.column_spaces[col].low) & (values <= self.column_spaces[col].high)):
                return False
        return True

    def __repr__(self):
        return f"DataFrameSpace(columns={self.columns}, n_rows={self.n_rows})"

    def __eq__(self, other):
        return (
            isinstance(other, DataFrameSpace)
            and self.column_spaces == other.column_spaces
            and self.n_rows == other.n_rows
        )

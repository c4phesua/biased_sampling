"""Base class for samplers"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.core.rng import RandomNumberGenerator
from src.core.state import State
from src.core.state_space import RealVectorStateSpace
from .space_info import SpaceInformation


class BaseSampler(ABC):
    """
    Abstract base class for all state samplers.
    """

    def __init__(self, space: RealVectorStateSpace, seed=None):
        if not space:
            raise ValueError("State space cannot be null")

        self.space = space
        self.rng = RandomNumberGenerator(seed)

    def _validate_state(self, state):
        if not isinstance(state, State):
            raise TypeError("Expected a State instance.")

        if state.dimension != self.space.dimension:
            raise ValueError("State dimension does not match.")

    @abstractmethod
    def sample_uniform(self, state):
        """
        Sample a state in the configuration space.
        :param state: The state to be populated.
        """
        pass

    @abstractmethod
    def sample_uniform_near(self, state, near_state, distance: float):
        """
        Sample a state near a given state within a specified distance.
        :param state: The target state to populate.
        :param near_state: The reference state.
        :param distance: The maximum distance radius from near_state.
        :return: True if successful, False otherwise.
        """
        pass

    @abstractmethod
    def sample_gaussian(self, state, mean, std):
        """
        Sample a state from a Gaussian distribution
        :param state: The target state to populate.
        :param mean: Mean of the Gaussian distribution
        :param std: Standard deviation of the Gaussian distribution
        """
        pass


class BaseValidStateSampler(ABC):
    """
    Abstract base class for valid state samplers
    """

    def __init__(self, space_info: SpaceInformation, name="BaseValidStateSampler"):
        if space_info is None:
            raise ValueError("Space information cannot be null.")

        self.si = space_info
        self.name = name

    @abstractmethod
    def sample(self, state: State) -> bool:
        """
        Attempt to populate state with a valid configuration.

        Returns:
            True if a valid state was found.
            False otherwise.
        """
        pass

    @abstractmethod
    def sample_near(self, state: State, near_state: State, distance: float = 0.0) -> bool:
        """
        Attempt to find a valid state near near_state.

        Returns:
            True if a valid state was found.
            False otherwise.
        """
        pass

    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name
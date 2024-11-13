from dataclasses import dataclass, field
import itertools
import string

import numpy as np


@dataclass
class BoundVector:
    """
    A vector in 3D space with a magnitude, location, and an optional name.

    Attributes:
        magnitude (np.ndarray): The vector's magnitude, represented as a 3D numpy array.
        location (np.ndarray): The vector's location in 3D space, represented as a 3D numpy array.
        name (str): An optional name for the vector. If not provided, a unique alphabetical name
                    will be generated automatically.

    Class Attributes:
        _name_generator (itertools.cycle): A generator for unique alphabetical names in sequence
                                           (e.g., 'A', 'B', ..., 'Z', 'AA', 'AB', etc.).
    """

    magnitude: np.ndarray = field(default_factory=lambda: np.zeros(3))
    location: np.ndarray = field(default_factory=lambda: np.zeros(3))
    name: str = None  # Name will be auto-generated if not provided

    # Class-level iterator for alphabetical names
    _name_generator = itertools.cycle(
        "".join(chars)
        for chars in itertools.chain.from_iterable(
            itertools.product(string.ascii_uppercase, repeat=i) for i in range(1, 3)
        )
    )

    def __post_init__(self):
        """
        Post-initialization to assign an alphabetical name if none is provided.
        """
        if self.name is None:
            self.name = next(self._name_generator)

    def unit_vector(self):
        """
        Calculate the unit vector of the magnitude vector.

        Returns:
            np.ndarray: The unit vector of the magnitude.
        """
        return self.magnitude / np.linalg.norm(self.magnitude)

    def __add__(self, other):
        """
        Add two BoundVectors, if they share the same location.

        Args:
            other (BoundVector, list, or tuple): Another vector to add.

        Returns:
            BoundVector: A new BoundVector with the combined magnitude.

        Raises:
            AssertionError: If the locations of the vectors do not match.
        """
        if isinstance(other, (list, tuple)):
            other = np.asarray(other)
        if isinstance(other, BoundVector):
            assert np.array_equal(self.location, other.location), "Locations must match"
        return self.__class__(
            magnitude=self.magnitude + other.magnitude, location=self.location
        )

    def __sub__(self, other):
        """
        Subtract another BoundVector, if they share the same location.

        Args:
            other (BoundVector, list, or tuple): Another vector to subtract.

        Returns:
            BoundVector: A new BoundVector with the resultant magnitude.

        Raises:
            AssertionError: If the locations of the vectors do not match.
        """
        if isinstance(other, (list, tuple)):
            other = np.asarray(other)
        if isinstance(other, BoundVector):
            assert np.array_equal(self.location, other.location), "Locations must match"
        return self.__class__(
            magnitude=self.magnitude - other.magnitude, location=self.location
        )

    def __neg__(self):
        """
        Negate the BoundVector's magnitude.

        Returns:
            BoundVector: A new BoundVector with negated magnitude.
        """
        return self.__class__(magnitude=-self.magnitude, location=self.location)


@dataclass
class Load(BoundVector):
    magnitude: np.ndarray = field(default_factory=lambda: np.zeros(6))

    def __post_init__(self):
        super().__post_init__()
        if self.magnitude.size < 6:
            self.magnitude = np.pad(self.magnitude, (0, 6 - self.magnitude.size))
        if self.magnitude.size > 6:
            raise ValueError("Magnitude must be a 1x6 vector.")

    def normal_force(self):
        return np.linalg.norm(self.magnitude[:3])

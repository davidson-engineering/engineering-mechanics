# Engineering Mechanics

The **Engineering Mechanics** repository provides a set of Python classes and tools to model and solve engineering mechanics problems, including statics, and dyanmics problems. This repository is aimed at supporting mechanical engineering calculations and simulations.

## Overview

This repository includes:
- `BoundVector`: A class to represent vectors in 3D space with automatic alphabetical naming for ease in handling multiple vectors in calculations.
- **Statics Calculations**: Utilities for constructing equilibrium matrices, solving for reactions, and ensuring stability within mechanical systems.
- **Constraint Modeling**: Flexible constraint matrices that allow for complex and realistic modeling of supports, including coupled constraints.

### Statics and Constraints
The statics tools provide utilities to:
- Set up equilibrium matrices for multi-reaction systems.
- Analyze whether a system is overconstrained or underconstrained.
- Handle advanced constraint matrices with non-diagonal elements, suitable for inclined or coupled constraints.

## Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/davidson-engineering/engineering-mechanics.git
cd engineering-mechanics
pip install .
```

## Examples

### Static Equilibrium Check

For a fully constrained system with multiple reactions, you can construct an equilibrium matrix and check for overconstraints:

```python
from statics import StaticsCalculator, BoundVector, Reaction

# Define forces and reactions here, then use StaticsCalculator to analyze
forces = [BoundVector(magnitude=np.array([0, 0, -100]), location=np.array([1, 0, 0]))]
moments = [BoundVector(magnitude=np.array([0, 10, -5]), location=np.array([0, 1, 0]))]
reactions = [Reaction(location=np.array([0, 0, 0]), constraint=np.eye(6))]

calculator = StaticsCalculator(forces, moments, reactions)
reactions_result = calculator.solve_reactions()
print(reactions_result)
>>> [[   0.    0.  100.    5. -110.    5.]]
```

## License

This project is licensed under the MIT License.
# Engineering Mechanics

This repository provides a set of Python classes and tools to model and solve engineering mechanics problems, including statics, and dyanmics problems. This repository is aimed at supporting mechanical engineering calculations and simulations.

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
import numpy as np
from statics import StaticsCalculator, Load, Reaction

# Define forces and reactions here, then use StaticsCalculator to analyze
loads = [
    Load(
        magnitude=np.array([0, 0, -100, 0, -10, 0]),
        location=np.array([1, 0, 0]),
        name="Load_A",
    )
]
reactions = [
    Reaction(
        location=np.array([0, 0, 0]), constraint=np.eye(6), name="Fixed support"
    ),
]

calculator = ReactionSolver(loads, reactions)
result = calculator.run()
result.print_summary(html_report_path="report.html")
```

One can also print a summary table using prettytable:

```python
reactions_result = calculator.run()
```

```console
Input loads Summary:
+------+-------+-------+-------+------+------+---------+------+--------+------+
| Load | Loc X | Loc Y | Loc Z |  Fx  |  Fy  |    Fz   |  Mx  |   My   |  Mz  |
+------+-------+-------+-------+------+------+---------+------+--------+------+
| F_a  |  1.00 |  0.00 |  0.00 | 0.00 | 0.00 | -100.00 | 0.00 | -10.00 | 0.00 |
+------+-------+-------+-------+------+------+---------+------+--------+------+

Constraints Summary:
+---------------+-------------------+
| Reaction      | Constraint Matrix |
+---------------+-------------------+
| Fixed support |   [1 0 0 0 0 0]   |
|               |   [0 1 0 0 0 0]   |
|               |   [0 0 1 0 0 0]   |
|               |   [0 0 0 1 0 0]   |
|               |   [0 0 0 0 1 0]   |
|               |   [0 0 0 0 0 1]   |
+---------------+-------------------+

Reactions Summary:
+---------------+-------+-------+-------+------+------+--------+------+--------+-------+
| Reaction      | Loc X | Loc Y | Loc Z |  Fx  |  Fy  |   Fz   |  Mx  |   My   |   Mz  |
+---------------+-------+-------+-------+------+------+--------+------+--------+-------+
| Fixed support |  0.00 |  0.00 |  0.00 | 0.00 | 0.00 | 100.00 | 0.00 | -90.00 | 10.00 |
+---------------+-------+-------+-------+------+------+--------+------+--------+-------+
```
## License

This project is licensed under the MIT License.

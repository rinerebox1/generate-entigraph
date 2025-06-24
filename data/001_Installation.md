Title: Installation — codableopt v0.1 ドキュメント

URL Source: https://codable-model-optimizer.readthedocs.io/ja/latest/

Markdown Content:
[codableopt](https://codable-model-optimizer.readthedocs.io/ja/latest/#)

Use pip[](https://codable-model-optimizer.readthedocs.io/ja/latest/#use-pip "この見出しへのパーマリンク")
---------------------------------------------------------------------------------------------

$ pip install codableopt

Use setup.py[](https://codable-model-optimizer.readthedocs.io/ja/latest/#use-setup-py "この見出しへのパーマリンク")
-------------------------------------------------------------------------------------------------------

# Master branch
$ git clone https://github.com/recruit-tech/codable-model-optimizer
$ python3 setup.py install

Sample:[](https://codable-model-optimizer.readthedocs.io/ja/latest/#sample "この見出しへのパーマリンク")
--------------------------------------------------------------------------------------------

import numpy as np
from codableopt import *

# set problem
problem = Problem(is_max_problem=True)

# define variables
x = IntVariable(name='x', lower=np.double(0), upper=np.double(5))
y = DoubleVariable(name='y', lower=np.double(0.0), upper=None)
z = CategoryVariable(name='z', categories=['a', 'b', 'c'])

# define objective function
def objective_function(var_x, var_y, var_z, parameters):
    obj_value = parameters['coef_x'] * var_x + parameters['coef_y'] * var_y

    if var_z == 'a':
        obj_value += 10.0
    elif var_z == 'b':
        obj_value += 8.0
    else:
        # var_z == 'c'
        obj_value -= 3.0

    return obj_value

# set objective function and its arguments
problem += Objective(objective=objective_function,
                     args_map={'var_x': x,
                               'var_y': y,
                               'var_z': z,
                               'parameters': {'coef_x': -3.0, 'coef_y': 4.0}})

# define constraint
problem += 2 * x + 4 * y + 2 * (z == 'a') + 3 * (z == ('b', 'c')) <= 8
problem += 2 * x - y + 2 * (z == 'b') > 3

print(problem)

solver = OptSolver()

# generate optimization methods to be used within the solver
method = PenaltyAdjustmentMethod(steps=40000)

answer, is_feasible = solver.solve(problem, method)
print(f'answer:{answer}, answer_is_feasible:{is_feasible}')


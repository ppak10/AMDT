import numpy as np

from amdt import EagarTsai

mesh = {
    "b_c": "temp",
    "x_min": -1000e-5,
    "x_max": 1000e-5,
    "y_min": -1000e-5,
    "y_max": 1000e-5,
    "z_min": -200e-6,
    "z_max": 0,
}


et = EagarTsai(mesh=mesh)

timestep = 1000e-6
et.forward(timestep, 0)
et.forward(timestep, np.pi / 2)
results = et.meltpool(True, True)
print(results)

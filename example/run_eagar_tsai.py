import numpy as np

from amdt import EagarTsai

bounds = {
    "x": [-1000e-5, 1000e-5],
    "y": [-1000e-5, 1000e-5],
    "z": [-200e-6, 0],
}

et = EagarTsai(bc="temp", bounds=bounds, b=200e-6)

timestep = 1000e-6
et.forward(timestep, 0)
et.forward(timestep, np.pi / 2)
results = et.meltpool(True, True)
print(results)

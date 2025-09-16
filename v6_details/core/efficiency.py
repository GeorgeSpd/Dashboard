# pylint: disable=C0114
# pylint: disable=C0103
# pylint: disable=C0116
# pylint: disable=C0301
# pylint: disable=I1101
# pylint: disable=W0401
# pylint: disable=W0614
# pylint: disable=C0303
# pylint: disable=W0105
# pylint: disable=W0621
# pylint: disable=W0718
# pylint: disable=C0302

"""
core/efficiency.py

Temperature-dependent efficiency profiles (COP/EER) for heating/cooling.
Edit the points below to match your equipment datasheets.
"""

import numpy as np

def make_curve(x_pts, y_pts, y_min=0.5, y_max=10.0):
    """
    Build a callable f(x) that linearly interpolates (x_pts -> y_pts)
    and clips results to [y_min, y_max]. x is outdoor temperature [°C].
    """
    x_pts = np.asarray(x_pts, dtype=float)
    y_pts = np.asarray(y_pts, dtype=float)

    def f(x):
        x_arr = np.asarray(x, dtype=float)
        y = np.interp(x_arr, x_pts, y_pts)
        return np.clip(y, y_min, y_max)

    return f

# HEATING (COP) profiles
HEATING_COP = {
    # Air-to-water heat pump — Standard family
    "ASHP Std (35°C)": make_curve([-15,-10,-5, 0, 5,10,15], [1.9,2.2,2.6,3.1,3.6,4.1,4.5]),
    "ASHP Std (45°C)": make_curve([-15,-10,-5, 0, 5,10,15], [1.6,1.9,2.2,2.6,3.1,3.5,3.9]),
    "ASHP Std (60°C)": make_curve([-15,-10,-5, 0, 5,10,15], [1.3,1.6,1.9,2.3,2.7,3.0,3.3]),

    # Air-to-water heat pump — High-efficiency family
    "ASHP Hi (35°C)":  make_curve([-15,-10,-5, 0, 5,10,15], [2.2,2.5,2.9,3.4,4.0,4.6,5.0]),
    "ASHP Hi (45°C)":  make_curve([-15,-10,-5, 0, 5,10,15], [1.9,2.2,2.5,3.0,3.5,4.0,4.4]),
    "ASHP Hi (60°C)":  make_curve([-15,-10,-5, 0, 5,10,15], [1.6,1.9,2.2,2.6,3.1,3.5,3.8]),

    # Ground-source (water-to-water). Flatter dependence.
    "GSHP Std (35°C)": make_curve([-15,-10,-5, 0, 5,10,15], [3.6,3.8,4.0,4.2,4.4,4.6,4.8]),
    "GSHP Std (45°C)": make_curve([-15,-10,-5, 0, 5,10,15], [3.4,3.6,3.8,4.0,4.2,4.4,4.6]),
    "GSHP Std (60°C": make_curve([-15,-10,-5, 0, 5,10,15], [3.2,3.4,3.6,3.8,4.0,4.2,4.4]),

    # Electric resistance backup
    "Electric heater": lambda x: np.ones_like(np.asarray(x, float)) * 1.0,
}

# Gas boiler efficiencies
BOILER_EFF = {
    "Condensing boiler (η=0.95)": lambda x: np.ones_like(np.asarray(x, float)) * 0.95,
    "Standard boiler (η=0.85)":   lambda x: np.ones_like(np.asarray(x, float)) * 0.85,
}

# COOLING (EER) profiles
COOLING_EER = {
    # Air-to-water reversible HP, standard efficiency
    "ASHP Std (7°C)":  make_curve([20,25,30,35,40], [5.2,4.6,4.0,3.4,3.0]),
    "ASHP Std (12°C)": make_curve([20,25,30,35,40], [6.0,5.4,4.8,4.2,3.7]),
    "ASHP Std (18°C)": make_curve([20,25,30,35,40], [6.8,6.2,5.6,5.0,4.5]),

    # High-efficiency family
    "ASHP Hi (7°C)":   make_curve([20,25,30,35,40], [5.8,5.2,4.6,4.0,3.5]),
    "ASHP Hi (12°C)":  make_curve([20,25,30,35,40], [6.6,6.0,5.4,4.8,4.3]),
    "ASHP Hi (18°C)":  make_curve([20,25,30,35,40], [7.4,6.8,6.2,5.6,5.1]),

    # Ground-source (weak ambient dependence, flatter curves)
    "GSHP Std (7°C)":  make_curve([20,25,30,35,40], [6.2,6.0,5.8,5.6,5.4]),
    "GSHP Std (12°C)": make_curve([20,25,30,35,40], [7.0,6.8,6.6,6.4,6.2]),
    "GSHP Std (18°C)": make_curve([20,25,30,35,40], [7.8,7.6,7.4,7.2,7.0]),
}

HEATING_ALL = {**HEATING_COP, **BOILER_EFF}
FUEL_BY_HEAT_CURVE = {k: "elec" for k in HEATING_COP}
FUEL_BY_HEAT_CURVE.update({k: "gas" for k in BOILER_EFF})

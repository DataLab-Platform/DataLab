# -*- coding: utf-8 -*-
"""
Tutorial 'Working with Spyder'
==============================

This tutorial shows how to use the DataLab remotely from a Python script running
outside DataLab (e.g. in Spyder).
"""

# ---- Begin of the example code ----
# %% Connecting to DataLab current session

from sigima.client import SimpleRemoteProxy

proxy = SimpleRemoteProxy()
proxy.connect()

# %% Visualizing 1D data from my work

from my_work import test_my_1d_algorithm

x, y = test_my_1d_algorithm()  # Here is all my research/technical work!
proxy.add_signal("My 1D result data", x, y)  # Let's visualize it in DataLab
proxy.compute_wiener()  # Denoise the signal using the Wiener filter


# %% Visualizing 2D data from my work

from my_work import test_my_2d_algorithm

z = test_my_2d_algorithm()[1]  # Here is all my research/technical work!
proxy.add_image("My 2D result data", z)  # Let's visualize it in DataLab

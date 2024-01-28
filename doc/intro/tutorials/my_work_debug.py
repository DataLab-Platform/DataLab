# -*- coding: utf-8 -*-
"""
Tutorial 'Working with Spyder'
==============================

This script is meant to be run in Spyder. It shows how to use the DataLab remotely
to help debugging a Python script running outside DataLab (e.g. in Spyder).

The main idea here is to enable the debug mode in the function while running the
script in Spyder, preferably with the debugger enabled, so that the script stops
at the breakpoint inside the function and let us inspect the variables in both
Spyder (with the Variable Explorer) and DataLab (with the visualizations).
"""

# ---- Begin of the example code ----
# %% Debugging my work with DataLab

from my_work import generate_2d_data

x, z = generate_2d_data(debug_with_datalab=True)

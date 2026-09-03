# DataLab template: Simple signal example

import numpy as np

from datalab.control.proxy import RemoteProxy

proxy = RemoteProxy()

# Create a sinc signal and add it to the Signals panel.
x = np.linspace(-10, 10, 500)
y = np.sin(x) / (x + 1e-9)
proxy.add_signal("sinc", x, y)
print("Created signal 'sinc'")

# Switch to the Signals panel so the new signal becomes visible.
proxy.set_current_panel("signal")

print("All done!")

# DataLab template: Image processing example

import numpy as np

from datalab.control.proxy import RemoteProxy

proxy = RemoteProxy()

# Generate a noisy 2D Gaussian.
size = 256
y, x = np.mgrid[0:size, 0:size]
cx, cy = size / 2, size / 2
img = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 30**2))
img += 0.05 * np.random.rand(size, size)

proxy.add_image("gaussian", img)
print("Created image 'gaussian'")
proxy.set_current_panel("image")

# Apply an FFT to the newly created image.
proxy.calc("fft")
print("FFT computed")

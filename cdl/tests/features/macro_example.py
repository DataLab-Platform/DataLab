# Simple DataLab macro example

import numpy as np

from cdl.proxy import RemoteProxy

proxy = RemoteProxy()

z = np.random.rand(20, 20)
proxy.add_image("toto", z)
proxy.compute_fft()

print("All done! :)")

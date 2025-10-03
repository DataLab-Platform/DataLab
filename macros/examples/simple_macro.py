# Simple DataLab macro example

import numpy as np

# When using this code outside of DataLab (which is possible!), you may use the
# client provided by the `Sigima` package (see https://pypi.org/project/sigima/),
# installed with `pip install sigima`, for example, instead of the full DataLab package:
#
# from sigima.client import SimpleRemoteProxy as RemoteProxy
from datalab.proxy import RemoteProxy

proxy = RemoteProxy()

z = np.random.rand(20, 20)
proxy.add_image("toto", z)
proxy.compute_fft()

print("All done!")

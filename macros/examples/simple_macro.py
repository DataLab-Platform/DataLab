# Simple DataLab macro example

import numpy as np

# When using this code outside of DataLab (which is possible!), you may use the
# *Simple DataLab Client* (see https://pypi.org/project/cdlclient/), installed with
# `pip install cdlclient`), as it's more lightweight than the full DataLab package:
#
# from cdlclient import SimpleRemoteProxy as RemoteProxy
from cdl.proxy import RemoteProxy

proxy = RemoteProxy()

z = np.random.rand(20, 20)
proxy.add_image("toto", z)
proxy.compute_fft()

print("All done!")

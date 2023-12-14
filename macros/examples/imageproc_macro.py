# Image Processing DataLab macro example

import numpy as np

# When using this code outside of DataLab (which is possible!), you may use the
# *Simple DataLab Client* (see https://pypi.org/project/cdlclient/), installed with
# `pip install cdlclient`), as it's more lightweight than the full DataLab package:
#
# from cdlclient import SimpleRemoteProxy as RemoteProxy
from cdl.proxy import RemoteProxy

proxy = RemoteProxy()

proxy.set_current_panel("image")
if len(proxy.get_object_uuids()) == 0:
    raise RuntimeError("No image to process!")

uuid = proxy.get_sel_object_uuids()[0]
image = proxy.get_object(uuid)

# Filter image with a custom kernel
data = np.array(image.data, copy=True)
kernel = np.array([[0, 1, 0, 0], [1, -4, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]])
pad = (data.shape[0] - kernel.shape[0]) // 2
kernel = np.pad(kernel, pad, mode="constant")
data = np.fft.ifft2(np.fft.fft2(data) * np.fft.fft2(kernel)).real
data -= data.min()
data /= data.max()
data *= 65535
data = data.astype(np.uint16)

# Add new image to the panel
proxy.add_image("My custom filtered data", data)

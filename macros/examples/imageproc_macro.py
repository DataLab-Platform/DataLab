# Image Processing DataLab macro example

import numpy as np
import scipy.ndimage as spi

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
def filter_func(values: np.ndarray) -> float:
    """Apply a custom denoising filter to an image.

    This filter averages the pixels in a 5x5 neighborhood, but gives less weight
    to pixels that significantly differ from the central pixel.
    """
    central_pixel = values[len(values) // 2]
    differences = np.abs(values - central_pixel)
    weights = np.exp(-differences / np.mean(differences))
    return np.average(values, weights=weights)


data = np.array(image.data, copy=True)
data = spi.generic_filter(data, filter_func, size=5)

# Add new image to the panel
proxy.add_image("My custom filtered data", data)

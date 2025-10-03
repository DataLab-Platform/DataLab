# Image Processing DataLab macro example

import numpy as np
import scipy.ndimage as spi

# When using this code outside of DataLab (which is possible!), you may use the
# client provided by the `Sigima` package (see https://pypi.org/project/sigima/),
# installed with `pip install sigima`, for example, instead of the full DataLab package:
#
# from sigima.client import SimpleRemoteProxy as RemoteProxy
from datalab.proxy import RemoteProxy

proxy = RemoteProxy()
proxy.set_current_panel("image")
image = proxy.get_object()
if image is None:
    raise RuntimeError("No image to process!")


# Filter image with a custom kernel
def weighted_average_denoise(values: np.ndarray) -> float:
    """Apply a custom denoising filter to an image.

    This filter averages the pixels in a 5x5 neighborhood, but gives less weight
    to pixels that significantly differ from the central pixel.
    """
    central_pixel = values[len(values) // 2]
    differences = np.abs(values - central_pixel)
    weights = np.exp(-differences / np.mean(differences))
    return np.average(values, weights=weights)


data = np.array(image.data, copy=True)
data = spi.generic_filter(data, weighted_average_denoise, size=5)

# Add new image to the panel
proxy.add_image("My custom filtered data", data)

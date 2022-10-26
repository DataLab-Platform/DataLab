# Macro simple example

import numpy as np

from codraft.remotecontrol import RemoteClient
from codraft.tests.data import create_2d_gaussian

print("toto")

remote = RemoteClient()
remote.try_and_connect()

z = create_2d_gaussian(2000, np.uint16)
remote.add_image("toto", z)

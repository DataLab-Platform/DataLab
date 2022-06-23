# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
Remote server/client test
"""

import xmlrpc.client
from io import BytesIO
from typing import List
from xmlrpc.server import SimpleXMLRPCRequestHandler, SimpleXMLRPCServer

import numpy as np
from qtpy import QtCore as QC

from codraft.core.gui.main import CodraFTMainWindow
from codraft.core.model.image import ImageParam, create_image
from codraft.core.model.signal import create_signal
from codraft.tests import codraft_app_context
from codraft.tests.data import create_2d_gaussian, create_test_signal1
from codraft.utils.qthelpers import qt_app_context

# === Python 2.7 client side:
# import xmlrpclib
# import numpy as np
# def array_to_binary(data):
#     """Convert NumPy array to XML-RPC Binary object, with shape and dtype"""
#     dbytes = BytesIO()
#     np.save(dbytes, data, allow_pickle=False)
#     return xmlrpc.Binary(dbytes.getvalue())
# s = xmlrpclib.ServerProxy("http://127.0.0.1:8000")
# data = np.array([[3, 4, 5], [7, 8, 0]], dtype=np.uint16)
# s.add_image("toto", array_to_binary(data))


class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ("/RPC2",)


def array_to_rpcbinary(data: np.ndarray) -> xmlrpc.client.Binary:
    """Convert NumPy array to XML-RPC Binary object, with shape and dtype"""
    dbytes = BytesIO()
    np.save(dbytes, data, allow_pickle=False)
    return xmlrpc.client.Binary(dbytes.getvalue())


def rpcbinary_to_array(binary: xmlrpc.client.Binary) -> np.ndarray:
    """Convert XML-RPC binary to NumPy array"""
    dbytes = BytesIO(binary.data)
    return np.load(dbytes, allow_pickle=False)


class DummyCodraFTWindow:
    """Dummy CodraFT window, for test only"""

    def switch_to_signal_panel(self):
        """Switch to signal panel"""
        print(self.switch_to_signal_panel.__doc__)

    def switch_to_image_panel(self):
        """Switch to image panel"""
        print(self.switch_to_image_panel.__doc__)

    def has_objects(self):
        """Return True if sig/ima panels have any object"""
        return False

    def reset_all(self):
        """Reset all application data"""
        print(self.reset_all.__doc__)

    def save_to_h5_file(self, filename=None):
        """Save to a CodraFT HDF5 file"""
        print(self.save_to_h5_file.__doc__, filename)

    def open_h5_files(
        self,
        h5files: List[str] = None,
        import_all: bool = None,
        reset_all: bool = None,
    ) -> None:
        """Open a CodraFT HDF5 file or import from any other HDF5 file"""
        print(self.open_h5_files.__doc__, h5files, import_all, reset_all)

    def import_h5_file(self, filename: str, reset_all: bool = None) -> None:
        """Open CodraFT HDF5 browser to Import HDF5 file"""
        print(self.import_h5_file.__doc__, filename, reset_all)

    def add_object(self, obj, refresh=True):
        """Add object - signal or image"""
        print(self.add_object.__doc__, obj, refresh)


class RPCServerThread(QC.QThread):
    """XML-RPC server thread"""

    SIG_ADD_OBJECT = QC.Signal(object)

    def __init__(self, port: str, win: CodraFTMainWindow):
        QC.QThread.__init__(self)
        self.port = port
        self.SIG_ADD_OBJECT.connect(win.add_object)
        # TODO: Add signals for other methods (see Dummy main window object)

    def run(self):
        """Thread execution method"""
        with SimpleXMLRPCServer(
            ("127.0.0.1", self.port), requestHandler=RequestHandler, logRequests=False
        ) as server:
            server.register_introspection_functions()
            server.register_function(self.add_signal)
            server.register_function(self.add_image)
            server.serve_forever()

    def add_signal(
        self,
        title: str,
        xbinary: xmlrpc.client.Binary,
        ybinary: xmlrpc.client.Binary,
        xunit: str = None,
        yunit: str = None,
        xlabel: str = None,
        ylabel: str = None,
    ):
        """Add signal data to CodraFT"""
        xdata = rpcbinary_to_array(xbinary)
        ydata = rpcbinary_to_array(ybinary)
        signal = create_signal(title, xdata, ydata)
        signal.xunit = xunit
        signal.yunit = yunit
        signal.xlabel = xlabel
        signal.ylabel = ylabel
        self.SIG_ADD_OBJECT.emit(signal)
        return True

    def add_image(
        self,
        title: str,
        zbinary: xmlrpc.client.Binary,
        xunit: str = None,
        yunit: str = None,
        zunit: str = None,
        xlabel: str = None,
        ylabel: str = None,
        zlabel: str = None,
    ):
        """Add image data to CodraFT"""
        data = rpcbinary_to_array(zbinary)
        image = create_image(title, data)
        image.xunit = xunit
        image.yunit = yunit
        image.zunit = zunit
        image.xlabel = xlabel
        image.ylabel = ylabel
        image.zlabel = zlabel
        self.SIG_ADD_OBJECT.emit(image)
        return True


def test(port=8000):
    """Remote server/client test"""
    win = DummyCodraFTWindow()
    # with qt_app_context() as qapp:
    with codraft_app_context(console=False) as win:
        server_thread = RPCServerThread(port, win)
        server_thread.start()
        # qapp.exec()

        s = xmlrpc.client.ServerProxy(f"http://127.0.0.1:{port}")
        x, y = create_test_signal1().get_data()
        print(s.add_signal("tutu", array_to_rpcbinary(x), array_to_rpcbinary(y)))
        z = create_2d_gaussian(2000, np.uint16)
        print(s.add_image("toto", array_to_rpcbinary(z)))
        print(s.system.listMethods())


if __name__ == "__main__":
    test()

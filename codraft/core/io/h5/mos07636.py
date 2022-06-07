# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
CodraFT MOS07636 HDF5 format support
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from codraft.core.io.h5 import common, utils
from codraft.core.model.image import create_image
from codraft.core.model.signal import create_signal
from codraft.utils.misc import to_string

# Add ignored dataset names
common.NODE_FACTORY.add_ignored_datasets(("PALETTE",))


class BaseMOS07636Node(common.BaseNode):
    """Object representing a HDF5 node, according to MOS07636"""

    ATTR_PATTERN = (None, None)

    def __init__(self, h5file, dset):
        super().__init__(h5file, dset)
        self.xunit = None
        self.yunit = None
        self.zunit = None
        self.xlabel = None
        self.ylabel = None
        self.zlabel = None

    @classmethod
    def match(cls, dset):
        """Return True if h5 dataset match node pattern"""
        name, value = cls.ATTR_PATTERN
        return dset.attrs.get(name) == value

    @property
    def data(self):
        """Data associated to node, if available"""
        return self.dset["valeur"][()]

    @property
    def description(self):
        """Return node description"""
        desc = utils.process_scalar_value(self.dset, "description", utils.fix_ldata)
        if desc is not None:
            return desc
        return super().description

    def create_object(self):
        """Create native object, if supported"""
        self.xunit, self.yunit, self.zunit = utils.process_label(self.dset, "unite")
        self.xlabel, self.ylabel, self.zlabel = utils.process_label(self.dset, "label")
        for label in ("description", "source"):
            val = utils.process_scalar_value(self.dset, label, utils.fix_ldata)
            if val is not None:
                self.metadata[label] = val

    def get_scalar(self, dnames, pname=None, callback=None, ancestor=0):
        """Try and get value from h5 file using callback"""
        nameprefix = "/".join(self.dset.name.split("/")[:-ancestor]) + "/"
        pname = "valeur" if pname is None else pname
        callback = utils.fix_ndata if callback is None else callback
        for dname in dnames:
            dataset = self.h5file.get(nameprefix + dname)
            if dataset is not None:
                return utils.process_scalar_value(dataset, pname, callback)
        return None


class ScalarNode(BaseMOS07636Node):
    """Object representing a scalar HDF5 node, according to MOS07636"""

    ATTR_PATTERN = ("CLASS", b"ELEMENTAIRE")

    @property
    def data(self):
        """Data associated to node, if available"""
        try:
            return super().data
        except ValueError:
            #  Handles invalid scalar datasets...
            return self.dset[()]

    @property
    def icon_name(self):
        """Icon name associated to node"""
        return "h5scalar.svg"

    @property
    def text(self):
        """Return node textual representation"""
        text = to_string(self.data)
        if self.xunit:
            text += " " + self.xunit
        return text


common.NODE_FACTORY.register(ScalarNode)


class SignalNode(BaseMOS07636Node):
    """Object representing a Signal HDF5 node, according to MOS07636"""

    IS_ARRAY = True
    ATTR_PATTERN = ("CLASS", b"COURBE")

    @property
    def icon_name(self):
        """Icon name associated to node"""
        return "signal.svg"

    @property
    def text(self):
        """Return node textual representation"""

    def create_object(self):
        """Create native object, if supported"""
        super().create_object()
        obj = create_signal(
            self.object_title,
            units=(self.xunit, self.yunit),
            labels=(self.xlabel, self.ylabel),
        )
        self.set_signal_data(obj)
        return obj


common.NODE_FACTORY.register(SignalNode)


class ImageNode(BaseMOS07636Node):
    """Object representing an Image HDF5 node, according to MOS07636"""

    IS_ARRAY = True
    ATTR_PATTERN = ("CLASS", b"IMAGE")

    @property
    def icon_name(self):
        """Icon name associated to node"""
        return "image.svg"

    @property
    def text(self):
        """Return node textual representation"""

    def create_object(self):
        """Create native object, if supported"""
        super().create_object()
        obj = create_image(
            self.object_title,
            units=(self.xunit, self.yunit, self.zunit),
            labels=(self.xlabel, self.ylabel, self.zlabel),
        )
        self.set_image_data(obj)
        x0, y0 = utils.process_xy_values(self.dset, "origine")
        if x0 is not None and y0 is not None:
            obj.x0, obj.y0 = x0, y0
        dx, dy = utils.process_xy_values(self.dset, "resolution")
        if dx is not None and dy is not None:
            obj.dx, obj.dy = dx, dy
        dnames = ("Parametres_IMG/MargeGauche", "Parametres_ACQ/MargeGauche")
        marge_gauche = self.get_scalar(dnames, ancestor=2)
        if marge_gauche:
            obj.x0 = marge_gauche
        dnames = ("Parametres_IMG/MargeHaute", "Parametres_ACQ/MargeHaute")
        marge_haute = self.get_scalar(dnames, ancestor=2)
        if marge_haute:
            obj.y0 = marge_haute
        return obj


common.NODE_FACTORY.register(ImageNode)

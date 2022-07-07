# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
CodraFT MOS07636 HDF5 format support
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
from guidata.utils import update_dataset

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
        self.__obj_templates = []

    def add_object_default_values(self, **template):
        """Add object default values (object template)"""
        self.__obj_templates.append(template)

    def update_from_object_default_values(self, obj):
        """Update object (signal/image) from default values (template), if available"""
        for template in self.__obj_templates:
            update_dataset(obj, template)

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


class ScalarNode(BaseMOS07636Node):
    """Object representing a scalar HDF5 node, according to MOS07636"""

    ATTR_PATTERN = ("CLASS", b"ELEMENTAIRE")

    def __init__(self, h5file, dset):
        super().__init__(h5file, dset)
        self.xunit = utils.process_scalar_value(self.dset, "unite", self._fix_unit)

    @staticmethod
    def _fix_unit(scdata):
        """Fix unit data"""
        data = scdata[0]
        if not isinstance(data, bytes):
            data = data[0]  # Should not be necessary (invalid format)
        return utils.fix_ldata(data)

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
        suffix = "" if self.xunit is None else " " + self.xunit
        if not text.endswith(suffix):  # Should not be necessary (invalid format)
            text += suffix
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
        self.update_from_object_default_values(obj)
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
        self.update_from_object_default_values(obj)
        return obj


common.NODE_FACTORY.register(ImageNode)


def handle_margins_vimba(importer: common.H5Importer):
    """Post-collection trigger handling image margins when available (Vimba Cameras)"""
    try:
        image = importer.get("/Acquisition/AcquisitionBrute")
        margegauche = importer.get("/Parametres_ACQ/MargeGauche")
        margehaute = importer.get("/Parametres_ACQ/MargeHaute")
        binningx = importer.get("/Parametres_ACQ/BinningX")
        binningy = importer.get("/Parametres_ACQ/BinningY")
    except KeyError:
        return
    image.add_object_default_values(
        x0=margegauche.data, y0=margehaute.data, dx=binningx.data, dy=binningy.data
    )


common.NODE_FACTORY.add_post_trigger(handle_margins_vimba)


def handle_margins_istar(importer: common.H5Importer):
    """Post-collection trigger handling image margins when available (IStar Cameras)"""
    try:
        image = importer.get("/Entrees/Acquisition/AcquisitionBrute")
        margegauche = importer.get("/Entrees/Parametres_IMG/MargeGauche")
        margehaute = importer.get("/Entrees/Parametres_IMG/MargeHaute")
        binningx = importer.get("/Entrees/Parametres_IMG/BinningX")
        binningy = importer.get("/Entrees/Parametres_IMG/BinningY")
    except KeyError:
        return
    image.add_object_default_values(
        x0=margegauche.data, y0=margehaute.data, dx=binningx.data, dy=binningy.data
    )


common.NODE_FACTORY.add_post_trigger(handle_margins_istar)


def handle_streakcameratimeaxis(importer: common.H5Importer):
    """Post-collection trigger handling streak X-axis time conv. when available"""
    try:
        image = importer.get("/Acquisition/AcquisitionCorrigee")
        tempspixel = importer.get("/Acquisition/TempsPixel")
        offsettemporel = importer.get("/Acquisition/OffsetTemporel")
    except KeyError:
        return
    image.add_object_default_values(
        x0=offsettemporel.data, dx=tempspixel.data, xunit=tempspixel.xunit
    )


common.NODE_FACTORY.add_post_trigger(handle_streakcameratimeaxis)

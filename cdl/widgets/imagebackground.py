# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Image background selection dialog."""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from guidata.configtools import get_icon
from plotpy.builder import make
from plotpy.plot import PlotDialog

from cdl.config import _

if TYPE_CHECKING:
    from plotpy.items import ImageItem, RangeComputation2d, RectangleShape
    from qtpy.QtWidgets import QWidget

    from cdl.obj import ImageObj


class ImageBackgroundDialog(PlotDialog):
    """Image background selection dialog.

    Args:
        image: image object
        parent: parent widget. Defaults to None.
    """

    def __init__(self, image: ImageObj, parent: QWidget | None = None) -> None:
        self.__background: float | None = None
        self.__indexrange: tuple[int, int, int, int] | None = None
        self.imageitem: ImageItem | None = None
        self.rectarea: RectangleShape | None = None
        self.comput2d: RangeComputation2d | None = None
        super().__init__(
            title=_("Image background selection"), edit=True, parent=parent
        )
        self.setObjectName("backgroundselection")
        if parent is None:
            self.setWindowIcon(get_icon("DataLab.svg"))
        self.__image = image.copy()
        self.__setup_dialog()

    def __compute_background(
        self,
        x: np.ndarray,  # pylint: disable=unused-argument
        y: np.ndarray,  # pylint: disable=unused-argument
        z: np.ndarray,
    ) -> float:
        """Compute background value"""
        x0, y0, x1, y1 = self.rectarea.get_rect()
        ix0, iy0, ix1, iy1 = self.imageitem.get_closest_index_rect(x0, y0, x1, y1)
        self.__indexrange = ix0, iy0, ix1, iy1
        self.__background = z.mean()
        return self.__background

    def __setup_dialog(self) -> None:
        """Setup dialog box"""
        obj = self.__image
        self.imageitem = obj.make_item()
        plot = self.get_plot()
        self.rectarea = make.rectangle(
            obj.x0, obj.y0, obj.xc, obj.yc, _("Background area")
        )
        self.comput2d = make.computation2d(
            self.rectarea,
            "TL",
            _("Background value:") + " %g",
            self.imageitem,
            self.__compute_background,
        )
        for item in (self.imageitem, self.rectarea, self.comput2d):
            plot.add_item(item)
        plot.replot()
        plot.set_active_item(self.rectarea)

    def get_background(self) -> float:
        """Get background value"""
        return self.__background

    def get_index_range(self) -> tuple[int, int, int, int]:
        """Get index range

        Returns:
            Index range (x0, y0, x1, y1)
        """
        return self.__indexrange

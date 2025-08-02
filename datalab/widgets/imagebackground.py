# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Image background selection dialog."""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from guidata.configtools import get_icon
from plotpy.builder import make
from plotpy.plot import PlotDialog

from datalab.adapters_plotpy.factories import create_adapter_from_object
from datalab.config import _

if TYPE_CHECKING:
    from plotpy.items import ImageItem, RangeComputation2d, RectangleShape
    from qtpy.QtWidgets import QWidget

    from sigima.objects import ImageObj


class ImageBackgroundDialog(PlotDialog):
    """Image background selection dialog.

    Args:
        image: image object
        parent: parent widget. Defaults to None.
    """

    def __init__(self, image: ImageObj, parent: QWidget | None = None) -> None:
        self.__background: float | None = None
        self.__rect_coords: tuple[float, float, float, float] | None = None
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
        self.__rect_coords = self.rectarea.get_rect()
        self.__background = z.mean()
        return self.__background

    def __setup_dialog(self) -> None:
        """Setup dialog box"""
        obj = self.__image
        self.imageitem = create_adapter_from_object(obj).make_item()
        plot = self.get_plot()
        self.rectarea = make.rectangle(
            obj.x0, obj.y0, obj.xc + obj.dx, obj.yc + obj.dy, _("Background area")
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

    def get_rect_coords(self) -> tuple[float, float, float, float]:
        """Get rectangle coordinates

        Returns:
            tuple: rectangle coordinates (x0, y0, x1, y1)

        Raises:
            ValueError: if rectangle coordinates are not set
        """
        if self.__rect_coords is None:
            raise ValueError("Rectangle coordinates not set")
        return self.__rect_coords

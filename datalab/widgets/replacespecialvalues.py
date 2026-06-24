# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Replace special values dialog
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Custom dialog widget for the "Replace special values" processing feature.

The dialog extends the standard guidata DataSet edition layout with:

- **Count display**: colored badges showing the number (and percentage) of
  NaN, +Inf and -Inf samples present in the source signal or image.
- **Kernel preview**: visual grid of the active neighbor mask, displayed
  whenever a neighbor-based strategy is selected for any of the three targets.
- **Live preview update**: the kernel preview is refreshed automatically
  whenever a parameter value changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from guidata.dataset.qtwidgets import DataSetEditLayout
from guidata.qthelpers import exec_dialog
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW
from sigima.enums import ReplacementStrategyImage, ReplacementStrategySignal
from sigima.proc.base import (
    ReplaceSpecialValuesImageParam,
    ReplaceSpecialValuesSignalParam,
)
from sigima.tools.image.replace_values import count_special_values_2d
from sigima.tools.signal.replace_values import count_special_values

from datalab.config import _

if TYPE_CHECKING:
    from qtpy.QtWidgets import QWidget

# -- Badge configuration --------------------------------------------------------

_BADGE_COLORS: dict[str, str] = {
    "nan": "#e74c3c",
    "posinf": "#e67e22",
    "neginf": "#3498db",
}

_BADGE_LABELS: dict[str, str] = {
    "nan": "NaN",
    "posinf": "+\u221e",
    "neginf": "\u2212\u221e",
}

# -- Helper widgets -------------------------------------------------------------


def _make_count_badge(key: str, count: int, total: int) -> QW.QLabel:
    """Create a rich-text label with a colored dot and count information."""
    color = _BADGE_COLORS[key]
    label = _BADGE_LABELS[key]
    if total > 0 and count > 0:
        pct = count / total * 100
        text = (
            f'<span style="color:{color}">\u25cf</span> '
            f"{label}: <b>{count}</b> ({pct:.1f}%)"
        )
    else:
        text = f'<span style="color:{color}">\u25cf</span> {label}: <b>{count}</b>'
    lbl = QW.QLabel(text)
    lbl.setTextFormat(QC.Qt.RichText)
    return lbl


class _KernelPreviewWidget(QW.QGroupBox):
    """Visual grid showing kernel/mask weights.

    The center cell is highlighted in red, other cells use a blue intensity
    proportional to their weight.
    """

    def __init__(self, title: str = "", parent: QWidget | None = None) -> None:
        super().__init__(title or _("Kernel / Mask preview"), parent)
        layout = QW.QVBoxLayout(self)
        self._info = QW.QLabel()
        layout.addWidget(self._info)
        self._table = QW.QTableWidget()
        self._table.setEditTriggers(QW.QAbstractItemView.NoEditTriggers)
        self._table.setSelectionMode(QW.QAbstractItemView.NoSelection)
        self._table.setMaximumHeight(140)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setVisible(False)
        layout.addWidget(self._table)
        self.setVisible(False)

    # -- public API -------------------------------------------------------------

    def show_1d(
        self, kernel: np.ndarray, info: str = "", show_values: bool = True
    ) -> None:
        """Display a 1-D kernel as a single-row table.

        Args:
            kernel: 1-D weight array.
            info: label text shown above the table.
            show_values: if False, cells show only the coloured background
                (useful for non-weighted stat filters like min/max/median).
        """
        self._info.setText(info)
        n = len(kernel)
        self._table.setRowCount(1)
        self._table.setColumnCount(n)
        center = n // 2
        vmax = float(np.max(np.abs(kernel))) or 1.0
        for c in range(n):
            val = kernel[c]
            text = f"{val:.3g}" if show_values else ""
            item = QW.QTableWidgetItem(text)
            item.setTextAlignment(QC.Qt.AlignCenter)
            if c == center:
                item.setBackground(QG.QColor(255, 180, 180, 200))
            else:
                alpha = int(abs(val) / vmax * 180) if show_values else 120
                item.setBackground(QG.QColor(180, 180, 255, alpha))
            self._table.setItem(0, c, item)
        self._table.resizeColumnsToContents()
        self._table.resizeRowsToContents()
        self.setVisible(True)

    def show_2d(
        self, kernel: np.ndarray, info: str = "", show_values: bool = True
    ) -> None:
        """Display a 2-D kernel as a coloured grid.

        Args:
            kernel: 2-D weight array.
            info: label text shown above the table.
            show_values: if False, cells show only the coloured background
                (useful for non-weighted stat filters like min/max/median).
        """
        self._info.setText(info)
        rows, cols = kernel.shape
        self._table.setRowCount(rows)
        self._table.setColumnCount(cols)
        cr, cc = rows // 2, cols // 2
        vmax = float(np.max(np.abs(kernel))) or 1.0
        for r in range(rows):
            for c in range(cols):
                val = kernel[r, c]
                text = f"{val:.3g}" if show_values else ""
                item = QW.QTableWidgetItem(text)
                item.setTextAlignment(QC.Qt.AlignCenter)
                if r == cr and c == cc:
                    item.setBackground(QG.QColor(255, 180, 180, 200))
                else:
                    alpha = int(abs(val) / vmax * 180) if show_values else 120
                    item.setBackground(QG.QColor(180, 180, 255, alpha))
                self._table.setItem(r, c, item)
        self._table.resizeColumnsToContents()
        self._table.resizeRowsToContents()
        self.setVisible(True)

    def hide_preview(self) -> None:
        """Clear and hide the preview group."""
        self._table.setRowCount(0)
        self._table.setColumnCount(0)
        self.setVisible(False)


# -- Main dialog ----------------------------------------------------------------


class ReplaceSpecialValuesDialog(QW.QDialog):
    """Custom dialog for the *Replace special values* feature.

    Shows coloured count badges (NaN, +Inf, -Inf) at the top, the standard
    DataSet parameter editing form in the middle, and a kernel/mask preview
    at the bottom when a neighbor-based strategy is selected.

    Args:
        instance: DataSet parameter instance to edit.
        counts: dictionary with keys ``"nan"``, ``"posinf"``, ``"neginf"``.
        total_size: total number of data points.
        is_image: ``True`` for image parameters, ``False`` for signals.
        parent: parent widget.
    """

    def __init__(
        self,
        instance: ReplaceSpecialValuesSignalParam | ReplaceSpecialValuesImageParam,
        counts: dict[str, int],
        total_size: int,
        is_image: bool,
        info_message: str | None = None,
        can_apply: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.instance = instance
        self._is_image = is_image
        self._info_message = info_message
        self._can_apply = can_apply
        self.setWindowTitle(instance.get_title())
        self.setMinimumWidth(480)

        main_layout = QW.QVBoxLayout(self)

        # --- Count badges ---
        count_row = QW.QHBoxLayout()
        for key in ("nan", "posinf", "neginf"):
            count_row.addWidget(_make_count_badge(key, counts[key], total_size))
        count_row.addStretch()
        main_layout.addLayout(count_row)

        line = QW.QFrame()
        line.setFrameShape(QW.QFrame.HLine)
        line.setFrameShadow(QW.QFrame.Sunken)
        main_layout.addWidget(line)

        if info_message:
            info_label = QW.QLabel(info_message)
            info_label.setWordWrap(True)
            info_label.setStyleSheet(
                "background-color: #eef6ff; border: 1px solid #8fb7e8; "
                "padding: 8px; border-radius: 4px;"
            )
            main_layout.addWidget(info_label)

        # --- DataSet editing ---
        grid = QW.QGridLayout()
        self.edit_layout: DataSetEditLayout | None = None
        edit_layout = DataSetEditLayout(
            self, instance, grid, change_callback=self._on_change
        )
        self.edit_layout = edit_layout
        main_layout.addLayout(grid)

        # --- Kernel previews (one per group) ---
        self._kernel_previews: dict[str, _KernelPreviewWidget] = {}
        for key in ("nan", "posinf", "neginf"):
            label = _BADGE_LABELS[key]
            color = _BADGE_COLORS[key]
            preview = _KernelPreviewWidget(
                title=_("{label} — Kernel / Mask preview").format(label=label),
            )
            preview.setStyleSheet(
                f"_KernelPreviewWidget {{ border: 1px solid {color}; }}"
            )
            self._kernel_previews[key] = preview
            main_layout.addWidget(preview)

        # --- Buttons ---
        bbox = QW.QDialogButtonBox(QW.QDialogButtonBox.Ok | QW.QDialogButtonBox.Cancel)
        bbox.accepted.connect(self.accept)
        bbox.rejected.connect(self.reject)
        ok_button = bbox.button(QW.QDialogButtonBox.Ok)
        if ok_button is not None:
            ok_button.setEnabled(can_apply)
        main_layout.addWidget(bbox)

        # Initial preview
        self._refresh_preview()

    # -- internal helpers -------------------------------------------------------

    def _on_change(self) -> None:
        """Slot called whenever a DataSet widget value changes."""
        if self.edit_layout is None:
            return
        # Sync current widget values → DataSet so the preview reads live data
        self.edit_layout.accept_changes()
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        """Update every kernel preview independently."""
        p = self.instance
        for prefix in ("nan", "posinf", "neginf"):
            strategy = getattr(p, f"{prefix}_strategy")
            preview = self._kernel_previews[prefix]
            shown = (
                self._try_show_image_kernel(strategy, prefix, preview)
                if self._is_image
                else self._try_show_signal_kernel(strategy, prefix, preview)
            )
            if not shown:
                preview.hide_preview()

    def _try_show_signal_kernel(
        self,
        strategy: ReplacementStrategySignal,
        prefix: str,
        preview: _KernelPreviewWidget,
    ) -> bool:
        """Show kernel for signal neighbor strategies. Returns True if shown."""
        _S = ReplacementStrategySignal
        p = self.instance
        if strategy in (
            _S.NEIGHBOR_MIN,
            _S.NEIGHBOR_MAX,
            _S.NEIGHBOR_MEAN,
            _S.NEIGHBOR_MEDIAN,
        ):
            n = getattr(p, f"{prefix}_neighbor_size", 3)
            size = 2 * n + 1
            kernel = np.ones(size) / size
            show_values = strategy == _S.NEIGHBOR_MEAN
            preview.show_1d(
                kernel,
                _("±{n} points").format(n=n),
                show_values=show_values,
            )
            return True
        return False

    def _try_show_image_kernel(
        self,
        strategy: ReplacementStrategyImage,
        prefix: str,
        preview: _KernelPreviewWidget,
    ) -> bool:
        """Show kernel for image neighbor strategies. Returns True if shown."""
        _S = ReplacementStrategyImage
        p = self.instance
        if strategy in (
            _S.NEIGHBOR_MIN,
            _S.NEIGHBOR_MAX,
            _S.NEIGHBOR_MEAN,
            _S.NEIGHBOR_MEDIAN,
        ):
            n = getattr(p, f"{prefix}_neighbor_size", 3)
            size = 2 * n + 1
            kernel = np.ones((size, size)) / (size * size)
            show_values = strategy == _S.NEIGHBOR_MEAN
            preview.show_2d(
                kernel,
                _("±{n} rows × ±{n} columns").format(n=n),
                show_values=show_values,
            )
            return True
        return False

    # -- public interface -------------------------------------------------------

    def accept(self) -> None:
        """Validate all widget values, then commit to the DataSet."""
        if not self._can_apply:
            return
        if self.edit_layout is not None:
            if not self.edit_layout.check_all_values():
                return
            self.edit_layout.accept_changes()
        super().accept()


# -- DataLab-specific parameter subclasses --------------------------------------


class ReplaceSpecialValuesSignalParamDL(ReplaceSpecialValuesSignalParam):
    """Signal parameter subclass with counts and custom dialog.

    Overrides :meth:`edit` to display :class:`ReplaceSpecialValuesDialog`.
    """

    _counts: dict[str, int]
    _total_size: int

    def update_from_obj(self, obj: object) -> None:
        """Pre-analyse the signal to compute special-value counts."""
        _, y = obj.get_data()  # type: ignore[union-attr]
        self._counts = count_special_values(y)
        self._total_size = int(y.size)

    def create_dialog(
        self,
        parent: QWidget | None = None,
        object_name: str | None = None,
    ) -> ReplaceSpecialValuesDialog:
        """Create the custom replace-special-values dialog."""
        counts = getattr(self, "_counts", {"nan": 0, "posinf": 0, "neginf": 0})
        total = getattr(self, "_total_size", 0)
        dlg = ReplaceSpecialValuesDialog(
            self, counts, total, is_image=False, parent=parent
        )
        dlg.setObjectName(object_name or self.__class__.__name__ + "Dialog")
        return dlg

    def edit(
        self,
        parent: QWidget | None = None,
        apply: object = None,
        wordwrap: bool = True,
        size: object = None,
        object_name: str | None = None,
    ) -> int:
        """Open the custom replace-special-values dialog."""
        dlg = self.create_dialog(parent=parent, object_name=object_name)
        return exec_dialog(dlg)


class ReplaceSpecialValuesImageParamDL(ReplaceSpecialValuesImageParam):
    """Image parameter subclass with counts and custom dialog.

    Overrides :meth:`edit` to display :class:`ReplaceSpecialValuesDialog`.
    """

    _counts: dict[str, int]
    _total_size: int
    _can_apply: bool
    _info_message: str | None

    def update_from_obj(self, obj: object) -> None:
        """Pre-analyse the image to compute special-value counts."""
        data = obj.data  # type: ignore[union-attr]
        self._counts = count_special_values_2d(data)
        self._total_size = int(data.size)
        if np.issubdtype(data.dtype, np.integer):
            self._can_apply = False
            self._info_message = _(
                "This image uses an integer data type, so it cannot contain NaN "
                "or infinite values. Replace special values is therefore not "
                "applicable."
            )
        else:
            self._can_apply = True
            self._info_message = None

    def create_dialog(
        self,
        parent: QWidget | None = None,
        object_name: str | None = None,
    ) -> ReplaceSpecialValuesDialog:
        """Create the custom replace-special-values dialog."""
        counts = getattr(self, "_counts", {"nan": 0, "posinf": 0, "neginf": 0})
        total = getattr(self, "_total_size", 0)
        dlg = ReplaceSpecialValuesDialog(
            self,
            counts,
            total,
            is_image=True,
            info_message=getattr(self, "_info_message", None),
            can_apply=getattr(self, "_can_apply", True),
            parent=parent,
        )
        dlg.setObjectName(object_name or self.__class__.__name__ + "Dialog")
        return dlg

    def edit(
        self,
        parent: QWidget | None = None,
        apply: object = None,
        wordwrap: bool = True,
        size: object = None,
        object_name: str | None = None,
    ) -> int:
        """Open the custom replace-special-values dialog."""
        dlg = self.create_dialog(parent=parent, object_name=object_name)
        return exec_dialog(dlg)

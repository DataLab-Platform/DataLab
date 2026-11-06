"""Processing editors keep Apply semantics with locally enabled sliders."""

from __future__ import annotations

from collections.abc import Callable

from guidata.dataset.qtwidgets import DataSetEditGroupBox, DataSetEditLayout

__all__ = ["ProcessingParametersEditor"]


class ProcessingParametersEditor(DataSetEditGroupBox):
    """Notify the owner through the existing layout callback, without auto-Apply."""

    def __init__(self, *args, **kwargs):
        self.on_change: Callable | None = None
        self.dragging = False
        super().__init__(*args, **kwargs)

    def get_edit_layout(self) -> DataSetEditLayout:
        """Enable presentation options without changing shared DataItems."""
        return DataSetEditLayout(
            self,
            self.dataset,
            self.grid_layout,
            change_callback=self.change_callback,
            auto_sliders=True,
            slider_callback=self._slider_gesture,
        )

    def change_callback(self) -> None:
        """Preserve Apply activation, then notify the owning processing tab."""
        super().change_callback()
        if self.on_change is not None:
            self.on_change()

    def _slider_gesture(self, pressed: bool) -> None:
        self.dragging = pressed
        if self.on_change is not None:
            self.on_change()

    def set(self, check: bool = True) -> None:
        """Do not emit Apply while any active field is invalid."""
        if self.edit.check_all_values():
            super().set(check=True)

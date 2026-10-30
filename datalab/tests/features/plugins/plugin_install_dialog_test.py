# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Installing and uninstalling plugins from a file in the configuration dialog."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from guidata.qthelpers import qt_wait_until
from qtpy import QtWidgets as QW

from datalab import __version__
from datalab.config import Conf
from datalab.config.persistence import has_persisted_option, remove_persisted_option
from datalab.gui.plugins.config import PluginConfigDialog, PluginState
from datalab.gui.plugins.install import (
    InstalledPluginsWidget,
    PluginConsentDialog,
    run_plugin_task,
)
from datalab.plugins import PluginRegistry
from datalab.plugins import base as plugin_base
from datalab.plugins.store import InstalledPluginStore
from datalab.tests import datalab_test_app_context
from datalab.tests.backbone.plugins.wheel_factory import (
    make_plugin_wheel,
    wheel_filename,
)

PACKAGE = "datalab_install_ui_probe"
PLUGIN_ID = "org.example.install-ui-probe"


@pytest.fixture(name="store")
def fixture_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> InstalledPluginStore:
    """Use an empty installed plugin store and restore the import state."""
    store = InstalledPluginStore(
        str(tmp_path / "installed_plugins"),
        host_distributions={"datalab-platform": __version__},
    )
    monkeypatch.setattr(plugin_base, "_INSTALLED_PLUGIN_STORE", store)
    monkeypatch.setattr(sys, "path", list(sys.path))
    yield store
    for name in [name for name in sys.modules if name.startswith(PACKAGE)]:
        del sys.modules[name]


def write_wheel(directory: Path) -> str:
    """Write the probe plugin wheel and return its path."""
    distribution = "datalab-install-ui-probe"
    path = directory / wheel_filename(distribution, "1.0.0")
    path.write_bytes(
        make_plugin_wheel(
            distribution=distribution,
            version="1.0.0",
            package=PACKAGE,
            plugin_id=PLUGIN_ID,
            requires_dist=("datalab-platform>=1.0",),
        )
    )
    return str(path)


def test_installed_wheel_is_enabled_loaded_and_uninstalled(
    store: InstalledPluginStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An installed plugin is enabled even when the enabled list is explicit."""
    path = write_wheel(tmp_path)
    had_config = has_persisted_option(Conf, "plugins_enabled_list")
    original_enabled_list = Conf.plugins_enabled_list.get(None)
    consented: list[dict] = []
    monkeypatch.setattr(
        QW.QMessageBox, "question", lambda *_args, **_kwargs: QW.QMessageBox.Yes
    )
    try:
        with datalab_test_app_context(console=False) as win:
            Conf.plugins_enabled_list.set(
                [plugin.plugin_id for plugin in PluginRegistry.get_plugins()]
            )
            dialog = PluginConfigDialog(win)
            widget = dialog.installed_plugins_widget
            assert dialog.tabs.indexOf(widget) == 3
            monkeypatch.setattr(widget, "select_file", lambda: path)
            monkeypatch.setattr(
                widget,
                "confirm_installation",
                lambda manifest, origin=None: consented.append(manifest) or True,
            )

            widget.install_button.click()
            QW.QApplication.processEvents()

            assert [manifest["distribution"] for manifest in consented] == [
                "datalab-install-ui-probe"
            ]
            assert PluginRegistry.get_plugin(PLUGIN_ID) is not None
            assert PLUGIN_ID in Conf.plugins_enabled_list.get(None)
            (row,) = [
                row
                for row in dialog.plugin_widgets
                if row.plugin_class.get_plugin_id() == PLUGIN_ID
            ]
            assert row.state == PluginState.ENABLED
            (item,) = widget.item_widgets
            assert item.plugin == store.get_plugins()[0]

            item.uninstall_button.click()
            QW.QApplication.processEvents()

            assert not store.get_plugins()
            assert not widget.item_widgets
            assert PluginRegistry.get_plugin(PLUGIN_ID) is None
            assert PLUGIN_ID not in [
                row.plugin_class.get_plugin_id() for row in dialog.plugin_widgets
            ]
            dialog.close()
            dialog.deleteLater()
    finally:
        if had_config:
            Conf.plugins_enabled_list.set(original_enabled_list)
        else:
            remove_persisted_option(Conf, "plugins_enabled_list")


def test_refused_or_invalid_files_are_not_installed(
    store: InstalledPluginStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing is installed without consent or from an invalid file."""
    _app = QW.QApplication.instance() or QW.QApplication([])
    reloads: list[bool] = []
    warnings: list[str] = []
    widget = InstalledPluginsWidget(lambda: reloads.append(True))
    monkeypatch.setattr(
        QW.QMessageBox,
        "warning",
        lambda _parent, _title, text, *_args: warnings.append(text),
    )
    path = write_wheel(tmp_path)
    monkeypatch.setattr(widget, "select_file", lambda: path)
    monkeypatch.setattr(
        widget, "confirm_installation", lambda _manifest, origin=None: False
    )

    assert widget.install_from_file() is None

    invalid = tmp_path / "datalab_bad-name.py"
    invalid.write_text("", encoding="utf-8")
    monkeypatch.setattr(widget, "select_file", lambda: str(invalid))

    assert widget.install_from_file() is None
    assert len(warnings) == 1 and "datalab_<name>.py" in warnings[0]
    assert not store.get_plugins() and not reloads
    assert widget.placeholder.isVisibleTo(widget)

    dialog = PluginConsentDialog(store.inspect_wheel_file(path), widget)
    assert dialog.windowTitle() == "Install plugin"
    assert not dialog.install_button.isDefault()
    widget.deleteLater()


@pytest.mark.parametrize("accept", [True, False])
def test_additional_dependencies_are_planned_before_consent(
    store: InstalledPluginStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accept: bool,
) -> None:
    """The real asynchronous offline path installs nothing before confirmation."""
    _app = QW.QApplication.instance() or QW.QApplication([])
    dependency_name = "datalab-install-ui-dependency"
    folder = tmp_path / "dependencies"
    folder.mkdir()
    (folder / wheel_filename(dependency_name, "1.0.0")).write_bytes(
        make_plugin_wheel(
            distribution=dependency_name,
            version="1.0.0",
            package=f"{PACKAGE}_dependency",
            entry_points="",
            requires_dist=(),
            record=True,
        )
    )
    path = Path(write_wheel(tmp_path))
    path.write_bytes(
        make_plugin_wheel(
            distribution="datalab-install-ui-probe",
            version="1.0.0",
            package=PACKAGE,
            plugin_id=PLUGIN_ID,
            requires_dist=(f"{dependency_name}>=1",),
            record=True,
        )
    )
    reloads: list[bool] = []
    warnings: list[str] = []
    information: list[str] = []
    confirmed: list[dict] = []
    finished: list[bool] = []
    monkeypatch.setattr(
        QW.QMessageBox,
        "warning",
        lambda _parent, _title, text: warnings.append(text),
    )
    monkeypatch.setattr(
        QW.QMessageBox,
        "information",
        lambda _parent, _title, text: information.append(text),
    )
    widget = InstalledPluginsWidget(lambda: reloads.append(True))

    def consent(manifest: dict, origin: str | None = None) -> bool:
        assert not store.get_plugins()
        assert not store.get_dependency_paths()
        confirmed.append(manifest)
        dialog = PluginConsentDialog(manifest, widget, origin=origin)
        dialog.deleteLater()
        return accept

    monkeypatch.setattr(widget, "confirm_installation", consent)
    widget.install_path(str(path), on_finished=lambda: finished.append(True))
    qt_wait_until(lambda: bool(finished), timeout=90)
    assert finished == [True], warnings
    assert not warnings
    assert len(confirmed) == 1
    (artifact,) = confirmed[0]["additional_dependencies"]
    assert artifact["distribution"] == dependency_name
    assert not Path(artifact["path"]).exists()
    assert bool(store.get_plugins()) is accept
    assert bool(store.get_dependency_paths()) is accept
    assert bool(information) is accept
    assert not reloads
    assert not store.activate()
    widget.deleteLater()


@pytest.mark.parametrize(
    "error", [ValueError("invalid file"), RuntimeError("unexpected failure")]
)
def test_plugin_task_reports_every_error(error: Exception) -> None:
    """A failed operation always ends with the failure signal.

    An unexpected exception escaping the thread would emit no signal at all,
    leaving the plugin dialogs waiting forever.
    """
    _app = QW.QApplication.instance() or QW.QApplication([])
    succeeded: list[object] = []
    failed: list[str] = []

    def operation() -> None:
        raise error

    run_plugin_task(operation, succeeded.append, failed.append)
    qt_wait_until(lambda: bool(failed or succeeded), timeout=10)
    assert not succeeded
    assert len(failed) == 1 and str(error) in failed[0]


if __name__ == "__main__":
    pytest.main([__file__])

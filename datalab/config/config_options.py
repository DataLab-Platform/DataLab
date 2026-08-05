# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab configuration options container
---------------------------------------

Flat, SigimaX-style configuration container for DataLab
(:class:`DataLabOptions`), subclassing :class:`sigimax.config.SigimaXOptions`.

This module defines the in-memory option model only. On-disk persistence (the
INI<->JSON converter) and the wiring of the singleton ``Conf`` live in
:mod:`datalab.config`.

Design notes
~~~~~~~~~~~~

- Options common to any SigimaX-based application (color mode, window state,
  console, I/O, visualization defaults, processing behaviour) are inherited from
  :class:`~sigimax.config.SigimaXOptions`. DataLab only *adds* its own options
  and *replaces* a few inherited fields whose type differs
  (``traceback_log_path``, ``faulthandler_log_path`` become
  :class:`~datalab.utils.optionfields.ConfigPathOptionField`; ``base_dir``
  becomes :class:`~datalab.utils.optionfields.WorkingDirOptionField`).
- The former ``[macro]`` and ``[ai]`` INI sections are flattened with a
  ``macro_``/``ai_`` prefix to avoid generic-name collisions in the flat
  namespace (e.g. ``ai_enabled``, ``ai_provider``).
- :meth:`DataLabOptions.to_dict` / :meth:`DataLabOptions.from_dict` are
    overridden so that fields whose ``get``/``set`` transform the stored value
    (config paths, working directory, DataSet options) are (de)serialized using
    their *raw* value.
"""

from __future__ import annotations

from typing import Any

from guidata import configtools
from sigimax.config import OptionField, SigimaXOptions, TypedOptionField

from datalab import __docurl__, __homeurl__, __supporturl__, __version__
from datalab.config.optionfields import (
    ConfigPathOptionField,
    DataSetOptionField,
    FormatStringOptionField,
    WorkingDirOptionField,
)
from datalab.config.optionfields import FontOptionField as _FontOptionField

#: Application name used for default log file basenames.
APP_NAME = "DataLab"


class DataLabOptions(SigimaXOptions):
    """DataLab configuration options (flat, SigimaX-style container).

    Adds DataLab-specific options on top of :class:`sigimax.config.SigimaXOptions`
    and replaces a few inherited fields with DataLab-specific field types.
    """

    APP_NAME = APP_NAME
    CONF_VERSION = "1.0.0"

    def __init__(self) -> None:
        # INI write-through is disabled until the initial load has completed; it
        # is enabled by :mod:`datalab.config` after ``load_options_from_ini``.
        self._ini_persist_enabled = False
        super().__init__()

        # ===================================================================
        # Replace inherited fields whose type differs in DataLab
        # ===================================================================

        self.traceback_log_path = ConfigPathOptionField(
            self,
            "traceback_log_path",
            f".{APP_NAME}_traceback.log",
            description="Path to the traceback log file (config-dir basename).",
            category="main",
        )
        self.faulthandler_log_path = ConfigPathOptionField(
            self,
            "faulthandler_log_path",
            f".{APP_NAME}_faulthandler.log",
            description="Path to the faulthandler log file (config-dir basename).",
            category="main",
        )
        self.base_dir = WorkingDirOptionField(
            self,
            "base_dir",
            "",
            description="Base working directory for file dialogs.",
            category="main",
        )

        # ===================================================================
        # Main options — DataLab-specific
        # ===================================================================

        self.process_isolation_enabled = TypedOptionField(
            self,
            "process_isolation_enabled",
            category="main",
            default=True,
            expected_type=bool,
            description="If True, run each computation in a separate process.",
        )
        self.rpc_server_enabled = TypedOptionField(
            self,
            "rpc_server_enabled",
            category="main",
            default=True,
            expected_type=bool,
            description="If True, start the XML-RPC server for remote control.",
        )
        # rpc_server_port may be None (not yet assigned) or an int: no type check.
        self.rpc_server_port = OptionField(
            self,
            "rpc_server_port",
            category="main",
            default=None,
            description="XML-RPC server port (None until assigned at startup).",
        )
        self.webapi_localhost_no_token = TypedOptionField(
            self,
            "webapi_localhost_no_token",
            category="main",
            default=True,
            expected_type=bool,
            description="If True, allow localhost Web API connections without a token.",
        )
        # current_tab may be None or an int panel index: no type check.
        self.current_tab = OptionField(
            self,
            "current_tab",
            category="main",
            default=None,
            description="Index of the last active panel tab.",
        )
        self.plugins_enabled = TypedOptionField(
            self,
            "plugins_enabled",
            category="main",
            default=True,
            expected_type=bool,
            description="If True, enable the plugin system.",
        )
        # plugins_enabled_list: None = all enabled, [] = none, list = specific.
        self.plugins_enabled_list = OptionField(
            self,
            "plugins_enabled_list",
            category="main",
            default=None,
            description="Enabled plugin names (None = all, [] = none, or a list).",
        )
        self.plugins_path = TypedOptionField(
            self,
            "plugins_path",
            category="main",
            default="",
            expected_type=str,
            description="Deprecated single extra plugin directory (use "
            "plugins_path_list).",
        )
        self.plugins_path_list = TypedOptionField(
            self,
            "plugins_path_list",
            category="main",
            default=[],
            expected_type=list,
            description="List of extra plugin directories.",
        )
        self.tour_enabled = TypedOptionField(
            self,
            "tour_enabled",
            category="main",
            default=True,
            expected_type=bool,
            description="If True, offer the guided tour on first startup.",
        )
        self.v020_plugins_warning_ignore = TypedOptionField(
            self,
            "v020_plugins_warning_ignore",
            category="main",
            default=False,
            expected_type=bool,
            description="If True, do not warn about legacy v0.2.0 plugins.",
        )

        # ===================================================================
        # Processing options — DataLab-specific
        # ===================================================================

        self.small_mono_font = _FontOptionField(
            self,
            "small_mono_font",
            category="proc",
            default=(configtools.MONOSPACE, 8, False),
            description="Monospace font used by history and analysis tabs.",
        )

        # ===================================================================
        # View options — DataLab-specific
        # ===================================================================

        self.ignore_title_insertion_msg = TypedOptionField(
            self,
            "ignore_title_insertion_msg",
            category="view",
            default=False,
            expected_type=bool,
            description="Ignore the message shown when inserting a title as an "
            "annotation label.",
        )
        self.auto_refresh = TypedOptionField(
            self,
            "auto_refresh",
            category="view",
            default=True,
            expected_type=bool,
            description="If True, automatically refresh plots on changes.",
        )
        self.show_first_only = TypedOptionField(
            self,
            "show_first_only",
            category="view",
            default=False,
            expected_type=bool,
            description="If True, show only the first selected item.",
        )
        self.show_contrast = TypedOptionField(
            self,
            "show_contrast",
            category="view",
            default=True,
            expected_type=bool,
            description="If True, show the image contrast adjustment panel.",
        )
        self.sig_antialiasing = TypedOptionField(
            self,
            "sig_antialiasing",
            category="view",
            default=True,
            expected_type=bool,
            description="If True, enable anti-aliasing for signal curves.",
        )
        self.ima_aspect_ratio_1_1 = TypedOptionField(
            self,
            "ima_aspect_ratio_1_1",
            category="view",
            default=False,
            expected_type=bool,
            description="If True, lock image aspect ratio to 1:1.",
        )
        # Datetime format strings stored in clean form (%H...); the INI converter
        # handles ConfigParser percent-escaping at the persistence boundary.
        self.sig_datetime_format_s = FormatStringOptionField(
            self,
            "sig_datetime_format_s",
            category="view",
            default="%H:%M:%S",
            description="Datetime axis format for s, min, h units.",
        )
        self.sig_datetime_format_ms = FormatStringOptionField(
            self,
            "sig_datetime_format_ms",
            category="view",
            default="%H:%M:%S.%f",
            description="Datetime axis format for ms, us, ns units.",
        )
        self.max_shapes_to_draw = TypedOptionField(
            self,
            "max_shapes_to_draw",
            category="view",
            default=1000,
            expected_type=int,
            description="Maximum number of geometry shapes drawn on a plot.",
        )
        self.max_cells_in_label = TypedOptionField(
            self,
            "max_cells_in_label",
            category="view",
            default=100,
            expected_type=int,
            description="Maximum number of table cells in a merged result label.",
        )
        self.max_cols_in_label = TypedOptionField(
            self,
            "max_cols_in_label",
            category="view",
            default=15,
            expected_type=int,
            description="Maximum number of columns in a merged result label.",
        )
        self.show_result_label = TypedOptionField(
            self,
            "show_result_label",
            category="view",
            default=True,
            expected_type=bool,
            description="If True, show the merged result label on plots by default.",
        )
        self.show_marker_labels_in_table = TypedOptionField(
            self,
            "show_marker_labels_in_table",
            category="view",
            default=True,
            expected_type=bool,
            description="If True, prepend a marker-label column to result tables.",
        )

        # Annotated shape / marker visualization settings (DataSet options).
        # Default instances are assigned lazily from datalab.config once PlotPy
        # configuration is available (see initialize_default_plotpy_instances).
        self.sig_shape_param = DataSetOptionField(
            self,
            "sig_shape_param",
            category="view",
            description="Annotated shape visualization settings for signals.",
        )
        self.sig_marker_param = DataSetOptionField(
            self,
            "sig_marker_param",
            category="view",
            description="Marker visualization settings for signals.",
        )
        self.ima_shape_param = DataSetOptionField(
            self,
            "ima_shape_param",
            category="view",
            description="Annotated shape visualization settings for images.",
        )
        self.ima_marker_param = DataSetOptionField(
            self,
            "ima_marker_param",
            category="view",
            description="Marker visualization settings for images.",
        )

        # ===================================================================
        # I/O options — DataLab-specific (DataSet dialog settings)
        # ===================================================================

        self.save_to_directory_settings = DataSetOptionField(
            self,
            "save_to_directory_settings",
            category="io",
            description="Persisted settings for the 'save to directory' dialog.",
        )
        self.add_metadata_settings = DataSetOptionField(
            self,
            "add_metadata_settings",
            category="io",
            description="Persisted settings for the 'add metadata' dialog.",
        )

        # ===================================================================
        # Macro options — DataLab-specific (former [macro] section, prefixed)
        # ===================================================================

        self.macro_open_tab_uids = OptionField(
            self,
            "macro_open_tab_uids",
            category="macro",
            default=None,
            description="UUIDs of macro tabs open at last close (JSON list).",
        )
        self.macro_active_tab_uid = OptionField(
            self,
            "macro_active_tab_uid",
            category="macro",
            default=None,
            description="UUID of the active macro tab at last close.",
        )
        self.macro_splitter_state = OptionField(
            self,
            "macro_splitter_state",
            category="macro",
            default=None,
            description="Serialized editor/console splitter state.",
        )
        self.macro_console_max_lines = TypedOptionField(
            self,
            "macro_console_max_lines",
            category="macro",
            default=5000,
            expected_type=int,
            description="Maximum number of lines kept in the macro console.",
        )
        self.macro_close_tab_keeps_macro = TypedOptionField(
            self,
            "macro_close_tab_keeps_macro",
            category="macro",
            default=True,
            expected_type=bool,
            description="If True, closing a macro tab only hides the macro.",
        )
        self.macro_templates_path = TypedOptionField(
            self,
            "macro_templates_path",
            category="macro",
            default="",
            expected_type=str,
            description="Directory of user macro templates (resolved at startup).",
        )

        # ===================================================================
        # AI assistant options — DataLab-specific (former [ai] section, prefixed)
        # ===================================================================

        self.ai_enabled = TypedOptionField(
            self,
            "ai_enabled",
            category="ai",
            default=False,
            expected_type=bool,
            description="If True, enable the AI assistant.",
        )
        self.ai_provider = TypedOptionField(
            self,
            "ai_provider",
            category="ai",
            default="openai",
            expected_type=str,
            description="AI provider name (e.g. 'openai').",
        )
        self.ai_model = TypedOptionField(
            self,
            "ai_model",
            category="ai",
            default="gpt-4o-mini",
            expected_type=str,
            description="AI model name (e.g. 'gpt-4o-mini').",
        )
        self.ai_base_url = TypedOptionField(
            self,
            "ai_base_url",
            category="ai",
            default="",
            expected_type=str,
            description="Optional base URL override for OpenAI-compatible endpoints.",
        )
        self.ai_api_key = TypedOptionField(
            self,
            "ai_api_key",
            category="ai",
            default="",
            expected_type=str,
            description="AI API key (stored in plain text; never commit this).",
        )
        self.ai_temperature = TypedOptionField(
            self,
            "ai_temperature",
            category="ai",
            default=0.2,
            expected_type=float,
            description="AI sampling temperature (0.0-2.0).",
        )
        self.ai_timeout = TypedOptionField(
            self,
            "ai_timeout",
            category="ai",
            default=60.0,
            expected_type=float,
            description="AI HTTP timeout in seconds.",
        )
        self.ai_max_iterations = TypedOptionField(
            self,
            "ai_max_iterations",
            category="ai",
            default=8,
            expected_type=int,
            description="Maximum tool-call iterations per user prompt.",
        )
        self.ai_max_history_messages = TypedOptionField(
            self,
            "ai_max_history_messages",
            category="ai",
            default=0,
            expected_type=int,
            description="Maximum non-system messages sent per request (0 = unlimited).",
        )
        self.ai_auto_approve_readonly = TypedOptionField(
            self,
            "ai_auto_approve_readonly",
            category="ai",
            default=True,
            expected_type=bool,
            description="If True, auto-approve read-only inspection tools.",
        )
        self.ai_expose_macro_tool = TypedOptionField(
            self,
            "ai_expose_macro_tool",
            category="ai",
            default=True,
            expected_type=bool,
            description="If True, expose the create-and-run-macro tool to the LLM.",
        )

        # ===================================================================
        # Application branding — presentation metadata (see NON_PERSISTED)
        # ===================================================================
        # Set programmatically so that reused SigimaX widgets and windows display
        # DataLab's identity (name, version, icon, URLs, splash) when they read
        # the active configuration via ``sigimax.config.get_conf()``.
        self.app_name.set(APP_NAME)
        self.app_version.set(__version__)
        self.app_logo_path.set("DataLab.svg")
        self.app_docurl.set(__docurl__)
        self.app_homeurl.set(__homeurl__)
        self.app_supporturl.set(__supporturl__)
        self.splash_image_path.set("DataLab-Splash.png")
        self.splash_show_progress.set(False)

        # Recapture defaults now that DataLab-specific fields exist, using the
        # storage protocol so reset_to_defaults is correct for config-path,
        # working-dir and DataSet fields.
        self._defaults = self.to_dict()

    # -- Option categories (DataLab-specific extensions) --

    def get_option_categories(self) -> list[tuple[str, str]]:
        """Return DataLab option categories (inherited ones plus macro/ai).

        Returns:
            Ordered list of ``(category_id, label)`` pairs.
        """
        # TODO: wrap the macro/ai labels with the DataLab translation function
        # once the generic settings UI is wired.
        return super().get_option_categories() + [
            ("macro", "Macros"),
            ("ai", "AI assistant"),
        ]

    # -- Defaults introspection and INI write-through --

    def get_default_raw(self, field_name: str) -> Any:
        """Return the raw default value captured for an option field.

        Args:
            field_name: The option field name.

        Returns:
            The raw default value (as produced by :meth:`to_dict`), or ``None``
             when the field name is unknown.
        """
        return self._defaults.get(field_name)

    def set_ini_persist_enabled(self, enabled: bool) -> None:
        """Enable or disable INI write-through on option changes.

        When enabled, option changes flush the configuration to the INI file.
        This is disabled during the initial load and enabled afterwards by
        :mod:`datalab.config`.

        Args:
            enabled: Whether INI write-through is active.
        """
        self._ini_persist_enabled = enabled

    def is_ini_persist_enabled(self) -> bool:
        """Return whether option changes are persisted to the INI file."""
        return self._ini_persist_enabled

    def snapshot_option_context_state(self, name: str) -> bool:
        """Return whether an option was persisted before a temporary context."""
        if not self._ini_persist_enabled:
            return False
        # Imported lazily to avoid a config_options/config_persistence cycle.
        # pylint: disable=import-outside-toplevel
        from datalab.config.config_persistence import has_persisted_option

        return has_persisted_option(self, name)

    def restore_option_context_state(self, name: str, was_persisted: bool) -> None:
        """Restore persistence presence after a temporary option context."""
        if self._ini_persist_enabled and not was_persisted:
            # Imported lazily to avoid a config_options/config_persistence cycle.
            # pylint: disable=import-outside-toplevel
            from datalab.config.config_persistence import remove_persisted_option

            remove_persisted_option(self, name)

    def is_option_initialized(self, name: str) -> bool:
        """Return whether an option was loaded, set, or persisted in the INI."""
        if not self._ini_persist_enabled:
            return super().is_option_initialized(name)
        # Imported lazily to avoid a config_options/config_persistence cycle.
        # pylint: disable=import-outside-toplevel
        from datalab.config.config_persistence import has_persisted_option

        return has_persisted_option(self, name) or super().is_option_initialized(name)

    def option_changed(self, name: str) -> None:
        """Persist option changes when INI write-through is enabled."""
        del name
        if self._ini_persist_enabled:
            # Imported lazily to avoid a hard import cycle at module load time.
            # pylint: disable=import-outside-toplevel
            from datalab.config.config_persistence import save_options_to_ini

            save_options_to_ini(self)

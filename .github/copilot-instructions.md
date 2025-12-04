# DataLab AI Coding Agent Instructions

This document provides comprehensive guidance for AI coding agents working on the DataLab codebase. It covers architecture patterns, development workflows, and project-specific conventions.

## Project Overview

**DataLab** is an open-source platform for scientific and technical data processing with a Qt-based GUI. It processes **signals** (1D curves) and **images** (2D arrays).

### Key Architecture Components

1. **Sigima**: Separate computation library providing all processing algorithms (`sigima.proc`)
2. **DataLab GUI**: Qt application layer built on PlotPyStack (PlotPy + guidata)
3. **Processor Pattern**: Bridge between GUI and computation functions
4. **Action Handler**: Manages menus, toolbars, and GUI actions
5. **Plugin System**: Extensible architecture for third-party features
6. **Macro System**: User-scriptable automation via Python
7. **Remote Control**: XML-RPC API for external applications

### Technology Stack

- **Python**: 3.9+ (using `from __future__ import annotations`)
- **Core Libraries**: NumPy (≥1.26.4), SciPy (≥1.10.1), scikit-image, OpenCV
- **GUI**: Qt via PlotPy (≥2.8.2) and guidata (≥3.13.3)
- **Computation**: Sigima (≥1.0.2) - separate package
- **Testing**: pytest with coverage
- **Linting/Formatting**: Ruff (preferred), Pylint (with specific disables)
- **Internationalization**: gettext (.po files), sphinx-intl for docs
- **Documentation**: Sphinx with French translations
- **Packaging**: PyInstaller (standalone), WiX (MSI installer)

### Workspace Structure

```
DataLab/
├── datalab/              # Main application code
│   ├── gui/              # GUI layer
│   │   ├── processor/    # Processor pattern (signal.py, image.py, base.py)
│   │   ├── actionhandler.py  # Menu/action management
│   │   ├── main.py       # Main window
│   │   └── panel/        # Signal/Image panels
│   ├── control/          # Remote control API (proxy.py, remote.py)
│   ├── plugins/          # Plugin system
│   ├── tests/            # pytest test suite
│   ├── locale/           # Translations (.po files)
│   └── config.py         # Configuration management
├── doc/                  # Sphinx documentation
│   ├── locale/fr/        # French documentation translations
│   └── features/         # Feature documentation (signal/, image/)
├── macros/examples/      # Demo macros
├── scripts/              # Build/development scripts
│   └── run_with_env.py   # Environment loader (.env support)
├── .env                  # Local Python path (PYTHONPATH=.;../guidata;../plotpy;../sigima)
└── pyproject.toml        # Project configuration
```

**Related Projects** (sibling directories in multi-root workspace):
- `../Sigima/` - Computation library
- `../PlotPy/` - Plotting library
- `../guidata/` - GUI toolkit
- `../PythonQwt/` - Qwt bindings

## Development Workflows

### Running Commands

**ALWAYS use `scripts/run_with_env.py` for Python commands** to load environment from `.env`:

```powershell
# ✅ CORRECT - Loads PYTHONPATH from .env
python scripts/run_with_env.py python -m pytest

# ❌ WRONG - Misses local development packages
python -m pytest
```

### Testing

```powershell
# Run all tests
python scripts/run_with_env.py python -m pytest --ff

# Run specific test
python scripts/run_with_env.py python -m pytest datalab/tests/features/signal/

# Coverage
python scripts/run_with_env.py python -m coverage run -m pytest datalab
python -m coverage html
```

### Linting and Formatting

**Prefer Ruff** (fast, modern):

```powershell
# Format code
python scripts/run_with_env.py python -m ruff format

# Lint with auto-fix
python scripts/run_with_env.py python -m ruff check --fix
```

**Pylint** (with extensive disables for code structure):

```powershell
python scripts/run_with_env.py python -m pylint datalab \
    --disable=duplicate-code,fixme,too-many-arguments,too-many-branches, \
    too-many-instance-attributes,too-many-lines,too-many-locals, \
    too-many-public-methods,too-many-statements
```

### Translations

**UI Translations** (gettext):

```powershell
# Scan and update .po files
python scripts/run_with_env.py python -m guidata.utils.translations scan \
    --name datalab --directory . --copyright-holder "DataLab Platform Developers" \
    --languages fr

# Compile .mo files
python scripts/run_with_env.py python -m guidata.utils.translations compile \
    --name datalab --directory .
```

**Documentation Translations** (sphinx-intl):

```powershell
# Extract translatable strings
python scripts/run_with_env.py python -m sphinx build doc build/gettext -b gettext -W

# Update French .po files
python scripts/run_with_env.py python -m sphinx_intl update -d doc/locale -p build/gettext -l fr

# Build localized docs
python scripts/run_with_env.py python -m sphinx build doc build/doc -b html -D language=fr
```

## Core Patterns

### 1. Processor Pattern (GUI ↔ Computation Bridge)

**Location**: `datalab/gui/processor/`

**Key Concept**: Processors bridge GUI panels and Sigima computation functions. They define **generic processing types** based on input/output patterns.

#### Generic Processing Types

| Method | Pattern | Multi-selection | Use Cases |
|--------|---------|----------------|-----------|
| `compute_1_to_1` | 1 obj → 1 obj | k → k | Independent transformations (FFT, normalization) |
| `compute_1_to_0` | 1 obj → metadata | k → 0 | Analysis producing scalar results (FWHM, centroid) |
| `compute_1_to_n` | 1 obj → n objs | k → k·n | ROI extraction, splitting |
| `compute_n_to_1` | n objs → 1 obj | n → 1 (or n → n pairwise) | Averaging, summing, concatenation |
| `compute_2_to_1` | 1 obj + 1 operand → 1 obj | k + 1 → k (or n + n pairwise) | Binary operations (add, multiply) |

**Example: Implementing a New Processing Feature**

```python
# 1. Implement computation in Sigima (sigima/proc/signal/processing.py)
def my_processing_func(src: SignalObj, param: MyParam) -> SignalObj:
    """My processing function."""
    dst = src.copy()
    # ... computation logic ...
    return dst

# 2. Register in DataLab processor (datalab/gui/processor/signal.py)
def register_processing(self) -> None:
    self.register_1_to_1(
        sips.my_processing_func,
        _("My Processing"),
        paramclass=MyParam,
        icon_name="my_icon.svg",
    )
```

#### Registration Methods

```python
# In SignalProcessor or ImageProcessor class
def register_operations(self) -> None:
    """Register operations (basic math, transformations)."""

    # 1-to-1: Apply to each selected object independently
    self.register_1_to_1(
        sips.normalize,
        _("Normalize"),
        paramclass=sigima.params.NormalizeParam,
        icon_name="normalize.svg",
    )

    # 2-to-1: Binary operation with a second operand
    self.register_2_to_1(
        sips.difference,
        _("Difference"),
        icon_name="difference.svg",
        obj2_name=_("signal to subtract"),
        skip_xarray_compat=False,  # Enable X-array compatibility check
    )

    # n-to-1: Aggregate multiple objects
    self.register_n_to_1(
        sips.average,
        _("Average"),
        icon_name="average.svg",
    )
```

### 2. X-array Compatibility System

**Critical Pattern**: For **2-to-1** and **n-to-1** operations on signals, DataLab checks if X arrays match. If not, it **interpolates** the second signal to match the first.

**When to Skip**: Use `skip_xarray_compat=True` when operations **intentionally use mismatched X arrays** (e.g., replacing X with Y values from another signal).

```python
# ❌ BAD: Will trigger unwanted interpolation
self.register_2_to_1(
    sips.replace_x_by_other_y,
    _("Replace X by other signal's Y"),
)

# ✅ GOOD: Skips compatibility check
self.register_2_to_1(
    sips.replace_x_by_other_y,
    _("Replace X by other signal's Y"),
    skip_xarray_compat=True,  # Operation uses Y values as X coordinates
)
```

**Code Location**: `datalab/gui/processor/base.py` (lines ~1764, 1886)

### 3. Action Handler Pattern (Menu Management)

**Location**: `datalab/gui/actionhandler.py`

**Purpose**: Manages all GUI actions (menus, toolbars, context menus). Actions point to processors, panels, or direct operations.

**Key Classes**:
- `SignalActionHandler`: Signal-specific actions
- `ImageActionHandler`: Image-specific actions
- `SelectCond`: Conditions for enabling/disabling actions

**Example: Adding a Menu Action**

```python
# In SignalActionHandler or ImageActionHandler
def setup_processing_actions(self) -> None:
    """Setup processing menu actions."""

    # Reference registered processor operation
    act = self.action_for("my_processing_func")

    # Add to menu
    self.processing_menu.addAction(act)
```

**Menu Organization**:

Menus are organized by function:
- `File` → Import/export, project management
- `Edit` → Copy/paste, delete, metadata editing
- `Operation` → Basic math (add, multiply, etc.)
- `Processing` → Advanced transformations, filters
  - `Axis transformation` → Calibration, X-Y mode, replace X
- `Analysis` → Measurements, ROI extraction
- `Computing` → FFT, convolution, fit

The complete menu structure is defined in `datalab/gui/actionhandler.py`.
A text extract of the menu hiearchy is available in `scripts/datalab_menus.txt` (it is
generated with `scripts/print_datalab_menus.py`).

### 4. Plugin System

**Location**: `datalab/plugins.py`, `datalab/plugins/`

**Key Classes**:
- `PluginBase`: Base class for all plugins (uses metaclass `PluginRegistry`)
- `PluginInfo`: Plugin metadata (name, version, description, icon)

**Example: Creating a Plugin**

```python
from datalab.plugins import PluginBase, PluginInfo

class MyPlugin(PluginBase):
    """My custom plugin."""

    def __init__(self):
        super().__init__()
        self.info = PluginInfo(
            name="My Plugin",
            version="1.0.0",
            description="Does something useful",
            icon="my_icon.svg",
        )

    def register(self, mainwindow: DLMainWindow) -> None:
        """Register plugin with main window."""
        # Add menu items, actions, etc.
        pass

    def unregister(self) -> None:
        """Unregister plugin."""
        pass
```

**Plugin Discovery**: Plugins are loaded from:
1. `datalab/plugins/` (built-in)
2. User-defined paths in `Conf.get_path("plugins")`
3. For frozen apps, from `plugins/` directory next to executable

### 5. Macro System

**Location**: `macros/examples/`

**Purpose**: User-scriptable automation using Python. Macros use the **Remote Proxy** API to control DataLab.

**Example Macro**:

```python
from datalab.control.proxy import RemoteProxy
import numpy as np

proxy = RemoteProxy()

# Create signal
x = np.linspace(0, 10, 100)
y = np.sin(x)
proxy.add_signal("My Signal", x, y)

# Apply processing
proxy.calc("normalize")
proxy.calc("moving_average", sigima.params.MovingAverageParam.create(n=5))
```

**Key API Methods**:
- `proxy.add_signal()`, `proxy.add_image()`: Create objects
- `proxy.calc()`: Run processor methods
- `proxy.get_object()`: Retrieve data
- `proxy.call_method()`: Call any public panel or window method

**Generic Method Calling**:
```python
# Remove objects from current panel
proxy.call_method("remove_object", force=True)

# Call method on specific panel
proxy.call_method("delete_all_objects", panel="signal")

# Call main window method
panel_name = proxy.call_method("get_current_panel")
```

### 6. Remote Control API

**Location**: `datalab/control/`

**Classes**:
- `RemoteProxy`: XML-RPC client for remote DataLab control
- `LocalProxy`: Direct access for same-process scripting

**Calling Processor Methods**:

```python
# Without parameters
proxy.calc("average")

# With parameters
p = sigima.params.MovingAverageParam.create(n=30)
proxy.calc("moving_average", p)
```

## Coding Conventions

### Naming

- **Functions**: `snake_case` (e.g., `replace_x_by_other_y`)
- **Classes**: `PascalCase` (e.g., `SignalProcessor`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `PARAM_DEFAULTS`)
- **Private methods**: `_snake_case` (single underscore prefix)

### UI Text vs Function Names

- **Function name**: Technical, precise (e.g., `replace_x_by_other_y`)
- **UI text**: User-friendly, explicit (e.g., _("Replace X by other signal's Y"))
- **French translation**: Natural phrasing (e.g., "Remplacer X par le Y d'un autre signal")

### Type Annotations

**Always use** `from __future__ import annotations` for forward references:

```python
from __future__ import annotations

def process_signal(src: SignalObj) -> SignalObj:
    """Process signal."""
    pass
```

### Docstrings

Use **Google-style docstrings** with type hints:

```python
def compute_feature(obj: SignalObj, param: MyParam) -> SignalObj:
    """Compute feature on signal.

    Args:
        obj: Input signal object
        param: Processing parameters

    Returns:
        Processed signal object
    """
```

For continued lines in enumerations (args, returns), indent subsequent lines by 1 space:

```python
def compute_feature(obj: SignalObj, param: MyParam) -> SignalObj:
    """Compute feature on signal.

    Args:
        obj: Input signal object
        param: Processing parameters, with a very long description that
         continues on the next line.

    Returns:
     Processed signal object
    """
```

### Imports

**Order**: Standard library → Third-party → Local

```python
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
from guidata.qthelpers import create_action

from datalab.config import _
from datalab.gui.processor.base import BaseProcessor

if TYPE_CHECKING:
    from sigima.objects import SignalObj
```

### Internationalization

**Always wrap UI strings** with `_()`:

```python
from datalab.config import _

# ✅ CORRECT
menu_title = _("Processing")
action_text = _("Replace X by other signal's Y")

# ❌ WRONG
menu_title = "Processing"  # Not translatable!
```

## Common Tasks

### Adding a New Signal Processing Feature

**Complete workflow**:

1. **Implement computation in Sigima** (`sigima/proc/signal/processing.py`):
   ```python
   def my_feature(src: SignalObj, param: MyParam | None = None) -> SignalObj:
       """My feature."""
       dst = src.copy()
       # ... computation ...
       return dst
   ```

2. **Export from Sigima** (`sigima/proc/signal/__init__.py`):
   ```python
   from sigima.proc.signal.processing import my_feature  # Import
   __all__ = [..., "my_feature"]  # Export
   ```

3. **Register in DataLab processor** (`datalab/gui/processor/signal.py`):
   ```python
   def register_processing(self) -> None:
       self.register_1_to_1(
           sips.my_feature,
           _("My Feature"),
           paramclass=MyParam,
           icon_name="my_icon.svg",
       )
   ```

4. **Add menu action** (`datalab/gui/actionhandler.py`):
   ```python
   def setup_processing_actions(self) -> None:
       act = self.action_for("my_feature")
       self.processing_menu.addAction(act)
   ```

5. **Add tests** (`datalab/tests/features/signal/`):
   ```python
   def test_my_feature():
       obj = SignalObj.create(...)
       result = sips.my_feature(obj)
       assert result is not None
   ```

6. **Document** (`doc/features/signal/menu_processing.rst`):
   ````rst
   My Feature
   ^^^^^^^^^^

   Create a new signal by applying my feature:

   .. image:: /images/my_feature.png

   Parameters:

   - **Parameter 1**: Description
   ````

7. **Translate**:
   ```powershell
   # UI translation
   python scripts/run_with_env.py python -m guidata.utils.translations scan ...

   # Doc translation
   python scripts/run_with_env.py python -m sphinx_intl update -d doc/locale -p build/gettext -l fr
   ```

### Working with X-array Compatibility

**Rule of thumb**:
- **Most 2-to-1 operations**: Default behavior (auto-interpolation) is correct
- **X coordinate manipulation**: Set `skip_xarray_compat=True`

**Examples**:
- ✅ `difference` (subtract two signals): Compatible X arrays expected → `skip_xarray_compat=False`
- ✅ `xy_mode` (swap X and Y): Uses Y as new X → `skip_xarray_compat=True`
- ✅ `replace_x_by_other_y`: Takes Y from second signal as X → `skip_xarray_compat=True`

### Debugging Tips

1. **Check processor registration**: Look in `register_operations()` methods
2. **Verify action handler**: Search `actionhandler.py` for `action_for("function_name")`
3. **Test in isolation**: Use pytest with `--ff` flag (fail-fast)
4. **Check translations**: Missing `_()` wrapper causes English-only UI
5. **Verify imports**: Ensure function is in `__all__` export list

## Release Classification

**Bug Fix** (patch release: 1.0.x):
- Fixes incorrect behavior
- Restores expected functionality
- Addresses user-reported issues
- **Example**: Adding `replace_x_by_other_y` to handle wavelength calibration (was previously impossible)

**Feature** (minor release: 1.x.0):
- Adds entirely new capability
- Extends functionality beyond original scope
- Introduces new UI elements or workflows

## VS Code Tasks

The workspace includes predefined tasks (`.vscode/tasks.json`). Access via:
- `Ctrl+Shift+B` → "🧽🔦 Ruff" (format + lint)
- Terminal → "Run Task..." → "🚀 Pytest", "📚 Compile translations", etc.

**Key Tasks**:
- `🧽🔦 Ruff`: Format and lint code
- `🚀 Pytest`: Run tests with `--ff`
- `📚 Compile translations`: Build .mo files
- `🔎 Scan translations`: Update .po files
- `🌐 Build/open HTML doc`: Generate and open Sphinx docs

## Multi-Root Workspace

DataLab development uses a **multi-root workspace** (`.code-workspace` file) with sibling projects:

- `DataLab/` - Main GUI application
- `Sigima/` - Computation library
- `PlotPy/` - Plotting library
- `guidata/` - GUI toolkit
- `PythonQwt/` - Qwt bindings

**When working across projects**:
1. Changes in `Sigima` require importing in `DataLab`
2. Use `.env` file to point to local development versions
3. Test changes in both Sigima unit tests AND DataLab integration tests

## Release Notes Guidelines

**Location**: `doc/release_notes/release_MAJOR.MINOR.md` where MINOR is zero-padded to 2 digits (e.g., `release_1.00.md` for v1.0.x, `release_1.01.md` for v1.1.x)

**Writing Style**: Focus on **user impact**, not implementation details.

**Good release note** (user-focused):
- ✅ "Fixed syntax errors when using f-strings with nested quotes in macros"
- ✅ "Fixed corrupted Unicode characters in macro console output on Windows"
- ✅ "Fixed 'Lock LUT range' setting not persisting after closing Settings dialog"

**Bad release note** (implementation-focused):
- ❌ "Removed `code.replace('"', "'")` that broke f-strings"
- ❌ "Changed QTextCodec.codecForLocale() to codecForName(b'UTF-8')"
- ❌ "Added missing `ima_def_keep_lut_range` option in configuration"

**Structure**:
- **What went wrong**: Describe the symptom users experienced
- **When it occurred**: Specify the context/scenario
- **What's fixed**: Explain the benefit, not the implementation

**Example**:
```markdown
**Macro execution:**

* Fixed syntax errors when using f-strings with nested quotes in macros (e.g., `f'text {func("arg")}'` now works correctly)
* Fixed corrupted Unicode characters in macro console output on Windows - special characters like ✅, 💡, and → now display correctly instead of showing garbled text
```

## Key Files Reference

| File | Purpose |
|------|---------|
| `datalab/gui/processor/signal.py` | Signal processor registration |
| `datalab/gui/processor/image.py` | Image processor registration |
| `datalab/gui/processor/base.py` | Base processor class (generic methods) |
| `datalab/gui/actionhandler.py` | Menu and action management |
| `datalab/config.py` | Configuration, `_()` translation function |
| `datalab/plugins.py` | Plugin system implementation |
| `datalab/control/proxy.py` | Remote control API (RemoteProxy, LocalProxy) |
| `sigima/proc/signal/processing.py` | Signal computation functions |
| `sigima/proc/image/processing.py` | Image computation functions |
| `scripts/run_with_env.py` | Environment loader (loads `.env`) |
| `.env` | Local PYTHONPATH for development |
| `doc/release_notes/release_MAJOR.MINOR.md` | Release notes (MINOR is zero-padded: release_1.00.md for v1.0.x, release_1.01.md for v1.1.x) |

## Getting Help

- **Documentation**: https://datalab-platform.com/
- **Issues**: https://github.com/DataLab-Platform/DataLab/issues
- **Contributing**: https://datalab-platform.com/en/contributing/index.html
- **Support**: p.raybaut@codra.fr

---

**Remember**: Always use `scripts/run_with_env.py` for Python commands, wrap UI strings with `_()`, and consider X-array compatibility when adding 2-to-1 signal operations.

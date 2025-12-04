# Version 1.1 #

## DataLab Version 1.1.0 ##

### ✨ New Features ###

**Remote control API enhancements:**

* Added `call_method()` method to `RemoteProxy` class for calling any public method on DataLab main window or panels:
  * Enables programmatic access to operations not exposed through dedicated proxy methods (e.g., `remove_object`, `delete_all_objects`, `get_current_panel`)
  * Supports automatic method resolution: checks main window first, then current panel if method not found
  * Optional `panel` parameter allows targeting specific panel ("signal" or "image")
  * Thread-safe execution: GUI operations automatically execute in main thread to prevent freezing
  * New macro example `test_call_method.py` demonstrates usage: removing objects, getting current panel, and panel-specific operations
  * Expands automation capabilities for advanced macro workflows and external application control

* Added `remove_object()` method to proxy interface (local and remote) for selective object deletion
  * Removes currently selected object from active panel
  * Optional `force` parameter to skip confirmation dialog
  * Complements existing `reset_all()` method which clears the entire workspace (including all macros)

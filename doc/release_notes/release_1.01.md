# Version 1.1 #

## DataLab Version 1.1.0 (unreleased) ##

### ✨ New Features ###

**Proxy API:**

* Added `remove_object()` method to proxy interface (local and remote) for selective object deletion
  * Removes currently selected object from active panel
  * Optional `force` parameter to skip confirmation dialog
  * Complements existing `reset_all()` method which removes all objects

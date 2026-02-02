# Test call_method macro example

import numpy as np

from datalab.control.proxy import RemoteProxy

proxy = RemoteProxy()

# Add some test signals
x = np.linspace(0, 10, 100)
for i in range(3):
    y = np.sin(x + i)
    proxy.add_signal(f"Signal {i + 1}", x, y)

print(f"Added signals: {proxy.get_object_titles('signal')}")

# Remove first object using call_method (no panel parameter needed!)
# It will auto-detect and use the current panel
proxy.set_current_panel("signal")
proxy.select_objects([1])
proxy.call_method("remove_object", force=True)

print(f"After removing first: {proxy.get_object_titles('signal')}")

# Test method resolution: main window method is found first
current_panel = proxy.call_method("get_current_panel")
print(f"Current panel (from main window method): {current_panel}")

# Remove all objects from signal panel
# Can specify panel explicitly if needed
proxy.call_method("remove_all_objects", panel="signal")

print(f"After removing all: {proxy.get_object_titles('signal')}")

print("✅ Test completed successfully!")
print("💡 Note: call_method() tries main window first, then current panel")

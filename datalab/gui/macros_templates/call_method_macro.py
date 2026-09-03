# DataLab template: Call panel methods

from datalab.control.proxy import RemoteProxy

proxy = RemoteProxy()

# List currently visible signals and images via generic method calls.
sigs = proxy.call_method("get_object_titles", panel="signal")
imgs = proxy.call_method("get_object_titles", panel="image")
print(f"Signals: {len(sigs)} | Images: {len(imgs)}")

# Switch panels through the proxy.
proxy.set_current_panel("signal")
print(f"Current panel: {proxy.get_current_panel()}")

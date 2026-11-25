# Version 1.5 #

## DataLab Version 1.5.0 ##

### Requirements ###

* **Third-party**:
  * The new `mcp` optional dependency (MCP SDK ≥ 1.10, < 2) enables the MCP server for external assistants: install it with `pip install datalab-platform[mcp]`. It is included in the standalone version.

### ✨ New Features ###

* **External assistants (MCP)**: an external AI assistant, e.g. GitHub Copilot in VS Code, can now work on the open DataLab session through the Model Context Protocol. **File > Connect an external assistant (MCP)...**, also available from the AI Assistant panel, starts a server listening on this computer only, protected by a token, and gives the VS Code configuration to use. The assistant can list and inspect signals and images, create synthetic data, run processing and analysis features on the objects it chooses, and read their results with their units; you see the results in DataLab and confirm every action modifying the workspace, unless you approve the calls automatically. Running macros, loading files and triggering plugin actions are not available to external assistants (closes [Issue #584](https://github.com/DataLab-Platform/DataLab/issues/584)).
* **AI Assistant**: the assistant can now read the analysis results stored in objects (e.g. FWHM or centroid) with their units, apply an operation to given objects instead of the selected ones, and knows the choices, bounds, units and default values of the processing parameters. Invalid parameter values are reported with the valid ones instead of failing later (closes [Issue #584](https://github.com/DataLab-Platform/DataLab/issues/584)).

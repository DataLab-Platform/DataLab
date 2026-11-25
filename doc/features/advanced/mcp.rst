.. _mcp_server:

External assistants (MCP)
=========================

.. meta::
    :description: Connect an external AI assistant (VS Code Copilot, Claude Desktop...) to DataLab, the open-source scientific data analysis and visualization platform, with the Model Context Protocol (MCP)
    :keywords: DataLab, MCP, Model Context Protocol, AI, assistant, VS Code, Copilot, agent

DataLab includes a **Model Context Protocol (MCP)** server: an external
assistant, e.g. GitHub Copilot in VS Code, can then use the tools of the
:ref:`ai_assistant` on the open DataLab session to list the signals and images,
inspect them, create synthetic data, run processing and analysis features and
read their results.

The assistant acts on the objects shown in DataLab: you see the results appear,
and you confirm every action modifying the workspace, unless you approve the
calls automatically (see :ref:`mcp_security`).

.. note::

    The MCP server needs the optional ``mcp`` package. It is included in the
    standalone version of DataLab; with pip, install it with
    ``pip install datalab-platform[mcp]``. Without it, the
    **File > Connect an external assistant (MCP)...** action is disabled and
    its tooltip explains how to install it.

Starting the server
-------------------

1. Select **File > Connect an external assistant (MCP)...**, or click **MCP
   server...** in the toolbar of the AI Assistant panel.
2. Click **Start server**. The dialog shows the URL of the server (by default
   ``http://127.0.0.1:18090/mcp``, or the next free port) and its token.
3. Copy the VS Code configuration given by the dialog (see below).

The server stops when you click **Stop server** or when DataLab closes. It
listens on this computer only (``127.0.0.1``); the preferred port can be
changed in the **General** tab of the settings, and the ``DATALAB_MCP_PORT``
environment variable overrides it.

Connecting VS Code
------------------

Save the configuration given by the dialog as ``.vscode/mcp.json`` in your
workspace (or add it to your user MCP configuration), for example:

.. code-block:: json

    {
      "servers": {
        "datalab": {
          "type": "http",
          "url": "http://127.0.0.1:18090/mcp",
          "headers": {
            "Authorization": "Bearer ${input:datalab-mcp-token}"
          }
        }
      },
      "inputs": [
        {
          "type": "promptString",
          "id": "datalab-mcp-token",
          "description": "DataLab MCP token",
          "password": true
        }
      ]
    }

The token is not written in this file: VS Code asks for it when it starts the
server, and stores it securely. Start the ``datalab`` server from the
``mcp.json`` file or from the **MCP: List Servers** command, paste the token
copied from DataLab, then ask Copilot in agent mode, for example: *"List the
signals of DataLab, compute the FFT of the first one and give me the frequency
of its main peak"*.

VS Code keeps the token entered. When the token changes (new start of the server
without the persistent token, or **Renew**), DataLab refuses the former token and
VS Code cannot connect anymore: in ``mcp.json``, hover over
``${input:datalab-mcp-token}``, choose **Edit** and paste the new token, then
restart the ``datalab`` server.

Other assistants supporting the *Streamable HTTP* transport of MCP with an
``Authorization`` header can connect the same way.

Available tools
---------------

The server publishes the tools of the AI Assistant that act on data:
``list_objects``, ``get_current_panel``, ``get_object_info``,
``list_available_operations``, ``get_api_help``, ``capture_view``,
``create_synthetic_signal``, ``create_synthetic_image``, ``apply_operation`` and
``get_results``.

- ``list_available_operations`` describes the parameters of each processing
  feature: types, choices, bounds, units and default values.
- ``apply_operation`` runs a processing or analysis feature on the objects
  given by their UUID (``target_uuids``, and ``operand_uuid`` for the
  operations with a second operand), or on the selected objects. It returns the
  new objects, the analysis results with their units, and the errors and
  warnings of the computation.
- ``get_results`` returns the analysis results stored in objects.

For safety, the tools running Python code (macros), reading files or
triggering plugin actions are not published.

.. _mcp_security:

Security
--------

- **Local only**: the server listens on ``127.0.0.1`` and refuses requests
  addressed to another host or coming from a web page of another origin.
- **Token**: every request needs the token. By default, a new token is created
  at each start of the server; with the **Persistent MCP token** option of the
  settings, the token is kept between sessions, in a file of the DataLab
  configuration directory readable by you only. **Renew** replaces the token:
  the assistants using the former one are refused.
- **Confirmation**: every action modifying the workspace shows a confirmation
  dialog naming the external assistant, as in the AI Assistant panel. The
  read-only tools do not need confirmation. To let the external assistant work
  without confirmation, check **Approve the calls automatically** in the MCP
  dialog (or the **Approve external assistant calls automatically** option of
  the settings): the actions modifying the workspace then run directly, and
  remain visible in DataLab.
- **One call at a time**: calls are run one after the other on the DataLab
  session; a call which has not started yet is cancelled when the server stops.

.. warning::

    The external assistant receives the information returned by the tools, e.g.
    the titles, metadata and results of your objects or a capture of the view,
    and may send it to the language model it uses. Use a model and a provider
    suited to the confidentiality of your data.

.. seealso::

    The :ref:`ai_assistant` describes the tools, and the :ref:`ref-to-webapi`
    gives programmatic access to DataLab from notebooks.

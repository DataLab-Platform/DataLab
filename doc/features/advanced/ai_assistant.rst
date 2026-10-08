.. _ai_assistant:

AI Assistant
============

.. meta::
    :description: The built-in AI Assistant of DataLab, the open-source scientific data analysis and visualization platform
    :keywords: DataLab, AI, assistant, LLM, OpenAI, Ollama, chat, macro, automation

DataLab embeds an optional **AI Assistant**: a dockable chat panel connected to a
Large Language Model (LLM) that can inspect your workspace, create and process
signals and images, and write and run Python macros on your behalf.

.. warning::

    The AI Assistant sends your messages — and, when you ask for it, information
    about the objects of your workspace — to the configured provider. Nothing is
    sent as long as the assistant is disabled or as long as you do not start a
    conversation. Review the :ref:`ai-assistant-privacy` section before using a
    remote provider on sensitive data.

Getting started
---------------

1. Open ``File > Settings``, go to the **AI Assistant** tab and enable the
   assistant (see :ref:`ai-assistant-settings` for the complete list of options).
2. Choose a provider and a model, then set the API key — or, better, leave the
   field empty and export the provider's standard environment variable.
3. Click **Test connection** to check the endpoint before starting.
4. Show the **AI Assistant** panel from the ``View`` menu if it is hidden. It is
   docked on the right side of the main window, tabified with the Macro panel.

.. tip::

    To discover the assistant without any account or API key, select the
    ``mock`` provider: it replies with scripted answers triggered by simple
    keywords, and exercises the whole pipeline (tool calls, confirmation
    dialogs, conversation storage) offline.

Supported providers
-------------------

The assistant talks to any service exposing an **OpenAI-compatible** chat
completion API. This covers, among others:

- **OpenAI** (``https://api.openai.com/v1``)
- **GitHub Models** (``https://models.github.ai/inference``)
- **Azure OpenAI**
- Local runtimes such as **Ollama**, **LM Studio**, **llama.cpp** or **vLLM**

The **Load preset...** button of the settings dialog pre-fills the *Base URL*
and *Model* fields for the most common endpoints.

Using the chat panel
--------------------

The panel is made of a toolbar, the conversation view and an input box:

- **New conversation**: start a fresh conversation. The previous one is kept in
  the conversation store.
- **History...**: browse past conversations, load, rename or delete them, and
  **export** one as a Markdown file. Conversations are stored in the DataLab
  user configuration directory.
- **Send** / **Stop**: send the current message, or interrupt an ongoing
  request. Requests run in a background thread, so the interface stays
  responsive.
- The token counter on the right of the toolbar shows the context size of the
  last request and the cumulated usage of the conversation.

The input box keeps a history of previously sent messages, navigable with the
keyboard like a shell prompt.

When the assistant proposes a macro, the generated code is **transient**: it is
executed without cluttering the Macro panel. A **Save to Macros** link is offered
after the run if you want to keep it.

Tools available to the assistant
--------------------------------

The assistant does not act directly on DataLab: it may only call a fixed set of
declared **tools**. Tools marked as read-only never modify the workspace and may
be auto-approved (see the corresponding setting); every other tool requires an
explicit confirmation.

.. list-table::
    :header-rows: 1
    :widths: 30 12 58

    * - Tool
      - Read-only
      - Purpose
    * - ``list_objects``
      - ✓
      - List the signals or images of a panel
    * - ``get_current_panel``
      - ✓
      - Return the currently active panel
    * - ``get_object_info``
      - ✓
      - Inspect a specific object (shape, units, metadata, ...)
    * - ``list_available_operations``
      - ✓
      - Introspect the processing catalog exposed by Sigima
    * - ``list_plugin_actions``
      - ✓
      - List the actions contributed by third-party plugins
    * - ``get_macro_console_output``
      - ✓
      - Read back the Macro panel console
    * - ``get_api_help``
      - ✓
      - Return the public API of the proxy, ``SignalObj``, ``ImageObj`` or
        ``sigima.params``, so the model does not invent method names
    * - ``capture_view``
      - ✓
      - Grab a screenshot of the current plot and inject it in the conversation,
        so a multimodal model can visually inspect the data
    * - ``trigger_plugin_action``
      - ..
      - Trigger a plugin action by its menu path
    * - ``create_synthetic_signal``
      - ..
      - Create a synthetic signal (sine, cosine, Gaussian, noise, ramp)
    * - ``create_synthetic_image``
      - ..
      - Create a synthetic image (2D Gaussian, ramp, noise, checkerboard)
    * - ``load_file``
      - ..
      - Load a file into a panel
    * - ``apply_operation``
      - ..
      - Run any registered processing feature
    * - ``create_and_run_macro``
      - ..
      - Create and execute a Python macro

.. note::

    The ``create_and_run_macro`` tool is only exposed to the model when the
    **Allow AI to create and run macros** option is enabled. When it is
    disabled, the assistant cannot even propose arbitrary code execution.

Confirming tool calls
---------------------

Whenever the assistant wants to run a tool that modifies the workspace, a
confirmation dialog shows the tool name and the parameter values. When a macro
is proposed, the dialog also displays its **syntax-highlighted source code**, so
that the code can be reviewed before being executed. Rejecting the call returns
the refusal to the model, which may then propose something else.

.. _ai-assistant-privacy:

Privacy and security
--------------------

- **No telemetry**: DataLab does not log or forward the content of your
  conversations anywhere else than to the provider you configured.
- **Local providers**: to keep everything on your machine, point the *Base URL*
  to a local runtime (Ollama, LM Studio, llama.cpp, vLLM). Data then never
  leaves your computer.
- **API key storage**: the key is stored in plain text in the DataLab
  configuration file. Prefer the provider's environment variable.
- **Explicit confirmation**: every action modifying the workspace — and every
  macro execution — requires an explicit user confirmation by default.
- **Iteration cap**: the number of chained tool calls is bounded by the
  **Max tool-call iterations** setting, so a misbehaving model cannot loop
  indefinitely.

.. seealso::

    The macro system used by the assistant is described in :ref:`about_macros`,
    and the underlying control API in :ref:`ref-to-remote-control`.

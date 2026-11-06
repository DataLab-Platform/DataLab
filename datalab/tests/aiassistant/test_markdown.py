# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for the AI assistant Markdown renderer (code block footers)."""

from __future__ import annotations

from datalab.aiassistant.widgets.markdown import (
    is_python_code_block,
    markdown_to_html,
)


def test_is_python_code_block() -> None:
    """Python fences are recognized by their language or their content."""
    assert is_python_code_block("python", "x = 1")
    assert is_python_code_block("py", "x = 1")
    assert is_python_code_block("Python3", "x = 1")
    assert is_python_code_block("", "import numpy as np")
    assert is_python_code_block("", "from sigima import params")
    assert is_python_code_block("", "proxy.calc('fft')")
    assert not is_python_code_block("", "just some text")
    assert not is_python_code_block("", "")
    assert not is_python_code_block("bash", "import numpy")
    assert not is_python_code_block("json", '{"proxy.x": 1}')


def test_code_block_footer_receives_raw_code() -> None:
    """The footer callback gets the language and the unescaped code, and its
    HTML is inserted right after the block."""
    calls: list[tuple[str, str]] = []

    def footer(lang: str, code: str) -> str:
        calls.append((lang, code))
        return f"<span class='footer'>{len(calls)}</span>"

    text = (
        "Intro\n\n```python\nif a < b:\n    print('x')\n```\n\n```bash\necho hi\n```\n"
    )
    out = markdown_to_html(text, code_block_footer=footer)
    assert calls == [("python", "if a < b:\n    print('x')"), ("bash", "echo hi")]
    # Code stays escaped in the rendered block; footers follow their block.
    assert "if a &lt; b:" in out
    assert out.index("</pre><span class='footer'>1</span>") < out.index("echo hi")
    assert out.endswith("</pre><span class='footer'>2</span>")


def test_no_footer_by_default() -> None:
    """Without callback, the rendering is unchanged."""
    text = "```python\nprint(1)\n```"
    assert markdown_to_html(text) == markdown_to_html(
        text, code_block_footer=lambda _lang, _code: ""
    )
    assert markdown_to_html(text).endswith("</code></pre>")

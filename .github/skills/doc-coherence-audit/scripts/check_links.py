"""Check the documentation links of the DataLab-Platform repositories.

Two read-only checks:

- URLs (http/https) of the platform domains found in the documentation,
  Markdown and Python files tracked by git, fetched with a GET request
  (redirections followed).
- Relative links of Markdown files (``[text](path)``), resolved from the file.

Usage (standard library only)::

    python check_links.py C:\\Dev\\Projets\\DataLab C:\\Dev\\Projets\\Sigima
    python check_links.py <repo> ... --all-domains --workers 16

Prints a Markdown table of the broken links and exits with code 1 if any.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "build",
    "dist",
    "_build",
    "locale",
    "__pycache__",
    ".pytest_cache",
    "test-results",
    "playwright-report",
}
SUFFIXES = {".md", ".rst", ".txt", ".py", ".toml", ".cff", ".yml", ".yaml"}
PLATFORM_DOMAINS = (
    "datalab-platform.com",
    "github.com/DataLab-Platform",
    "github.com/PlotPyStack",
    "readthedocs.io",
    "pypi.org/project",
    "anaconda.org",
)
URL_RE = re.compile(r"https?://[^\s<>\"'`)\]}|\\]+")
MD_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
USER_AGENT = "Mozilla/5.0 (DataLab documentation audit)"


def iter_files(repo: Path):
    """Yield the documentation-like files tracked by git in a repository."""
    try:
        output = subprocess.run(
            ["git", "-C", str(repo), "ls-files", "-z"],
            capture_output=True,
            check=True,
        ).stdout.decode("utf-8")
        paths = [repo / name for name in output.split("\0") if name]
    except (OSError, subprocess.CalledProcessError):
        paths = list(repo.rglob("*"))
    for path in paths:
        if any(part in EXCLUDED_DIRS for part in path.relative_to(repo).parts):
            continue
        if path.is_file() and path.suffix.lower() in SUFFIXES:
            yield path


def clean_url(url: str) -> str:
    """Strip the trailing punctuation captured by the URL regular expression."""
    if url.endswith("`_"):
        url = url[:-2]
    return url.rstrip(".,;:!?*_>")


def is_placeholder(url: str) -> bool:
    """Return True for URLs that cannot be checked by a GET request.

    Template, local or example URLs, pip VCS requirements (``repo.git@branch``),
    prefixes completed by code (``.../blob/``) and pages requiring a GitHub login.
    """
    if any(token in url for token in ("{", "<", "localhost", "127.0.0.1")):
        return True
    if "example." in url or ".git@" in url or "/security/" in url:
        return True
    return url.endswith(("/blob/", "/tree/", "/archive/", "/raw/", "/releases/tag/"))


def collect(repos: list[Path], all_domains: bool):
    """Return the URLs (url -> locations) and the broken relative links."""
    urls: dict[str, list[str]] = {}
    broken_relative: list[tuple[str, str, str]] = []
    for repo in repos:
        for path in iter_files(repo):
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                location = f"{path}:{lineno}"
                for match in URL_RE.finditer(line):
                    url = clean_url(match.group(0))
                    if is_placeholder(url):
                        continue
                    if all_domains or any(d in url for d in PLATFORM_DOMAINS):
                        urls.setdefault(url, []).append(location)
                if path.suffix.lower() != ".md":
                    continue
                for match in MD_LINK_RE.finditer(line):
                    target = match.group(1)
                    if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I):
                        continue
                    rel = target.split("#", 1)[0].split("?", 1)[0]
                    if not rel or rel.startswith("<"):
                        continue
                    base = repo if rel.startswith("/") else path.parent
                    if not (base / rel.lstrip("/")).exists():
                        broken_relative.append((location, target, "missing file"))
    return urls, broken_relative


def fetch(url: str, retries: int = 2) -> str | None:
    """Return None if the URL is reachable, else an error description."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                response.read(1024)
            return None
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 502, 503, 504) and attempt < retries:
                time.sleep(2 * (attempt + 1))
                continue
            return f"HTTP {exc.code}"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
                continue
            return f"{type(exc).__name__}: {getattr(exc, 'reason', exc)}"
    return "unknown error"


def main() -> int:
    """Run the link checks and print the report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("repos", nargs="+", type=Path, help="repository folders")
    parser.add_argument(
        "--all-domains", action="store_true", help="check every external URL"
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--no-network", action="store_true", help="only check relative links"
    )
    args = parser.parse_args()
    repos = [repo.resolve() for repo in args.repos]
    urls, broken_relative = collect(repos, args.all_domains)
    broken_urls: list[tuple[str, str, str]] = []
    if not args.no_network:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            results = dict(zip(urls, executor.map(fetch, urls)))
        for url, error in sorted(results.items()):
            for location in urls[url] if error else []:
                broken_urls.append((location, url, error))
    print(f"Checked {len(urls)} distinct URLs in {len(repos)} repositories.")
    print(
        f"Broken URLs: {len(broken_urls)}; "
        f"broken relative links: {len(broken_relative)}\n"
    )
    if broken_urls or broken_relative:
        print("| Location | Link | Error |\n| --- | --- | --- |")
        for location, link, error in broken_urls + broken_relative:
            print(f"| `{location}` | `{link}` | {error} |")
    return 1 if broken_urls or broken_relative else 0


if __name__ == "__main__":
    sys.exit(main())

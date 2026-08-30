#!/usr/bin/env python3
"""Inject a version switcher banner into pdoc HTML pages for GitHub Pages."""

from __future__ import annotations

import os
import sys
from pathlib import Path

MARKER = "hylite-doc-version-bar"


def _pages_hrefs() -> tuple[str, str]:
    base = os.environ.get("PAGES_BASE", "/hylite").rstrip("/")
    return f"{base}/", f"{base}/dev/"


def _inactive_link(href: str, label: str) -> str:
    return (
        f'<a href="{href}" style="color:#7eb8ff !important;text-decoration:underline !important;'
        f'font-weight:400 !important;">{label}</a>'
    )


def _active_label(label: str) -> str:
    return (
        f'<span style="color:#fff !important;font-weight:600 !important;'
        f'text-decoration:none !important;">{label}</span>'
    )


def banner_html(active: str) -> str:
    stable_href, dev_href = _pages_hrefs()
    stable_active = active == "stable"
    if stable_active:
        stable_el = _active_label("Stable (master)")
        dev_el = _inactive_link(dev_href, "Development (dev)")
    else:
        stable_el = _inactive_link(stable_href, "Stable (master)")
        dev_el = _active_label("Development (dev)")
    return (
        f'<div id="{MARKER}" style="position:sticky;top:0;z-index:9999;'
        "background:#1a1a2e;color:#eee;padding:8px 16px;"
        "font-family:system-ui,sans-serif;font-size:14px;"
        'border-bottom:1px solid #444;display:flex;align-items:center;gap:12px;flex-wrap:wrap;">'
        '<span style="opacity:0.85;">hylite documentation</span>'
        '<span style="opacity:0.5;">|</span>'
        f"{stable_el}{dev_el}"
        "</div>"
    )


def _html_files(root: Path, active: str):
    """Yield HTML files to inject; skip ``dev/`` when tagging stable at combined site root."""
    for html in sorted(root.rglob("*.html")):
        if active == "stable":
            try:
                if html.relative_to(root).parts[:1] == ("dev",):
                    continue
            except ValueError:
                pass
        yield html


def inject_file(path: Path, banner: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        start = text.find(f'<div id="{MARKER}"')
        if start == -1:
            return False
        end = text.find("</div>", start)
        if end == -1:
            return False
        text = text[:start] + banner + text[end + len("</div>") :]
        path.write_text(text, encoding="utf-8")
        return True
    for needle in ("<body>", "<body ", "<BODY>"):
        idx = text.find(needle)
        if idx == -1:
            continue
        close = text.find(">", idx)
        if close == -1:
            continue
        path.write_text(text[: close + 1] + banner + text[close + 1 :], encoding="utf-8")
        return True
    print(f"warning: no <body> tag in {path}", file=sys.stderr)
    return False


def main() -> None:
    if len(sys.argv) < 3:
        print("usage: inject_docs_banner.py <html-root> <stable|dev>", file=sys.stderr)
        sys.exit(2)
    root = Path(sys.argv[1])
    active = sys.argv[2]
    if active not in ("stable", "dev"):
        print("active must be 'stable' or 'dev'", file=sys.stderr)
        sys.exit(2)
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    banner = banner_html(active)
    n = 0
    for html in _html_files(root, active):
        if inject_file(html, banner):
            n += 1
    print(f"injected banner into {n} file(s) under {root} ({active})")


if __name__ == "__main__":
    main()

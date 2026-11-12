"""Build the Medium HTML body for docs/blog/posts/gthulhu-vllm-cpu-pressure.md.

Medium's URL importer flattens Markdown tables, drops fenced code blocks and
truncates the tail, so the body is rebuilt from the canonical Markdown and
pasted into the editor. Tables become readable list fallbacks (Medium has no
table block type); fenced code becomes <pre>.
"""
import html
import pathlib
import re
import sys

SRC = pathlib.Path("docs/blog/posts/gthulhu-vllm-cpu-pressure.md")
OUT = pathlib.Path("draft/media/2026-11-01/gthulhu-vllm-cpu-pressure/medium-body.html")


def esc(s: str) -> str:
    return html.escape(s, quote=False)


TOKEN = re.compile(r"(\[[^\]]+\]\([^)]+\)|`[^`]+`|\*\*[^*]+\*\*)")


def inline(s: str) -> str:
    out = []
    for part in TOKEN.split(s):
        if not part:
            continue
        if m := re.fullmatch(r"\[([^\]]+)\]\(([^)]+)\)", part):
            label = "".join(
                f"<code>{esc(c)}</code>" if j % 2 else esc(c)
                for j, c in enumerate(m.group(1).split("`"))
            )
            out.append(f'<a href="{m.group(2)}">{label}</a>')
        elif m := re.fullmatch(r"`([^`]+)`", part):
            out.append(f"<code>{esc(m.group(1))}</code>")
        elif m := re.fullmatch(r"\*\*([^*]+)\*\*", part):
            out.append(f"<strong>{esc(m.group(1))}</strong>")
        else:
            out.append(esc(part))
    return "".join(out)


def cells_of(row: str) -> list[str]:
    return [c.strip() for c in row.strip().strip("|").split("|")]


def table_fallback(rows: list[str]) -> str:
    """Medium has no table block type: emit an exact, readable list fallback."""
    header = cells_of(rows[0])
    data = [cells_of(r) for r in rows[2:]]
    assert len({len(r) for r in data} | {len(header)}) == 1, "ragged table"
    test_col = 1 if header[1] == "test" else None
    fixed = [i for i in range(len(header))
             if i != test_col and len({r[i] for r in data}) == 1]
    shown = [i for i in range(len(header)) if i not in fixed]

    def kv(row: list[str], key: int) -> str:
        return ", ".join(
            f"{header[i]} {row[i]}" if header[i] not in ("test", "Scenario") else row[i]
            for i in shown
            if i != key and row[i] not in ("", "\u2014", "-")
        )

    intro = ""
    if fixed:
        intro = "<p>" + inline(", ".join(f"{header[i]} {data[0][i]}" for i in fixed)) + " in every row.</p>"

    if test_col is not None:
        items, batch = [], None
        for r in data:
            if r[test_col].startswith("pp"):
                batch = r[test_col]
                label = f"{batch} batch"
            else:
                label = f"{batch} \u2192 {r[test_col]}"
            items.append(f"<li><strong>{esc(label)}</strong> \u2014 {inline(kv(r, test_col))}</li>")
        return intro + "<ul>" + "".join(items) + "</ul>"

    items = [
        f"<li><strong>{inline(r[0])}</strong> \u2014 {inline(kv(r, 0))}</li>"
        for r in data
    ]
    return intro + "<ul>" + "".join(items) + "</ul>"


def build(md: str) -> str:
    body = md.split("---", 2)[2].lstrip("\n").replace("<!-- more -->\n", "")
    lines = body.split("\n")
    out: list[str] = []
    buf: list[str] = []
    kind: str | None = None

    def flush():
        nonlocal kind
        if buf:
            tag = kind or "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{inline(x)}</li>" for x in buf) + f"</{tag}>")
            buf.clear()
        kind = None

    def push(t: str, k: str | None = None):
        nonlocal kind
        if kind and k != kind:
            flush()
        kind = k
        buf.append(t)

    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            flush()
            i += 1
            continue
        if s.startswith("|"):
            flush()
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            out.append(table_fallback(block))
            continue
        if s.startswith("```"):
            flush()
            i += 1
            fence = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                fence.append(lines[i])
                i += 1
            i += 1
            out.append("<pre>" + esc("\n".join(fence)) + "</pre>")
            continue
        if m := re.fullmatch(r"(#{1,3})\s+(.*)", s):
            flush()
            level = len(m.group(1))
            out.append(f"<h{level}>{inline(m.group(2))}</h{level}>")
            i += 1
            continue
        if s.startswith("> "):
            flush()
            out.append(f"<blockquote>{inline(s[2:])}</blockquote>")
            i += 1
            continue
        if s == "---":
            flush()
            out.append("<hr>")
            i += 1
            continue
        if m := re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", s):
            flush()
            out.append(
                f'<figure><img src="{m.group(2)}" alt="{esc(m.group(1))}">'
                f"<figcaption>{esc(m.group(1))}</figcaption></figure>"
            )
            i += 1
            continue
        if m := re.fullmatch(r"\d+\.\s+(.*)", s):
            push(m.group(1), "ol")
            i += 1
            continue
        if m := re.fullmatch(r"[-*]\s+(.*)", s):
            push(m.group(1), "ul")
            i += 1
            continue
        flush()
        out.append(f"<p>{inline(s)}</p>")
        i += 1
    flush()
    out.append(
        "<p><em>Originally published at "
        '<a href="https://eunomia.dev/blog/2026/10/17/gthulhu-vllm-cpu-pressure/">eunomia.dev</a>'
        " on October 17, 2026.</em></p>"
    )
    return "\n".join(out)


if __name__ == "__main__":
    result = build(SRC.read_text())
    OUT.write_text(result + "\n")
    print(f"wrote {OUT} ({len(result)} chars)")
    if "--check" in sys.argv:
        for probe in ("6066.73", "1580.57", "16853.04", "9892.80", "~6.6 tok/s",
                      "23.96", "taskset -c 0-9", "EngineCore → kernel submission",
                      "eunomia.dev", "github.com/Gthulhu/Gthulhu", "chart_decode.svg"):
            print(("ok   " if probe in result else "MISS "), probe)

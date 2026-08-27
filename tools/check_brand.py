#!/usr/bin/env python3
"""Check that every copy of the brand agrees with every other copy.

The series mark exists in more places than one component can reach. The
masthead and the footer draw it from src/brand/brand.ts, but the favicon has
to be a literal data URI in index.html (a tab needs its icon before any
JavaScript runs), and the browser chrome colour has to be a literal in a meta
tag for the same reason. Those literals are the ones that go stale.

The course kit carries its own copy of the shared brand files, so somebody can
drop them into an empty repo and start a sibling course. A kit that has
drifted from the course it was extracted from is worse than no kit, so the
shared files are compared byte for byte.

    python3 tools/check_brand.py

Exit status is 0 when everything agrees. Stdlib only.
"""

import pathlib
import re
import sys
import urllib.parse

ROOT = pathlib.Path(__file__).resolve().parent.parent
BRAND = ROOT / "src" / "brand"
KIT = ROOT / "course-kit" / "brand"

# The files a sibling course copies without editing. brand.ts is deliberately
# not here: it is the file a course does edit, so the kit's copy is a template
# with different values and comparing them would always fail.
SHARED = ["brand.css", "Masthead.tsx", "Monogram.tsx", "SeriesFooter.tsx"]


def fail(problems: list[str], message: str) -> None:
    problems.append(message)


def resolve_accent(css: str) -> tuple[str, str]:
    """The hue token --accent points at, and its value."""
    token = re.search(r"--accent:\s*var\(--hue-([a-z]+)\);", css)
    if not token:
        raise SystemExit("could not find the --accent line in brand.css")
    name = token.group(1)
    value = re.search(rf"--hue-{name}:\s*(#[0-9a-fA-F]{{6}});", css)
    if not value:
        raise SystemExit(f"--accent points at --hue-{name}, which brand.css does not define")
    return name, value.group(1).lower()


def main() -> int:
    problems: list[str] = []

    css = (BRAND / "brand.css").read_text()
    ts = (BRAND / "brand.ts").read_text()
    html = (ROOT / "index.html").read_text()

    hue_name, accent = resolve_accent(css)

    # The glyph, as brand.ts declares it.
    # (?<![A-Za-z]) so the "id:" fields above the glyph do not match "d:".
    glyph_d = re.search(r'(?<![A-Za-z])d:\s*"([^"]+)"', ts)
    stroke_width = re.search(r"strokeWidth:\s*([0-9.]+)", ts)
    view_box = re.search(r'viewBox:\s*"([^"]+)"', ts)
    if not (glyph_d and stroke_width and view_box):
        raise SystemExit("could not read the glyph out of brand.ts")

    # The favicon, as index.html declares it.
    icon = re.search(r'rel="icon"\s*\n?\s*href="data:image/svg\+xml,([^"]+)"', html)
    if not icon:
        fail(problems, "index.html has no inline data-URI favicon")
    else:
        svg = urllib.parse.unquote(icon.group(1))
        favicon_d = re.search(r"d='([^']+)'", svg)
        favicon_fill = re.search(r"fill='(#[0-9a-fA-F]{3,6})'", svg)
        favicon_stroke = re.search(r"stroke='(#[0-9a-fA-F]{3,6})'", svg)
        favicon_width = re.search(r"stroke-width='([0-9.]+)'", svg)
        favicon_view = re.search(r"viewBox='([^']+)'", svg)

        if not favicon_d or favicon_d.group(1) != glyph_d.group(1):
            got = favicon_d.group(1) if favicon_d else "(none)"
            fail(problems, f"the favicon path is\n    {got}\n  but brand.ts draws\n    {glyph_d.group(1)}")
        if not favicon_view or favicon_view.group(1) != view_box.group(1):
            got = favicon_view.group(1) if favicon_view else "(none)"
            fail(problems, f"the favicon viewBox is '{got}', brand.ts says '{view_box.group(1)}'")
        if not favicon_width or float(favicon_width.group(1)) != float(stroke_width.group(1)):
            got = favicon_width.group(1) if favicon_width else "(none)"
            fail(problems, f"the favicon stroke-width is {got}, brand.ts says {stroke_width.group(1)}")
        if not favicon_fill or favicon_fill.group(1).lower() != accent:
            got = favicon_fill.group(1) if favicon_fill else "(none)"
            fail(problems, f"the favicon tile is {got}, but --accent resolves to {accent} (--hue-{hue_name})")
        # The glyph is drawn in the ink used on a filled accent, which
        # src/styles.css calls --on-accent.
        on_accent = re.search(r"--on-accent:\s*(#[0-9a-fA-F]{3,6});", (ROOT / "src" / "styles.css").read_text())
        want_ink = on_accent.group(1).lower() if on_accent else "#fff"
        if not favicon_stroke or favicon_stroke.group(1).lower().rstrip("f") != want_ink.lower().rstrip("f"):
            got = favicon_stroke.group(1) if favicon_stroke else "(none)"
            fail(problems, f"the favicon glyph is stroked {got}, but --on-accent is {want_ink}")

    # The title, which a rename reaches in brand.ts and leaves behind in the two
    # places the HTML has to spell it out before any JavaScript runs.
    title = re.search(r"COURSE_TITLE\s*=\s*`([^`]+)`", ts)
    if not title:
        fail(problems, "could not find COURSE_TITLE in brand.ts")
    else:
        wanted = (
            title.group(1)
            .replace("${COURSE.subject}", re.search(r'subject:\s*"([^"]+)"', ts).group(1))
            .replace("${SERIES.name}", re.search(r'name:\s*"([^"]+)"', ts).group(1))
        )
        for label, pattern in [
            ("<title>", r"<title>([^<]+)</title>"),
            ('og:title', r'property="og:title"\s+content="([^"]+)"'),
        ]:
            got = re.search(pattern, html)
            if not got:
                fail(problems, f"index.html has no {label}")
            elif got.group(1) != wanted:
                fail(problems, f"index.html's {label} is {got.group(1)!r}, but COURSE_TITLE is {wanted!r}")

        # The pre-mount skeleton spells the wordmark out too, because it has to
        # paint before the masthead component exists.
        series = re.search(r'name:\s*"([^"]+)"', ts).group(1)
        if f">\n          {series}\n        </p>" not in html.replace("\r\n", "\n"):
            fail(problems, f"index.html's loading skeleton does not carry the wordmark {series!r}")

    theme = re.search(r'name="theme-color"\s+content="(#[0-9a-fA-F]{6})"', html)
    if not theme:
        fail(problems, "index.html has no theme-color meta tag")
    elif theme.group(1).lower() != accent:
        fail(problems, f"theme-color is {theme.group(1)}, but --accent resolves to {accent}")

    if not KIT.exists():
        fail(problems, f"{KIT.relative_to(ROOT)} does not exist, so the kit carries no brand")
    else:
        for name in SHARED:
            here, there = BRAND / name, KIT / name
            if not there.exists():
                fail(problems, f"course-kit/brand/{name} is missing")
            elif here.read_bytes() != there.read_bytes():
                fail(problems, f"course-kit/brand/{name} has drifted from src/brand/{name}")
        kit_ts = KIT / "brand.ts"
        if not kit_ts.exists():
            fail(problems, "course-kit/brand/brand.ts is missing")
        else:
            exports = set(re.findall(r"export (?:const|interface) (\w+)", ts))
            kit_exports = set(re.findall(r"export (?:const|interface) (\w+)", kit_ts.read_text()))
            if exports != kit_exports:
                missing = ", ".join(sorted(exports - kit_exports)) or "none"
                extra = ", ".join(sorted(kit_exports - exports)) or "none"
                fail(problems, f"the kit's brand.ts exports differ: missing {missing}, extra {extra}")

    if problems:
        print(f"{len(problems)} problem(s):\n", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1

    print(f"Brand agrees: accent --hue-{hue_name} ({accent}), one glyph in "
          f"{len(SHARED) + 2} places, the title in 3, kit in step.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

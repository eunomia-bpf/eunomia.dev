"""
Generate /en/ redirect pages that point to the root (default language) equivalents.

The i18n plugin builds the default language (en) at the site root and other
languages under /<locale>/. This hook creates lightweight HTML redirects so
that /en/<path> resolves to /<path> instead of returning 404.
"""

import os
import shutil


_REDIRECT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="0; url={target}">
  <link rel="canonical" href="{target}">
  <title>Redirect</title>
</head>
<body>
  <p>Redirecting to <a href="{target}">{target}</a>...</p>
</body>
</html>
"""


def on_post_build(config, **kwargs):
    site_dir = config["site_dir"]
    en_dir = os.path.join(site_dir, "en")

    # Walk the root site_dir and mirror every .html file into en/
    for dirpath, dirnames, filenames in os.walk(site_dir):
        # Skip the en/ and zh/ (and any other locale) subdirectories
        rel = os.path.relpath(dirpath, site_dir)
        if rel == "en" or rel.startswith("en" + os.sep):
            continue
        if rel == "zh" or rel.startswith("zh" + os.sep):
            continue

        for fname in filenames:
            if not fname.endswith(".html"):
                continue

            # Compute the root-relative URL this file serves
            rel_path = os.path.relpath(os.path.join(dirpath, fname), site_dir)
            # Target URL: from /en/<rel_path> go to /<rel_path>
            target = "/" + rel_path.replace(os.sep, "/")
            # Normalise index.html → directory URL
            if target.endswith("/index.html"):
                target = target[: -len("index.html")]

            dest = os.path.join(en_dir, rel_path)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "w") as f:
                f.write(_REDIRECT_TEMPLATE.format(target=target))

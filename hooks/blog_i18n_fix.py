"""
Fix compatibility between mkdocs-material blog plugin and mkdocs-static-i18n.

The blog plugin marks posts as EXCLUDED in on_files(-50), then changes them
to NOT_IN_NAV in on_nav(-50). The i18n plugin creates NEW File objects in its
on_files(-100), copying the EXCLUDED inclusion level. Since the copies are
separate objects, the blog plugin's later NOT_IN_NAV update never reaches them.

Fix: patch the i18n plugin's suffix.create_i18n_file to use NOT_IN_NAV instead
of EXCLUDED for blog post files, so posts are built for all languages.
"""

from mkdocs.structure.files import InclusionLevel


def on_config(config):
    if "material/blog" not in config.plugins or "i18n" not in config.plugins:
        return

    from mkdocs_static_i18n import suffix

    _original_create = suffix.create_i18n_file

    def _patched_create(file, current_language, default_language, all_languages, mkdocs_config):
        result = _original_create(file, current_language, default_language, all_languages, mkdocs_config)
        # Blog posts are EXCLUDED (not in nav, not built). Change to NOT_IN_NAV
        # so they are still built but stay out of the navigation tree.
        if result.inclusion == InclusionLevel.EXCLUDED and file.src_path.startswith("blog/posts/"):
            result.inclusion = InclusionLevel.NOT_IN_NAV
        return result

    suffix.create_i18n_file = _patched_create

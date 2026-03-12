"""
Fix compatibility between mkdocs-material blog plugin and mkdocs-static-i18n.

Issues caused by i18n creating new File/Page objects that break identity checks:
1. Blog posts marked EXCLUDED never get built → patch create_i18n_file
2. on_page_markdown skips posts → no excerpts created → fix via hook
3. on_page_context skips blog index → no post listings → fix via hook
4. Blog index uses page.html instead of blog.html template → fix via hook
"""

from mkdocs.structure.files import InclusionLevel


def on_config(config):
    if "material/blog" not in config.plugins or "i18n" not in config.plugins:
        return

    # Fix 1: patch create_i18n_file to not copy EXCLUDED for blog posts
    from mkdocs_static_i18n import suffix

    _original_create = suffix.create_i18n_file

    def _patched_create(file, current_language, default_language, all_languages, mkdocs_config):
        result = _original_create(file, current_language, default_language, all_languages, mkdocs_config)
        if result.inclusion == InclusionLevel.EXCLUDED and file.src_path.startswith("blog/posts/"):
            result.inclusion = InclusionLevel.NOT_IN_NAV
        return result

    suffix.create_i18n_file = _patched_create


def on_page_markdown(markdown, page, config, files):
    """Fix 2+4: Create excerpts for blog posts, set blog template for index."""
    blog_plugin = config.plugins.get("material/blog")
    if not blog_plugin or not blog_plugin.config.enabled:
        return
    if not hasattr(blog_plugin, 'blog'):
        return

    # Fix 4: Force blog.html template on blog index BEFORE _build_page reads it
    if page.file.src_path == blog_plugin.blog.file.src_path:
        page.meta.setdefault("template", "blog.html")

    if page in blog_plugin.blog.posts:
        return  # Blog plugin will handle it

    post_paths = {p.file.src_path: p for p in blog_plugin.blog.posts}
    if page.file.src_path not in post_paths:
        return

    from material.plugins.blog.structure import Excerpt
    original_post = post_paths[page.file.src_path]

    original_post.markdown = markdown
    original_post.excerpt = Excerpt(original_post, config, files)
    original_post.excerpt.authors = original_post.authors[:blog_plugin.config.post_excerpt_max_authors]
    original_post.excerpt.categories = original_post.categories[:blog_plugin.config.post_excerpt_max_categories]


def on_page_context(context, page, config, nav):
    """Fix 3: Inject post listings for blog index."""
    blog_plugin = config.plugins.get("material/blog")
    if not blog_plugin or not blog_plugin.config.enabled:
        return
    if not hasattr(blog_plugin, 'blog') or not blog_plugin.blog:
        return
    if page.file.src_path != blog_plugin.blog.file.src_path:
        return

    if "posts" in context:
        return  # Blog plugin already handled it

    # Ensure blog view has a proper toc object
    blog_view = blog_plugin.blog
    if not hasattr(blog_view.toc, 'items'):
        from mkdocs.structure.toc import get_toc
        blog_view.toc = get_toc("")

    # Fix 3: Render post excerpts into context
    posts = []
    for post in blog_plugin.blog.posts:
        if hasattr(post, 'excerpt') and post.excerpt:
            posts.append(blog_plugin._render_post(post.excerpt, blog_view))

    context["posts"] = posts
    context["pagination"] = None

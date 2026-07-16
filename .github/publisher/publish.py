#!/usr/bin/env python3
"""
Auto-publish posts to Medium and Dev.to using the media publisher API.
Reads posts from the queue, publishes them, and removes successful entries from the queue.
"""

import json
import os
import sys
import re
import unicodedata
from collections import namedtuple

import requests


API_ENDPOINT = "https://media-publisher.vercel.app/api/publish-multi"
QUEUE_FILE = ".github/publisher/posts_queue.txt"
DEFAULT_PUBLISH_COUNT = 2
SITE_URL = "https://eunomia.dev"


def has_yaml_frontmatter(content):
    """Check if content starts with YAML frontmatter."""
    return content.strip().startswith('---')


def extract_title_from_markdown(content):
    """
    Extract the first H1 title from markdown content.
    Supports:
    1. YAML frontmatter with title in H1
    2. # Title format
    3. Title\n=== format
    """
    lines = content.strip().split('\n')
    start_index = 0

    # Skip YAML frontmatter if present
    if lines and lines[0].strip() == '---':
        # Find the closing ---
        for i in range(1, len(lines)):
            if lines[i].strip() == '---':
                start_index = i + 1
                break

    # Look for # Title format (after frontmatter)
    for i in range(start_index, len(lines)):
        line = lines[i].strip()
        if line.startswith('# '):
            title = line[2:].strip()
            return title, i

    # Look for Title\n=== format
    for i in range(start_index, len(lines) - 1):
        if lines[i+1].strip().startswith('===') or lines[i+1].strip().startswith('---'):
            if lines[i].strip():
                title = lines[i].strip()
                return title, i

    raise ValueError("No title found in markdown content")


def remove_title_from_content(content):
    """
    Remove the first title and YAML frontmatter from markdown content.
    Returns cleaned content suitable for publishing to Medium/Dev.to.
    """
    lines = content.strip().split('\n')
    start_index = 0

    # Skip YAML frontmatter if present
    if lines and lines[0].strip() == '---':
        for i in range(1, len(lines)):
            if lines[i].strip() == '---':
                start_index = i + 1
                break

    # Remove # Title format
    for i in range(start_index, len(lines)):
        if lines[i].strip().startswith('# '):
            # Remove the title line and any following empty lines
            new_lines = lines[i+1:]
            while new_lines and not new_lines[0].strip():
                new_lines.pop(0)
            return '\n'.join(new_lines).strip()

    # Remove Title\n=== format
    for i in range(start_index, len(lines) - 1):
        if lines[i+1].strip().startswith('===') or lines[i+1].strip().startswith('---'):
            if lines[i].strip():
                # Remove title and underline
                new_lines = lines[i+2:]
                while new_lines and not new_lines[0].strip():
                    new_lines.pop(0)
                return '\n'.join(new_lines).strip()

    # No title found, just remove frontmatter if it exists
    if start_index > 0:
        return '\n'.join(lines[start_index:]).strip()

    return content


def slugify_title(value):
    """
    Slugify a title using the same rules as the site content pipeline
    (app/lib/content/source.ts slugifyTitle): lowercase, NFKD-normalize, drop
    combining marks, collapse runs of non-alphanumeric characters into a single
    hyphen, and trim leading/trailing hyphens.
    """
    text = unicodedata.normalize("NFKD", value.lower())
    text = "".join(ch for ch in text if not unicodedata.category(ch).startswith("M"))
    chars = []
    prev_hyphen = False
    for ch in text:
        if ch.isalnum():
            chars.append(ch)
            prev_hyphen = False
        elif not prev_hyphen:
            chars.append("-")
            prev_hyphen = True
    slug = "".join(chars).strip("-")
    if slug:
        return slug
    return re.sub(r"\s+", "-", value.strip())


# Parsed frontmatter scalar: `text` is the value with quotes/comments removed,
# `is_string` reflects whether js-yaml (used by gray-matter on the site) would
# treat the value as a string. The site's parseSlug ignores non-string slugs.
FrontmatterScalar = namedtuple("FrontmatterScalar", ["text", "is_string"])

# YAML 1.1 (js-yaml) parses these unquoted scalars as non-strings (numbers,
# booleans, null), which the site's parseSlug rejects.
_YAML_NONSTRING_SCALAR = re.compile(
    r"^(?:"
    r"[+-]?\d+"                              # integer
    r"|[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?"  # float / scientific
    r"|0x[0-9a-fA-F]+|0o[0-7]+"              # hex / octal
    r"|true|false|null|~"                    # bool / null
    r"|yes|no|on|off"                        # YAML 1.1 booleans
    r")$",
    re.IGNORECASE,
)


def _parse_frontmatter_scalar(raw):
    """Interpret one unparsed frontmatter scalar the way js-yaml would for the
    fields we consume: a quoted value is a string; an unquoted value has any
    inline `# comment` stripped and is flagged as a string only when YAML would
    not parse it as a number, boolean, or null."""
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ("'", '"'):
        return FrontmatterScalar(raw[1:-1], True)
    raw = re.sub(r"\s+#.*$", "", raw).strip()
    is_string = bool(raw) and not _YAML_NONSTRING_SCALAR.match(raw)
    return FrontmatterScalar(raw, is_string)


def parse_frontmatter(content):
    """Parse top-level scalar fields from YAML frontmatter (no external deps).

    Returns a mapping of field name to FrontmatterScalar. Only simple inline
    scalars are recognized; block/flow values are out of scope for the canonical
    URL derivation below.
    """
    lines = content.split("\n")
    if not lines or lines[0].strip() != "---":
        return {}

    fields = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        match = re.match(r"^([A-Za-z0-9_-]+):\s*(.*)$", line)
        if match:
            fields[match.group(1)] = _parse_frontmatter_scalar(match.group(2))
    return fields


def derive_canonical_url(post_path, content, title):
    """
    Derive the canonical eunomia.dev URL for a source markdown path so
    syndicated copies (dev.to, Medium) point back to the original.

    Blog posts under docs/blog/posts/<name>.md map to
    https://eunomia.dev/blog/YYYY/MM/DD/<slug>/ where YYYY-MM-DD comes from the
    frontmatter `date` and <slug> is the frontmatter `slug` if present, otherwise
    the slugified title (matching the site's content pipeline). Other docs paths
    map to https://eunomia.dev/<path-without-docs-prefix-and-README.md>/.
    Returns None when no canonical URL can be derived.
    """
    normalized = post_path.replace("\\", "/")
    match = re.search(r"(?:^|/)docs/(.+)$", normalized)
    if not match:
        return None
    rel = match.group(1)

    blog_match = re.match(r"^blog/posts/(?:.+?)(\.zh)?\.md$", rel)
    if blog_match:
        is_zh = bool(blog_match.group(1))
        # The site derives blog date/slug from the English source for both locales
        # (app/lib/content/discovery.ts prefers sourceByLocale.en), so resolve the
        # English sibling when the queued file is the Chinese translation.
        source_content, source_title = content, title
        if is_zh:
            en_sibling = re.sub(r"\.zh\.md$", ".md", post_path)
            if os.path.isfile(en_sibling):
                with open(en_sibling, "r", encoding="utf-8") as sibling:
                    source_content = sibling.read()
                try:
                    source_title, _ = extract_title_from_markdown(source_content)
                except ValueError:
                    source_title = title
        frontmatter = parse_frontmatter(source_content)
        # The site parses `date` with new Date(...).toISOString() (UTC). A bare,
        # zero-padded YYYY-MM-DD maps unambiguously to that UTC day; a value with a
        # time/zone component (or a non-padded one) can shift the UTC day, so omit
        # the canonical rather than emit a wrong one.
        date_field = frontmatter.get("date")
        date_match = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", date_field.text) if date_field else None
        if not date_match:
            return None
        year, month, day = date_match.groups()
        # Mirror app/lib/content/markdown.ts: a string frontmatter `title` wins,
        # otherwise fall back to the H1 heading.
        title_field = frontmatter.get("title")
        fallback_title = title_field.text if title_field and title_field.is_string else source_title
        # Mirror parseSlug: only a string frontmatter `slug` is used (slugified);
        # non-string YAML scalars and empty slugs fall back to the slugified title.
        slug_field = frontmatter.get("slug")
        slug = slugify_title(slug_field.text) if slug_field and slug_field.is_string else ""
        if not slug:
            slug = slugify_title(fallback_title)
        locale_prefix = "/zh" if is_zh else ""
        return f"{SITE_URL}{locale_prefix}/blog/{year}/{month}/{day}/{slug}/"

    # Non-blog docs: strip the locale suffix (.zh.md/.zh-CN.md map to the /zh
    # tree; .en.md/.md map to the default tree), then collapse a trailing README
    # or index segment since the site serves those at the section root.
    locale_match = re.search(r"\.(zh-CN|zh|en)\.md$", rel)
    if locale_match:
        is_zh = locale_match.group(1) in ("zh", "zh-CN")
        rel = rel[: locale_match.start()]
    elif rel.endswith(".md"):
        is_zh = False
        rel = rel[: -len(".md")]
    else:
        return None
    rel = re.sub(r"(?:^|/)(README|index)$", "", rel)
    rel = rel.strip("/")
    if not rel:
        return None
    locale_prefix = "/zh" if is_zh else ""
    return f"{SITE_URL}{locale_prefix}/{rel}/"


def read_queue():
    """Read the posts queue file and return list of posts."""
    if not os.path.exists(QUEUE_FILE):
        print(f"Queue file not found: {QUEUE_FILE}")
        return []

    with open(QUEUE_FILE, 'r') as f:
        lines = f.readlines()

    posts = []
    for line in lines:
        line = line.strip()
        if line:
            try:
                post = json.loads(line)
                posts.append(post)
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON line: {line}")
                print(f"Error: {e}")

    return posts


def write_queue(posts):
    """Write the posts list back to the queue file."""
    with open(QUEUE_FILE, 'w') as f:
        for post in posts:
            f.write(json.dumps(post) + '\n')


def validate_post_path(post_path):
    """Validate that a queue path points to a markdown file."""
    if os.path.isfile(post_path):
        if not post_path.endswith(".md"):
            raise ValueError(f"Post path must point to a markdown file: {post_path}")
        return post_path

    if os.path.isdir(post_path):
        raise ValueError(
            f"Post path is a directory; set 'path' to the exact markdown file: {post_path}"
        )

    raise FileNotFoundError(f"Post file not found: {post_path}")


def get_publish_count():
    """Return how many queued posts this run should process."""
    raw_value = os.environ.get('PUBLISH_COUNT', str(DEFAULT_PUBLISH_COUNT))
    try:
        publish_count = int(raw_value)
    except ValueError:
        raise ValueError(f"PUBLISH_COUNT must be an integer, got: {raw_value}")

    if publish_count < 1:
        raise ValueError(f"PUBLISH_COUNT must be at least 1, got: {publish_count}")

    return publish_count


def prepare_post(post, index):
    """Read and validate a queued post."""
    post_path = post.get('path')
    tags = post.get('tags', [])

    if not post_path:
        raise ValueError(f"Post #{index} missing 'path' field")

    validated_post_path = validate_post_path(post_path)

    with open(validated_post_path, 'r', encoding='utf-8') as f:
        original_content = f.read()

    title, _title_line = extract_title_from_markdown(original_content)
    cleaned_content = remove_title_from_content(original_content)
    canonical_url = derive_canonical_url(validated_post_path, original_content, title)

    return {
        "title": title,
        "content": cleaned_content,
        "path": validated_post_path,
        "tags": tags,
        "canonical_url": canonical_url
    }


def print_post_preview(prepared_post, index, total):
    """Print the post summary used by dry runs and publish logs."""
    print(f"\nPost {index}/{total}")
    print(f"Title: {prepared_post['title']}")
    print(f"Path: {prepared_post['path']}")
    print(f"Tags: {', '.join(prepared_post['tags'])}")
    print(f"Canonical URL: {prepared_post.get('canonical_url') or '(none)'}")
    print(f"Content preview: {prepared_post['content'][:200]}...")


def publish_post(title, content, tags, password, is_draft=True, canonical_url=None):
    """
    Publish a post to Medium and Dev.to via the API.
    """
    payload = {
        "title": title,
        "content": content,
        "tags": tags,
        "is_draft": is_draft,
        "platforms": ["devto", "medium"]
    }
    # Canonical URL points syndicated copies back to the original on eunomia.dev.
    # The media-publisher Vercel service is responsible for forwarding this to
    # dev.to (canonical_url) and Medium (canonicalUrl).
    if canonical_url:
        payload["canonical_url"] = canonical_url

    headers = {
        "Content-Type": "application/json",
        "x-publish-password": password
    }

    try:
        response = requests.post(API_ENDPOINT, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error publishing post: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}")
        raise


def main():
    # Get options from environment variables
    dry_run = os.environ.get('DRY_RUN', 'false').lower() == 'true'
    draft_only = os.environ.get('DRAFT_ONLY', 'false').lower() == 'true'
    password = os.environ.get('PUBLISH_PASSWORD')
    try:
        publish_count = get_publish_count()
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # In scheduled runs, always publish as live posts (not drafts)
    # In manual runs, respect the draft_only flag
    is_draft = draft_only

    if dry_run:
        print("🔍 DRY RUN MODE - No actual publishing will occur\n")

    if is_draft:
        print("📝 DRAFT MODE - Posts will be published as drafts\n")
    else:
        print("🚀 LIVE MODE - Posts will be published publicly\n")

    if not password and not dry_run:
        print("Error: PUBLISH_PASSWORD environment variable not set")
        sys.exit(1)

    # Read the queue
    posts = read_queue()

    if not posts:
        print("No posts in queue to publish")
        return

    posts_to_publish = posts[:publish_count]
    prepared_posts = []
    for index, post in enumerate(posts_to_publish, start=1):
        try:
            prepared_post = prepare_post(post, index)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}")
            sys.exit(1)
        prepared_posts.append(prepared_post)
        print_post_preview(prepared_post, index, len(posts_to_publish))

    if dry_run:
        print(f"\n✅ DRY RUN COMPLETE - {len(prepared_posts)} posts validated successfully")
        print("❌ Not publishing (dry run mode)")
        print("❌ Not removing from queue (dry run mode)")
        return

    for index, prepared_post in enumerate(prepared_posts, start=1):
        try:
            result = publish_post(
                prepared_post["title"],
                prepared_post["content"],
                prepared_post["tags"],
                password,
                is_draft,
                prepared_post.get("canonical_url")
            )
            print(f"\n✅ Publish {index}/{len(prepared_posts)} successful!")
            print(json.dumps(result, indent=2))

            posts.pop(0)
            write_queue(posts)
            print(f"\n✅ Removed post from queue. {len(posts)} posts remaining.")

        except Exception as e:
            print(f"\n❌ Publish failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()

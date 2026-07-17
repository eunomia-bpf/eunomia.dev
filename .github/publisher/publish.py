#!/usr/bin/env python3
"""
Auto-publish posts to Medium and Dev.to using the media publisher API.
Reads posts from the queue, publishes them, and removes successful entries from the queue.
"""

import json
import os
import sys
import re
import requests


API_ENDPOINT = "https://media-publisher.vercel.app/api/publish-multi"
QUEUE_FILE = ".github/publisher/posts_queue.txt"
DEFAULT_PUBLISH_COUNT = 2


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

    return {
        "title": title,
        "content": cleaned_content,
        "path": validated_post_path,
        "tags": tags
    }


def print_post_preview(prepared_post, index, total):
    """Print the post summary used by dry runs and publish logs."""
    print(f"\nPost {index}/{total}")
    print(f"Title: {prepared_post['title']}")
    print(f"Path: {prepared_post['path']}")
    print(f"Tags: {', '.join(prepared_post['tags'])}")
    print(f"Content preview: {prepared_post['content'][:200]}...")


def publish_post(title, content, tags, password, is_draft=True):
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
                is_draft
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

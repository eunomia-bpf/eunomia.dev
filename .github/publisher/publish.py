#!/usr/bin/env python3
"""
Auto-publish posts to Medium and Dev.to using the media publisher API.
Reads the first post from the queue, publishes it, and removes it from the queue.
"""

import json
import os
import sys
import re
import requests


API_ENDPOINT = "https://media-publisher.vercel.app/api/publish-multi"
QUEUE_FILE = ".github/publisher/posts_queue.txt"


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

    # Get the first post
    post = posts[0]
    post_path = post.get('path')
    tags = post.get('tags', [])

    if not post_path:
        print("Error: Post missing 'path' field")
        sys.exit(1)

    # Read the post content
    if not os.path.exists(post_path):
        print(f"Error: Post file not found: {post_path}")
        sys.exit(1)

    with open(post_path, 'r', encoding='utf-8') as f:
        original_content = f.read()

    # Extract title
    try:
        title, title_line = extract_title_from_markdown(original_content)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Remove title from content
    cleaned_content = remove_title_from_content(original_content)

    print(f"Title: {title}")
    print(f"Path: {post_path}")
    print(f"Tags: {', '.join(tags)}")
    print(f"Content preview: {cleaned_content[:200]}...")

    if dry_run:
        print("\n✅ DRY RUN COMPLETE - Post validated successfully")
        print("❌ Not publishing (dry run mode)")
        print("❌ Not removing from queue (dry run mode)")
        return

    # Publish the post
    try:
        result = publish_post(title, cleaned_content, tags, password, is_draft)
        print(f"\n✅ Publish successful!")
        print(json.dumps(result, indent=2))

        # Remove the published post from the queue
        posts.pop(0)
        write_queue(posts)
        print(f"\n✅ Removed post from queue. {len(posts)} posts remaining.")

    except Exception as e:
        print(f"\n❌ Publish failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

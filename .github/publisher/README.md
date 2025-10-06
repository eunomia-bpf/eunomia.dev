# Auto Publisher Setup

Automatically publish posts to Medium and Dev.to on a weekly schedule or on-demand.

## Files

- `posts_queue.txt` - Queue of posts to publish (one JSON object per line)
- `publish.py` - Python script that publishes the first post in the queue
- `../workflows/publish-posts.yml` - GitHub Actions workflow

## Setup

### 1. Add GitHub Secret

Add your publish password as a repository secret:

1. Go to repository Settings → Secrets and variables → Actions
2. Create a new secret named `PUBLISH_PASSWORD`
3. Set the value to `11447722` (or your custom password)

### 2. Add Posts to Queue

Edit `posts_queue.txt` and add posts in JSON format (one per line):

```json
{"path": "posts/my-first-post.md", "tags": ["tutorial", "programming"]}
{"path": "posts/another-post.md", "tags": ["devops", "automation"]}
{"path": "posts/third-post.md", "tags": ["python", "api"]}
```

**Fields:**
- `path` - Relative path to the markdown post file
- `tags` - Array of tags for the post

### 3. Post Format

Posts must be in Markdown format with a title as the first H1 heading:

```markdown
# My Post Title

Content goes here...

## Section 1

More content...
```

The script will:
- Extract the title from the first H1 heading
- Remove the title from the content before publishing
- Publish to both Medium and Dev.to as drafts

## Usage

### Automatic (Weekly)

The workflow runs automatically every Monday at 9:00 AM UTC and publishes posts as **live** (publicly visible).

### Manual Trigger

1. Go to Actions tab in GitHub
2. Select "Auto Publish Posts" workflow
3. Click "Run workflow"
4. Configure options:
   - **Dry run**: Preview post without publishing (validates format only)
   - **Publish as draft only**: Publish to platforms as drafts (default: true)
5. Click "Run workflow" button

**Manual run modes:**
- **Dry run enabled**: Validates post format, shows preview, doesn't publish or modify queue
- **Draft only enabled**: Publishes as drafts (not publicly visible)
- **Draft only disabled**: Publishes live posts (publicly visible immediately)

## How It Works

1. Script reads the first post from `posts_queue.txt`
2. Reads the markdown file and extracts the title
3. Removes title from content
4. Publishes to Medium and Dev.to via API (as drafts)
5. Removes the published post from the queue
6. Commits the updated queue file

## Example Post Entry

For a post at `posts/2024/my-awesome-tutorial.md`:

```json
{"path": "posts/2024/my-awesome-tutorial.md", "tags": ["tutorial", "javascript", "web"]}
```

The post file should look like:

```markdown
# Building a Modern Web App

In this tutorial, we'll explore...

## Prerequisites

- Node.js installed
- Basic JavaScript knowledge

## Getting Started

Let's start by...
```

## Troubleshooting

- **No posts published**: Check that `posts_queue.txt` has entries
- **Authentication error**: Verify `PUBLISH_PASSWORD` secret is set correctly
- **File not found**: Ensure the `path` in the queue points to an existing file
- **No title found**: Make sure your markdown has a `# Title` as the first heading

## Queue Management

Posts are processed in order (first in, first out). After each successful publish, the post is automatically removed from the queue.

To stop publishing a post, simply remove its line from `posts_queue.txt`.

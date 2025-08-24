# Link Checker Documentation

## Overview

The `check_links.py` script is a comprehensive link checker for the eunomia.dev documentation. It scans all markdown files, extracts URLs, and verifies their availability.

## Features

- **Concurrent checking**: Uses thread pool for fast parallel checking
- **Multiple output formats**: Text, JSON, or Markdown reports
- **Smart filtering**: Skips localhost and known problematic URLs
- **Detailed reporting**: Shows which files contain broken links
- **Fix suggestions**: Can identify and suggest fixes for internal links
- **CI/CD ready**: Returns exit code 1 if broken links are found

## Usage

### Basic Usage

```bash
python .github/check_links.py
```

### Advanced Options

```bash
# Generate markdown report for GitHub issues
python .github/check_links.py --output-format markdown --output-file link-report.md

# Check with custom timeout and more workers
python .github/check_links.py --timeout 30 --max-workers 20

# Generate JSON report for automation
python .github/check_links.py --output-format json --output-file results.json

# Check and suggest fixes for internal links
python .github/check_links.py --fix-internal
```

## GitHub Actions Integration

Create `.github/workflows/check-links.yml`:

```yaml
name: Check Links

on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday
  workflow_dispatch:
  pull_request:
    paths:
      - '**.md'

jobs:
  check-links:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.x'
      
      - name: Install dependencies
        run: pip install requests
      
      - name: Check links
        run: python .github/check_links.py --output-format markdown --output-file link-report.md
        continue-on-error: true
      
      - name: Upload report
        if: failure()
        uses: actions/upload-artifact@v3
        with:
          name: link-report
          path: link-report.md
      
      - name: Comment PR
        if: github.event_name == 'pull_request' && failure()
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const report = fs.readFileSync('link-report.md', 'utf8');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: report
            });
```

## Known Issues

The script identifies several categories of broken links:

1. **Internal Links**: Many `/tutorials/` and `/blogs/` paths need updating
2. **Academic Papers**: arxiv.org and ACM links often return 403/404
3. **External Blogs**: Some technical blogs have moved or been deleted
4. **Documentation**: Some official docs (kernel.org, NVIDIA) are outdated

## Maintenance

### Adding Skip Patterns

Edit the `SKIP_URLS` list in the script:

```python
SKIP_URLS = [
    'http://localhost',
    'http://127.0.0.1',
    'https://example.com',  # Add problematic domains
]
```

### Adding Fix Rules

Edit the `KNOWN_REPLACEMENTS` dictionary:

```python
KNOWN_REPLACEMENTS = {
    'https://eunomia.dev/blogs/': 'https://eunomia.dev/blog/',
    # Add more replacement rules
}
```

## Output Examples

### Text Output
```
================================================================================
LINK CHECK REPORT
Generated: 2024-01-15 10:30:00
================================================================================

SUMMARY:
Total URLs checked: 1937
Working links: 1045
Broken links: 892
Skipped links: 10
Success rate: 54.0%

================================================================================
BROKEN LINKS (sorted by frequency)
================================================================================

URL: https://eunomia.dev/tutorials/
Status: HTTP 404
Found in 33 file(s):
  - docs/tutorials/6-sigsnoop/README.md
  - docs/tutorials/4-opensnoop/README.md
  ... and 31 more files
```

### Markdown Output
```markdown
# Link Check Report

**Generated:** 2024-01-15 10:30:00

## Summary

- **Total URLs checked:** 1937
- **Working links:** 1045 ✅
- **Broken links:** 892 ❌
- **Skipped links:** 10 ⏭️
- **Success rate:** 54.0%

## Broken Links

| URL | Status/Error | Files | Count |
|-----|--------------|-------|-------|
| https://eunomia.dev/tutorials/ | HTTP 404 | docs/tutorials/6-sigsnoop/README.md (+32 more) | 33 |
```

## Contributing

To improve the link checker:

1. Add new features to handle specific URL patterns
2. Improve error handling for edge cases
3. Add automatic fixing capabilities
4. Enhance reporting formats

Submit pull requests to the `.github/` directory.
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the source code for the eunomia-bpf project website (https://eunomia.dev), a documentation site built with MkDocs and Material theme. The site provides comprehensive tutorials and documentation for eBPF programming, the eunomia-bpf framework, bpftime, and related projects.

## Common Development Commands

### Documentation Development
```bash
# Install all dependencies
make install

# Run local development server (http://localhost:8000)
mkdocs serve

# Build the static site
make build
# or
mkdocs build

# Clean build artifacts
make clean
```

### Content Synchronization
```bash
# Clone/update the tutorial repository
make tutorial

# Clone/update other documentation repositories
make bpftime
make cuda-exp
make cupti-exp
```

**Important**: Tutorials are maintained in a separate repository (https://github.com/eunomia-bpf/bpf-developer-tutorial). Edit tutorials there, not in this repository.

## Architecture and Structure

### Documentation Organization
- `/docs/` - All website content
  - `tutorials/` - eBPF programming tutorials (synced from external repo)
  - `blog/` & `blogs/` - Technical blog posts
  - `eunomia-bpf/` - Framework documentation
  - `bpftime/` - Userspace eBPF runtime docs
  - `GPTtrace/` - GPTtrace tool documentation
  - `wasm-bpf/` - WebAssembly/eBPF integration docs
  - `others/` - Additional content (CUDA tutorials, ideas)

### Key Configuration
- `mkdocs.yaml` - Main site configuration
  - Configures Material theme, navigation, plugins
  - Sets up i18n for English/Chinese support
  - Defines site structure and features

### Deployment
- GitHub Actions automatically builds and deploys to GitHub Pages
- Deployed to the `docs` branch
- Triggers on push, PR, and repository dispatch events

### Content Management
- The site is multilingual (English and Chinese)
- Uses mkdocs-static-i18n for internationalization
- Blog posts support tags and RSS feeds
- Git revision date and authors are tracked automatically

### Working with eBPF Tutorial Code
When working with eBPF examples in the tutorials:
- Each tutorial has its own Makefile using the libbpf build system
- BPF programs use `.bpf.c` extension
- User-space programs are regular `.c` files
- Build with `make` in the tutorial directory
- Clean with `make clean`

## Important Notes
- Do not edit tutorial content directly in this repo - edit in the bpf-developer-tutorial repository
- Project home pages should be edited in their respective repository README files
- The site automatically syncs content from external repositories during build
- Always test changes locally with `mkdocs serve` before committing
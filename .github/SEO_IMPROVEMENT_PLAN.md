# SEO Improvement Plan for eunomia.dev

## Current State Analysis

### ✅ Existing SEO Features
- Basic site metadata (name, description, URL)
- Google Analytics integration with user feedback
- Social sharing buttons (X/Facebook) via custom hooks
- MkDocs Material social plugin for preview image generation
- Multi-language support (English/Chinese) with proper alternate links
- Git metadata tracking (revision dates, authors)
- Clean URL structure with organized navigation

### ❌ Missing SEO Elements
1. **Meta Tags**
   - No Open Graph tags (og:title, og:description, og:image, og:type)
   - No Twitter Card tags (twitter:card, twitter:title, twitter:description, twitter:image)
   - No per-page meta descriptions
   - No canonical URLs

2. **Technical SEO**
   - No robots.txt file
   - No explicit sitemap.xml configuration
   - No structured data (JSON-LD)
   - No meta keywords (though less important nowadays)

3. **Content Optimization**
   - Limited page-specific metadata
   - No SEO-focused content guidelines

## Implementation Plan

### Phase 1: Meta Tags Enhancement (Priority: High)

#### 1.1 Install and Configure Meta Plugin
```yaml
# mkdocs.yaml
plugins:
  - meta
```

#### 1.2 Create Base Template Override
Create `/material/overrides/partials/head.html` to add:
- Open Graph meta tags
- Twitter Card meta tags
- Canonical URLs
- Additional SEO meta tags

#### 1.3 Add Page-Level Meta Descriptions
For key pages, add front matter:
```yaml
---
description: "Comprehensive eBPF programming tutorials and documentation for developers"
tags:
  - eBPF
  - BPF
  - Linux
  - Kernel Programming
---
```

### Phase 2: Technical SEO (Priority: High)

#### 2.1 Create robots.txt
```
# /docs/robots.txt
User-agent: *
Allow: /
Sitemap: https://eunomia.dev/sitemap.xml
```

#### 2.2 Configure Sitemap Generation
```yaml
# mkdocs.yaml
plugins:
  - sitemap:
      changefreq: weekly
      priority: 0.5
```

#### 2.3 Add Structured Data
Implement JSON-LD for:
- Organization schema
- Website schema
- Article schema for blog posts
- Tutorial/HowTo schema for tutorials

### Phase 3: Content Optimization (Priority: Medium)

#### 3.1 Optimize Page Titles
- Keep under 60 characters
- Include primary keywords
- Make them descriptive and unique

#### 3.2 Improve Meta Descriptions
- Keep under 160 characters
- Include call-to-action
- Make them unique per page

#### 3.3 Header Tag Optimization
- Ensure proper H1-H6 hierarchy
- Include keywords naturally
- One H1 per page

### Phase 4: Performance & User Experience (Priority: Medium)

#### 4.1 Image Optimization
- Add alt text to all images
- Use descriptive filenames
- Implement lazy loading

#### 4.2 Internal Linking
- Add related content links
- Create topic clusters
- Improve navigation depth

### Phase 5: Monitoring & Analytics (Priority: Low)

#### 5.1 Search Console Integration
- Verify site ownership
- Submit sitemap
- Monitor search performance

#### 5.2 Enhanced Analytics
- Track search terms
- Monitor page performance
- Set up goal tracking

## MkDocs Material Native Features Research

### Built-in SEO Features

1. **Automatic Features**
   - Clean URLs
   - Mobile-responsive design
   - Fast page loading
   - Semantic HTML structure

2. **Configuration Options**
   - `site_url`: Essential for canonical URLs
   - `site_description`: Global meta description
   - `site_name`: Used in title tags
   - `extra.social`: Social media links

3. **Plugin Ecosystem**
   - `mkdocs-meta-plugin`: Per-page metadata
   - `mkdocs-sitemap`: Automatic sitemap generation
   - `mkdocs-minify-plugin`: HTML/CSS/JS minification
   - `mkdocs-redirects`: Handle URL redirects

### Recommended Configuration

```yaml
# mkdocs.yaml additions
site_name: eunomia - eBPF Development Platform
site_description: Learn eBPF programming with comprehensive tutorials, tools, and documentation. Build high-performance Linux kernel programs with eunomia-bpf.
site_url: https://eunomia.dev
site_author: eunomia-bpf community

extra:
  meta:
    - name: keywords
      content: eBPF, BPF, Linux, kernel programming, tutorials, eunomia-bpf, bpftime
    - name: author
      content: eunomia-bpf community
  
plugins:
  - search:
      lang: 
        - en
        - zh
  - meta
  - sitemap:
      changefreq: weekly
      priority: 0.5
      filename: sitemap.xml
  - minify:
      minify_html: true
      minify_js: true
      minify_css: true
      htmlmin_opts:
        remove_comments: true
```

## Timeline

- **Week 1-2**: Implement Phase 1 (Meta Tags)
- **Week 3**: Implement Phase 2 (Technical SEO)
- **Week 4**: Implement Phase 3 (Content Optimization)
- **Month 2**: Implement Phase 4-5 (Performance & Monitoring)

## Success Metrics

1. **Technical Metrics**
   - All pages have unique meta descriptions
   - Open Graph tags present on all pages
   - Valid sitemap.xml accessible
   - robots.txt properly configured

2. **Performance Metrics**
   - Improved search engine visibility
   - Higher click-through rates
   - Better social media engagement
   - Increased organic traffic

## Next Steps

1. Review and approve this plan
2. Create implementation issues/tasks
3. Begin with high-priority items
4. Test changes in staging environment
5. Deploy incrementally with monitoring
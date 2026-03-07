import type { Locale } from "../site-data";
import {
  getBlogEntries,
  getGenericSectionRouteBases,
  getLegacyBlogEntries,
  getTutorialDocSources
} from "./collections";
import {
  baseMarkdownPath,
  isSupportedSection,
  resolveLocalizedSource,
  sectionSourceToSlugSegments,
  tutorialSourceToSlugSegments
} from "./source";

export function docPathToRoute(relativePath: string, locale: Locale): string | null {
  const normalized = baseMarkdownPath(relativePath);
  const prefix = locale === "zh" ? "/zh" : "";

  if (normalized === "index.md") {
    return locale === "zh" ? "/zh/" : "/";
  }

  if (normalized === "blog/index.md") {
    return `${prefix}/blog/`;
  }

  if (normalized === "blogs/index.md") {
    return `${prefix}/blogs/`;
  }

  if (normalized.startsWith("blog/posts/")) {
    const key = normalized.split("/").at(-1)?.replace(/\.md$/, "");
    const blogEntry = getBlogEntries().find((entry) => entry.key === key);
    if (!blogEntry) {
      return null;
    }
    return `${prefix}/blog/${blogEntry.year}/${blogEntry.month}/${blogEntry.day}/${blogEntry.slug}/`;
  }

  if (normalized.startsWith("blogs/")) {
    const key = normalized.split("/").at(-1)?.replace(/\.md$/, "");
    return key ? `${prefix}/blogs/${key}/` : null;
  }

  const withoutExt = normalized.replace(/\.md$/, "");
  const segments = withoutExt.split("/");
  const head = segments.shift();

  if (!head) {
    return null;
  }

  if (head === "tutorials") {
    const trailing = segments.at(-1);
    const docSegments = trailing === "README" || trailing === "index" ? segments.slice(0, -1) : segments;
    return docSegments.length ? `${prefix}/tutorials/${docSegments.join("/")}/` : `${prefix}/tutorials/`;
  }

  if (!isSupportedSection(head)) {
    return null;
  }

  const trailing = segments.at(-1);
  const docSegments = trailing === "README" || trailing === "index" ? segments.slice(0, -1) : segments;
  return docSegments.length ? `${prefix}/${head}/${docSegments.join("/")}/` : `${prefix}/${head}/`;
}

export function listSitemapRoutes(): string[] {
  const routes = new Set<string>([
    "/",
    "/zh/",
    "/tutorials/",
    "/zh/tutorials/",
    "/blog/",
    "/zh/blog/",
    "/blogs/",
    "/zh/blogs/"
  ]);

  for (const sourceRelative of getTutorialDocSources()) {
    if (resolveLocalizedSource(sourceRelative, "en")) {
      routes.add(docPathToRoute(sourceRelative, "en") ?? "");
    }
    if (resolveLocalizedSource(sourceRelative, "zh")) {
      routes.add(docPathToRoute(sourceRelative, "zh") ?? "");
    }
  }

  for (const entry of getBlogEntries()) {
    routes.add(`/blog/${entry.year}/${entry.month}/${entry.day}/${entry.slug}/`);
    routes.add(`/zh/blog/${entry.year}/${entry.month}/${entry.day}/${entry.slug}/`);
  }

  for (const entry of getLegacyBlogEntries()) {
    routes.add(`/blogs/${entry.key}/`);
    routes.add(`/zh/blogs/${entry.key}/`);
  }

  for (const sourceRelative of getGenericSectionRouteBases()) {
    if (resolveLocalizedSource(sourceRelative, "en")) {
      routes.add(docPathToRoute(sourceRelative, "en") ?? "");
    }
    if (resolveLocalizedSource(sourceRelative, "zh")) {
      routes.add(docPathToRoute(sourceRelative, "zh") ?? "");
    }
  }

  return [...routes].filter(Boolean).sort();
}

export function getLegacyBlogRoutes(): string[][] {
  return getLegacyBlogEntries().map((entry) => [entry.key]);
}

export function getBlogRoutes(): string[][] {
  return getBlogEntries().map((entry) => [entry.year, entry.month, entry.day, entry.slug]);
}

export function getTutorialRoutes(locale: Locale): string[][] {
  return getTutorialDocSources()
    .filter((source) => Boolean(resolveLocalizedSource(source, locale)))
    .map((source) => tutorialSourceToSlugSegments(source));
}

export function getGenericSectionRoutes(locale: Locale): Array<{ section: string; slug: string[] }> {
  return getGenericSectionRouteBases()
    .filter((sourceRelative) => Boolean(resolveLocalizedSource(sourceRelative, locale)))
    .map((sourceRelative) => {
      const [section] = sourceRelative.split("/");
      return {
        section,
        slug: sectionSourceToSlugSegments(sourceRelative, section)
      };
    });
}

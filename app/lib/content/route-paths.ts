import type { Locale } from "../site-data";
import { localizePath } from "../paths";

export function buildHomePath(locale: Locale): string {
  return localizePath("/", locale);
}

export function buildTutorialIndexPath(locale: Locale): string {
  return localizePath("/tutorials/", locale);
}

export function buildTutorialPath(slugSegments: string[], locale: Locale): string {
  return localizePath(slugSegments.length ? `/tutorials/${slugSegments.join("/")}/` : "/tutorials/", locale);
}

export function buildBlogIndexPath(locale: Locale): string {
  return localizePath("/blog/", locale);
}

export function buildBlogPath(
  year: string,
  month: string,
  day: string,
  slug: string,
  locale: Locale
): string {
  return localizePath(`/blog/${year}/${month}/${day}/${slug}/`, locale);
}

export function buildLegacyBlogIndexPath(locale: Locale): string {
  return localizePath("/blogs/", locale);
}

export function buildLegacyBlogPath(slug: string, locale: Locale): string {
  return localizePath(`/blogs/${slug}/`, locale);
}

export function buildSectionPath(section: string, slugSegments: string[], locale: Locale): string {
  return localizePath(
    slugSegments.length ? `/${section}/${slugSegments.join("/")}/` : `/${section}/`,
    locale
  );
}

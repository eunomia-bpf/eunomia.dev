import type { Locale } from "../site-data";
import { getContentManifest, resolveRouteFromDocSource } from "./manifest";
import type { ContentManifestRecord } from "./types";

export { listSitemapRoutes } from "./manifest";

function filterManifest(
  predicate: (record: ContentManifestRecord) => boolean
): ContentManifestRecord[] {
  return getContentManifest().filter(predicate);
}

export function docPathToRoute(relativePath: string, locale: Locale): string | null {
  return resolveRouteFromDocSource(relativePath, locale);
}

export function getLegacyBlogRoutes(): string[][] {
  return filterManifest((record) => record.kind === "legacy-blog-page")
    .map((record) => record.slug)
    .filter((slug): slug is string[] => Array.isArray(slug));
}

export function getBlogRoutes(): string[][] {
  return filterManifest((record) => record.kind === "blog-page")
    .map((record) => record.slug)
    .filter((slug): slug is string[] => Array.isArray(slug));
}

export function getTutorialRoutes(locale: Locale): string[][] {
  return filterManifest((record) => record.kind === "tutorial-page" && Boolean(record.routeByLocale[locale]))
    .map((record) => record.slug)
    .filter((slug): slug is string[] => Array.isArray(slug));
}

export function getGenericSectionRoutes(locale: Locale): Array<{ section: string; slug: string[] }> {
  return filterManifest((record) => record.kind === "section-page" && Boolean(record.routeByLocale[locale]))
    .map((record) => ({
      section: record.section ?? "",
      slug: record.slug ?? []
    }))
    .filter((record) => Boolean(record.section));
}

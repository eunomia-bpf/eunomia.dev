import type { Locale } from "../site-data";
import { getDocument } from "./documents";
import { getContentModel } from "./model";
import type { BlogEntry, GenericSectionRouteEntry, LegacyBlogEntry } from "./types";

export function getTutorialReadmeSources(): string[] {
  return getContentModel().tutorialReadmeSources;
}

export function getTutorialDocSources(): string[] {
  return getContentModel().tutorialDocSources;
}

export function getBlogEntries(): BlogEntry[] {
  return getContentModel().blogEntries;
}

function resolveEntrySource(entry: BlogEntry, locale: Locale): string | null {
  return entry.sourceByLocale[locale] ?? entry.sourceByLocale.en ?? entry.sourceByLocale.zh ?? null;
}

export function getBlogEntriesForLocale(locale: Locale): BlogEntry[] {
  return getBlogEntries()
    .filter((entry) => Boolean(resolveEntrySource(entry, locale)))
    .map((entry) => {
      const source = resolveEntrySource(entry, locale);
      if (!source) {
        return entry;
      }

      const document = getDocument(source);
      return {
        ...entry,
        title: document.title,
        description: document.description,
        excerpt: document.excerpt
      };
    });
}

export function getRecentBlogEntriesForLocale(locale: Locale, limit = 3): BlogEntry[] {
  return getBlogEntriesForLocale(locale).slice(0, limit);
}

export function getLegacyBlogEntries(): LegacyBlogEntry[] {
  return getContentModel().legacyBlogEntries;
}

export function getGenericSectionRouteEntries(): GenericSectionRouteEntry[] {
  return getContentModel().genericSectionRoutes;
}

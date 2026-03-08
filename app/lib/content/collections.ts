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

export function getLegacyBlogEntries(): LegacyBlogEntry[] {
  return getContentModel().legacyBlogEntries;
}

export function getGenericSectionRouteEntries(): GenericSectionRouteEntry[] {
  return getContentModel().genericSectionRoutes;
}

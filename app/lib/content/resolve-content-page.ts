import { getSectionLabel } from "../site-ia";
import type { Locale } from "../site-data";
import { resolveManifestRecordFromRoute } from "./manifest";
import { getCollectionFamilyByKind } from "./registry";
import {
  loadBlogIndex,
  loadBlogPage,
  loadLegacyBlogIndex,
  loadLegacyBlogPage,
  loadTutorialIndex,
  loadTutorialPage
} from "./collection-loader";
import { loadSectionPage } from "./section-loader";
import type { DocsPage } from "./types";

export type ResolvedContentPage = {
  page: DocsPage;
  eyebrow: string;
};

const collectionResolvers = {
  tutorial: {
    loadIndex: loadTutorialIndex,
    loadPage: loadTutorialPage
  },
  blog: {
    loadIndex: loadBlogIndex,
    loadPage: loadBlogPage
  },
  "legacy-blog": {
    loadIndex: loadLegacyBlogIndex,
    loadPage: loadLegacyBlogPage
  }
};

export async function resolveContentPage(path: string, locale: Locale): Promise<ResolvedContentPage | null> {
  const record = resolveManifestRecordFromRoute(path);
  if (!record) {
    return null;
  }

  const family = getCollectionFamilyByKind(record.kind);
  if (family) {
    const loader = collectionResolvers[family.id];
    const page =
      record.kind === family.indexKind ? await loader.loadIndex(locale) : await loader.loadPage(record.slug, locale);

    if (!page) {
      return null;
    }

    return {
      eyebrow: family.eyebrow(locale),
      page
    };
  }

  if (record.kind !== "section-page") {
    return null;
  }

  const section = record.section ?? "";
  if (!section) {
    return null;
  }

  const page = await loadSectionPage(section, record.slug ?? [], locale);
  if (!page) {
    return null;
  }

  return {
    eyebrow: getSectionLabel(section, locale),
    page
  };
}

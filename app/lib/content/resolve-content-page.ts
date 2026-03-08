import { getSectionLabel } from "../site-ia";
import type { Locale } from "../site-data";
import { resolveManifestRecordFromRoute } from "./manifest";
import { getCollectionFamilyByKind } from "./registry";
import { loadCollectionIndex, loadCollectionPage } from "./collection-loader";
import { loadSectionPage } from "./section-loader";
import type { ContentManifestRecord, DocsPage } from "./types";

export type ResolvedContentPage = {
  page: DocsPage;
  eyebrow: string;
};

async function loadFamilyPage(record: ContentManifestRecord, locale: Locale) {
  const family = getCollectionFamilyByKind(record.kind);
  if (!family) {
    return null;
  }

  const page =
    record.kind === family.indexKind
      ? await loadCollectionIndex(family.id, locale)
      : await loadCollectionPage(family.id, record.slug, locale);

  if (!page) {
    return null;
  }

  return {
    eyebrow: family.eyebrow(locale),
    page
  };
}

export async function resolveContentPage(path: string, locale: Locale): Promise<ResolvedContentPage | null> {
  const record = resolveManifestRecordFromRoute(path);
  if (!record) {
    return null;
  }

  const familyPage = await loadFamilyPage(record, locale);
  if (familyPage) {
    return familyPage;
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

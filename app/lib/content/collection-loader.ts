import type { Locale } from "../site-data";
import { buildCollectionContinuation, buildIndexLink } from "./navigation";
import {
  getCollectionFamilyById,
  resolveCollectionPageSource,
  type CollectionFamilyDefinition,
  type CollectionFamilyId
} from "./registry";
import type { DocsPage } from "./types";
import { loadDirectoryPage, loadDocumentPage, requireDocument, withContinuation } from "./page-loader-utils";

async function loadCollectionIndexPage(
  family: CollectionFamilyDefinition,
  locale: Locale
): Promise<DocsPage | null> {
  const document = requireDocument(family.indexSource, locale);
  if (!document) {
    return null;
  }

  return loadDirectoryPage({
    sourceRelative: document.sourceRelative,
    publicPath: family.buildIndexPath(locale),
    locale,
    cards: family.buildIndexCards(locale),
    sidebar: family.buildSidebar(locale)
  });
}

async function loadCollectionArticlePage(
  family: CollectionFamilyDefinition,
  slugSegments: string[] | undefined,
  locale: Locale
): Promise<DocsPage | null> {
  if (!slugSegments?.length) {
    return null;
  }

  const sourceRelative = resolveCollectionPageSource(family.id, slugSegments, locale);
  if (!sourceRelative) {
    return null;
  }

  const publicPath = family.buildPagePath(slugSegments, locale);
  const page = await loadDocumentPage(sourceRelative, publicPath, locale);
  const continuation = buildCollectionContinuation({
    kind: family.pageKind,
    locale,
    currentPath: publicPath,
    index: buildIndexLink(requireDocument(family.indexSource, locale).sourceRelative, family.buildIndexPath(locale))
  });

  return {
    ...withContinuation(page, continuation),
    sidebar: family.buildSidebar(locale)
  };
}

export async function loadCollectionIndex(
  familyId: CollectionFamilyId,
  locale: Locale
): Promise<DocsPage> {
  const family = getCollectionFamilyById(familyId);
  if (!family) {
    throw new Error(`Unknown collection family: ${familyId}`);
  }

  const page = await loadCollectionIndexPage(family, locale);
  if (!page) {
    throw new Error(`Missing collection index page for ${familyId} (${locale})`);
  }

  return page;
}

export async function loadCollectionPage(
  familyId: CollectionFamilyId,
  slugSegments: string[] | undefined,
  locale: Locale
): Promise<DocsPage | null> {
  const family = getCollectionFamilyById(familyId);
  if (!family) {
    throw new Error(`Unknown collection family: ${familyId}`);
  }

  if (!slugSegments?.length) {
    return null;
  }

  return loadCollectionArticlePage(family, slugSegments, locale);
}

export function loadTutorialIndex(locale: Locale) {
  return loadCollectionIndex("tutorial", locale);
}

export function loadTutorialPage(slugSegments: string[] | undefined, locale: Locale) {
  return loadCollectionPage("tutorial", slugSegments, locale);
}

export function loadBlogIndex(locale: Locale) {
  return loadCollectionIndex("blog", locale);
}

export function loadBlogPage(slugSegments: string[] | undefined, locale: Locale) {
  return loadCollectionPage("blog", slugSegments, locale);
}

export function loadLegacyBlogIndex(locale: Locale) {
  return loadCollectionIndex("legacy-blog", locale);
}

export function loadLegacyBlogPage(slugSegments: string[] | undefined, locale: Locale) {
  return loadCollectionPage("legacy-blog", slugSegments, locale);
}

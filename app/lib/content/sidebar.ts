import { localizePath } from "../paths";
import { getSectionSidebarOverride } from "../site-ia";
import type { Locale } from "../site-data";
import { getTutorialDocSources } from "./collections";
import { getDocument, resolveDocument } from "./documents";
import { getContentManifest } from "./manifest";
import { getCollectionFamilyById, getCollectionPageDescriptors, type CollectionFamilyId } from "./registry";
import { resolveRecordHref, resolveRecordSource } from "./record-utils";
import { resolveLocalizedSource } from "./source";
import type { ContentManifestRecord, SidebarGroup, SidebarItem } from "./types";

function getSidebarCopy(locale: Locale) {
  return locale === "zh"
    ? {
        blog: "博客",
        sectionPrefix: "文档"
      }
    : {
        blog: "Blog",
        sectionPrefix: "Docs"
      };
}

function recordToSidebarItem(record: ContentManifestRecord, locale: Locale): SidebarItem | null {
  const source = resolveRecordSource(record, locale);
  const href = resolveRecordHref(record, locale);
  if (!source || !href) {
    return null;
  }

  return {
    title: getDocument(source).title,
    href,
    depth: record.slug?.length ?? 0
  };
}

function descriptorToSidebarItem(
  familyId: CollectionFamilyId,
  locale: Locale
) {
  return function toSidebarItem(descriptor: ReturnType<typeof getCollectionPageDescriptors>[number]): SidebarItem | null {
    const source =
      descriptor.sourceByLocale[locale] ?? descriptor.sourceByLocale.en ?? descriptor.sourceByLocale.zh ?? null;
    if (!source) {
      return null;
    }

    const document = resolveDocument(source, locale);
    if (!document) {
      return null;
    }

    return {
      title: document.title,
      href: descriptor.buildPath(locale),
      depth: familyId === "tutorial" ? descriptor.slug.length : 0
    };
  };
}

export function buildCollectionSidebar(
  familyId: CollectionFamilyId,
  locale: Locale
): SidebarGroup[] {
  const family = getCollectionFamilyById(familyId);
  if (!family) {
    throw new Error(`Unknown collection family: ${familyId}`);
  }

  const items = getCollectionPageDescriptors(familyId)
    .map(descriptorToSidebarItem(familyId, locale))
    .filter((item): item is SidebarItem => Boolean(item));

  const indexTitle = resolveDocument(family.indexSource, locale)?.title ?? family.eyebrow(locale);

  return [
    {
      title: family.eyebrow(locale),
      items: [
        {
          title: indexTitle,
          href: family.buildIndexPath(locale),
          depth: 0
        },
        ...items
      ]
    }
  ];
}

export function buildSectionSidebar(section: string, locale: Locale): SidebarGroup[] {
  const configuredSidebar = getSectionSidebarOverride(section, locale);
  if (configuredSidebar) {
    return configuredSidebar;
  }

  const copy = getSidebarCopy(locale);
  const records = getContentManifest().filter((record) => record.kind === "section-page" && record.section === section);
  const sectionIndexSource =
    resolveLocalizedSource(`${section}/index.md`, locale) ?? resolveLocalizedSource(`${section}/README.md`, locale);
  const sectionTitle = sectionIndexSource
    ? getDocument(sectionIndexSource).title
    : `${copy.sectionPrefix}: ${section}`;

  return [
    {
      title: sectionTitle,
      items: records
        .map((record) => recordToSidebarItem(record, locale))
        .filter((item): item is SidebarItem => Boolean(item))
    }
  ];
}

export function buildSearchSidebar(locale: Locale): SidebarGroup[] {
  const copy = getSidebarCopy(locale);
  const tutorialCount = getTutorialDocSources().length;

  return [
    {
      title: locale === "zh" ? "浏览" : "Explore",
      items: [
        {
          title: locale === "zh" ? `教程 (${tutorialCount})` : `Tutorials (${tutorialCount})`,
          href: localizePath("/tutorials/", locale)
        },
        {
          title: copy.blog,
          href: localizePath("/blog/", locale)
        }
      ]
    }
  ];
}

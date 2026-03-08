import { localizePath } from "../paths";
import { getPrimaryNav } from "../site-ia";
import type { Locale } from "../site-data";
import { getTutorialDocSources } from "./collections";
import { getDocument, resolveDocument } from "./documents";
import { getContentManifest } from "./manifest";
import { resolveLocalizedSource } from "./source";
import type { ContentManifestRecord, SidebarGroup, SidebarItem } from "./types";

function getSidebarCopy(locale: Locale) {
  return locale === "zh"
    ? {
        browse: "导航",
        tutorials: "教程",
        blog: "博客",
        legacyBlog: "旧博客",
        sectionPrefix: "文档"
      }
    : {
        browse: "Browse",
        tutorials: "Tutorials",
        blog: "Blog",
        legacyBlog: "Legacy Blog",
        sectionPrefix: "Docs"
      };
}

function buildPrimaryGroup(locale: Locale): SidebarGroup {
  const copy = getSidebarCopy(locale);

  return {
    title: copy.browse,
    items: [
      ...getPrimaryNav(locale).map((item) => ({
        title: item.label,
        href: item.href
      })),
      {
        title: copy.legacyBlog,
        href: localizePath("/blogs/", locale)
      }
    ]
  };
}

function resolveRecordSource(record: ContentManifestRecord, locale: Locale): string | null {
  return record.sourceByLocale[locale] ?? record.sourceByLocale.en ?? record.sourceByLocale.zh ?? null;
}

function resolveRecordHref(record: ContentManifestRecord, locale: Locale): string | null {
  return record.routeByLocale[locale] ?? record.routeByLocale.en ?? record.routeByLocale.zh ?? null;
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

export function buildSectionSidebar(section: string, locale: Locale): SidebarGroup[] {
  const copy = getSidebarCopy(locale);
  const records = getContentManifest().filter((record) => record.kind === "section-page" && record.section === section);
  const sectionIndexSource =
    resolveLocalizedSource(`${section}/index.md`, locale) ?? resolveLocalizedSource(`${section}/README.md`, locale);
  const sectionTitle = sectionIndexSource
    ? getDocument(sectionIndexSource).title
    : `${copy.sectionPrefix}: ${section}`;

  return [
    buildPrimaryGroup(locale),
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
    buildPrimaryGroup(locale),
    {
      title: copy.browse,
      items: [
        {
          title: locale === "zh" ? `教程 (${tutorialCount})` : `Tutorials (${tutorialCount})`,
          href: localizePath("/tutorials/", locale)
        },
        {
          title: copy.blog,
          href: localizePath("/blog/", locale)
        },
        {
          title: copy.legacyBlog,
          href: localizePath("/blogs/", locale)
        }
      ]
    }
  ];
}

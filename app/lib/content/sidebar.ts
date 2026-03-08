import { localizePath } from "../paths";
import { navByLocale, type Locale } from "../site-data";
import { getBlogEntries, getLegacyBlogEntries, getTutorialDocSources } from "./collections";
import { getContentManifest } from "./manifest";
import { parseMarkdown } from "./markdown";
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
      ...navByLocale[locale].map((item) => ({
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
    title: parseMarkdown(source).title,
    href,
    depth: record.slug?.length ?? 0
  };
}

export function buildTutorialSidebar(locale: Locale): SidebarGroup[] {
  const copy = getSidebarCopy(locale);
  const tutorialItems = getContentManifest()
    .filter((record) => record.kind === "tutorial-page")
    .map((record) => recordToSidebarItem(record, locale))
    .filter((item): item is SidebarItem => Boolean(item));

  return [
    buildPrimaryGroup(locale),
    {
      title: copy.tutorials,
      items: [
        {
          title: parseMarkdown(resolveLocalizedSource("tutorials/index.md", locale) ?? "tutorials/index.md").title,
          href: localizePath("/tutorials/", locale),
          depth: 0
        },
        ...tutorialItems
      ]
    }
  ];
}

export function buildBlogSidebar(locale: Locale): SidebarGroup[] {
  const copy = getSidebarCopy(locale);

  return [
    buildPrimaryGroup(locale),
    {
      title: copy.blog,
      items: [
        {
          title: parseMarkdown(resolveLocalizedSource("blog/index.md", locale) ?? "blog/index.md").title,
          href: localizePath("/blog/", locale)
        },
        ...getBlogEntries().map((entry) => ({
          title: entry.title,
          href: localizePath(`/blog/${entry.year}/${entry.month}/${entry.day}/${entry.slug}/`, locale)
        }))
      ]
    }
  ];
}

export function buildLegacyBlogSidebar(locale: Locale): SidebarGroup[] {
  const copy = getSidebarCopy(locale);

  return [
    buildPrimaryGroup(locale),
    {
      title: copy.legacyBlog,
      items: [
        {
          title: parseMarkdown(resolveLocalizedSource("blogs/index.md", locale) ?? "blogs/index.md").title,
          href: localizePath("/blogs/", locale)
        },
        ...getLegacyBlogEntries().map((entry) => ({
          title: entry.title,
          href: localizePath(`/blogs/${entry.key}/`, locale)
        }))
      ]
    }
  ];
}

export function buildSectionSidebar(section: string, locale: Locale): SidebarGroup[] {
  const copy = getSidebarCopy(locale);
  const records = getContentManifest().filter((record) => record.kind === "section-page" && record.section === section);
  const sectionIndexSource =
    resolveLocalizedSource(`${section}/index.md`, locale) ?? resolveLocalizedSource(`${section}/README.md`, locale);
  const sectionTitle = sectionIndexSource
    ? parseMarkdown(sectionIndexSource).title
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

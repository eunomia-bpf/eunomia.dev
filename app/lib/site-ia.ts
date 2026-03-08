import siteSectionPayload from "../.generated/content/site-sections.json";

import { localizePath } from "./paths";
import type { Locale } from "./site-data";
import type {
  SerializedPublishedSiteSection,
  SerializedSiteSectionDefinition,
  SiteSectionKey
} from "./site-ia-source";

export type SiteSectionDefinition = SerializedSiteSectionDefinition & SerializedPublishedSiteSection & {
  href: (locale: Locale) => string;
};

type HomeCounts = {
  tutorials: number;
  blogs: number;
  legacyBlogs: number;
};

const sectionDefinitions: SiteSectionDefinition[] = siteSectionPayload.sections.map((section) => ({
  ...section,
  ...section.published,
  href: (locale) => section.hrefByLocale[locale]
}));

function getSectionDefinition(key: SiteSectionKey): SiteSectionDefinition {
  const section = sectionDefinitions.find((candidate) => candidate.key === key);
  if (!section) {
    throw new Error(`Unknown site IA section: ${key}`);
  }
  return section;
}

export function getSiteSections(): SiteSectionDefinition[] {
  return sectionDefinitions;
}

export function getSiteSection(key: SiteSectionKey): SiteSectionDefinition {
  return getSectionDefinition(key);
}

export function getPrimaryNav(locale: Locale): Array<{ label: string; href: string }> {
  return sectionDefinitions
    .filter((section) => section.nav)
    .map((section) => ({
      label: section.labels[locale],
      href: section.href(locale)
    }));
}

export function getHomeTrackSections(): SiteSectionDefinition[] {
  return sectionDefinitions.filter((section) => section.homeTrack);
}

export function getHomeExploreSections(): SiteSectionDefinition[] {
  return sectionDefinitions.filter((section) => section.homeExplore);
}

export function getFooterExploreSections(): SiteSectionDefinition[] {
  return sectionDefinitions.filter((section) => section.footerExplore);
}

export function getFooterProjectSections(): SiteSectionDefinition[] {
  return sectionDefinitions.filter((section) => section.footerProject);
}

export function getSectionLabel(section: string, locale: Locale): string {
  const match = sectionDefinitions.find((candidate) => candidate.key === section);
  return match?.labels[locale] ?? section;
}

export function getHomePath(locale: Locale): string {
  return localizePath("/", locale);
}

export function getFeaturedHomeSections(): Array<{
  key: SiteSectionKey;
  source: string;
  href: (locale: Locale) => string;
}> {
  return getHomeTrackSections().map((section) => ({
    key: section.key,
    source: section.indexSource,
    href: section.href
  }));
}

export function getHomeBadge(
  key: SiteSectionKey,
  counts: HomeCounts,
  locale: Locale
): string | undefined {
  if (key === "tutorials") {
    return locale === "zh" ? `${counts.tutorials} 篇教程` : `${counts.tutorials} walkthroughs`;
  }

  if (key === "bpftime") {
    return "Userspace eBPF";
  }

  if (key === "eunomia-bpf") {
    return locale === "zh" ? "工具链" : "Toolchain";
  }

  return undefined;
}

export function getHomeStats(locale: Locale, counts: HomeCounts) {
  if (locale === "zh") {
    return [
      {
        label: "教程",
        value: String(counts.tutorials),
        detail: "从基础 trace 到更深的 kernel/runtime 主题。"
      },
      {
        label: "研究文章",
        value: String(counts.blogs),
        detail: "持续整理 eBPF、GPU、AI agent 和系统研究。"
      },
      {
        label: "旧博客",
        value: String(counts.legacyBlogs),
        detail: "旧 `/blogs/*` 路径仍然保留并可访问。"
      }
    ];
  }

  return [
    {
      label: "Tutorials",
      value: String(counts.tutorials),
      detail: "Hands-on walkthroughs from basic tracing to deeper kernel and runtime topics."
    },
    {
      label: "Research posts",
      value: String(counts.blogs),
      detail: "Ongoing writing on eBPF, GPU tooling, AI agents, and systems research."
    },
    {
      label: "Legacy posts",
      value: String(counts.legacyBlogs),
      detail: "The old `/blogs/*` paths remain live while the app cuts over."
    }
  ];
}

export type { SiteSectionKey } from "./site-ia-source";

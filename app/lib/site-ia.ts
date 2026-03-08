import type { Locale } from "./site-data";
import { buildBlogIndexPath, buildHomePath, buildLegacyBlogIndexPath, buildSectionPath, buildTutorialIndexPath } from "./content/route-paths";

export type SiteSectionKey =
  | "tutorials"
  | "blog"
  | "legacy-blog"
  | "bpftime"
  | "GPTtrace"
  | "eunomia-bpf"
  | "others"
  | "wasm-bpf";

export type SiteSectionDefinition = {
  key: SiteSectionKey;
  labels: Record<Locale, string>;
  indexSource: string;
  href: (locale: Locale) => string;
  nav?: boolean;
  homeTrack?: boolean;
  homeExplore?: boolean;
  footerExplore?: boolean;
  footerProject?: boolean;
};

type HomeCounts = {
  tutorials: number;
  blogs: number;
  legacyBlogs: number;
};

const sectionDefinitions: SiteSectionDefinition[] = [
  {
    key: "tutorials",
    labels: { en: "Tutorials", zh: "教程" },
    indexSource: "tutorials/index.md",
    href: buildTutorialIndexPath,
    nav: true,
    homeTrack: true,
    footerExplore: true
  },
  {
    key: "blog",
    labels: { en: "Blog", zh: "博客" },
    indexSource: "blog/index.md",
    href: buildBlogIndexPath,
    nav: true,
    footerExplore: true
  },
  {
    key: "legacy-blog",
    labels: { en: "Legacy blog", zh: "旧博客" },
    indexSource: "blogs/index.md",
    href: buildLegacyBlogIndexPath,
    homeExplore: true,
    footerExplore: true
  },
  {
    key: "bpftime",
    labels: { en: "bpftime", zh: "bpftime" },
    indexSource: "bpftime/index.md",
    href: (locale) => buildSectionPath("bpftime", [], locale),
    nav: true,
    homeTrack: true,
    footerExplore: true
  },
  {
    key: "GPTtrace",
    labels: { en: "eBPF×AI/LLMs", zh: "eBPF×AI/LLMs" },
    indexSource: "GPTtrace/index.md",
    href: (locale) => buildSectionPath("GPTtrace", [], locale),
    nav: true,
    homeExplore: true,
    footerProject: true
  },
  {
    key: "eunomia-bpf",
    labels: { en: "eunomia-bpf", zh: "eunomia-bpf" },
    indexSource: "eunomia-bpf/index.md",
    href: (locale) => buildSectionPath("eunomia-bpf", [], locale),
    nav: true,
    homeTrack: true,
    footerExplore: true
  },
  {
    key: "others",
    labels: { en: "Ecosystem", zh: "生态" },
    indexSource: "others/index.md",
    href: (locale) => buildSectionPath("others", [], locale),
    nav: true,
    homeExplore: true,
    footerExplore: true
  },
  {
    key: "wasm-bpf",
    labels: { en: "wasm-bpf", zh: "wasm-bpf" },
    indexSource: "wasm-bpf/index.md",
    href: (locale) => buildSectionPath("wasm-bpf", [], locale),
    homeExplore: true,
    footerProject: true
  }
];

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
  return buildHomePath(locale);
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

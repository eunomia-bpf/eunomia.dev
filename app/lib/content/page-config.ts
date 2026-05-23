import fs from "node:fs";
import path from "node:path";

import matter from "gray-matter";

import type { Locale } from "../site-data";
import { useContentCache } from "./cache";
import { contentRoot } from "./roots";

export type LocalizedText = Record<Locale, string>;

export type PageLinkConfig = {
  label: LocalizedText;
  href: string;
};

export type ProjectCardConfig = {
  key: string;
  title: string;
  tag: LocalizedText;
  href: string;
  image: string;
  imageAlt: string;
  description: LocalizedText;
  links: PageLinkConfig[];
};

export type ProjectGroupConfig = {
  key: string;
  title: LocalizedText;
  intro: LocalizedText;
  projectKeys: string[];
};

export type ProjectCatalogConfig = {
  projectGroups: ProjectGroupConfig[];
  projects: ProjectCardConfig[];
};

export type ProjectsLandingConfig = ProjectCatalogConfig & {
  variant: "project-index";
  title: LocalizedText;
  description: LocalizedText;
  summary: LocalizedText;
  sectionLabels: {
    featured: LocalizedText;
    groups: LocalizedText;
    groupCount: LocalizedText;
    entryCount: LocalizedText;
  };
  featuredProjectKeys: string[];
};

export type HomeProofPointConfig = {
  value: string;
  label: LocalizedText;
};

export type HomeHeroConfig = {
  kicker: LocalizedText;
  title: LocalizedText;
  summary: LocalizedText;
  primaryCta: LocalizedText;
  primaryHref: string;
  secondaryCta: LocalizedText;
  secondaryHref: string;
  proofPoints: HomeProofPointConfig[];
};

export type HomeCapabilityConfig = {
  key: string;
  eyebrow: LocalizedText;
  title: LocalizedText;
  description: LocalizedText;
};

export type HomePageConfig = {
  hero: HomeHeroConfig;
  capabilitiesTitle: LocalizedText;
  capabilitiesIntro: LocalizedText;
  capabilities: HomeCapabilityConfig[];
  projectsTitle: LocalizedText;
  projectsIntro: LocalizedText;
  projectGroupKeys: string[];
  latestTitle: LocalizedText;
  allPostsLabel: LocalizedText;
};

export type AiDirectionConfig = {
  key: string;
  eyebrow: LocalizedText;
  title: LocalizedText;
  description: LocalizedText;
  points: Array<{ label: LocalizedText }>;
};

export type AiUseCaseItemConfig = {
  title: LocalizedText;
  description: LocalizedText;
  href?: string;
};

export type AiUseCaseGroupConfig = {
  key: string;
  eyebrow: LocalizedText;
  title: LocalizedText;
  description: LocalizedText;
  items: AiUseCaseItemConfig[];
};

export type AiEbpfLandingConfig = {
  variant: "ai-ebpf";
  title: LocalizedText;
  description: LocalizedText;
  summary: LocalizedText;
  heroImage: string;
  sectionLabels: {
    directions: LocalizedText;
    projects: LocalizedText;
    useCases: LocalizedText;
    references: LocalizedText;
  };
  directions: AiDirectionConfig[];
  featuredProjects: ProjectCardConfig[];
  useCaseGroups: AiUseCaseGroupConfig[];
  references: PageLinkConfig[];
};

export type BlogLandingConfig = {
  variant: "blog-index";
  title: LocalizedText;
  description: LocalizedText;
  sectionLabels: {
    featured: LocalizedText;
    archive: LocalizedText;
    empty: LocalizedText;
  };
};

export type PageLandingConfig = ProjectsLandingConfig | AiEbpfLandingConfig | BlogLandingConfig;

let homePageConfigCache: HomePageConfig | null = null;
let projectsPageConfigCache: ProjectsLandingConfig | null = null;
let aiEbpfPageConfigCache: AiEbpfLandingConfig | null = null;
let blogPageConfigCache: BlogLandingConfig | null = null;

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function requireRecord(value: unknown, context: string): Record<string, unknown> {
  if (!isRecord(value)) {
    throw new Error(`Missing page config object: ${context}`);
  }

  return value;
}

function requireString(value: unknown, context: string): string {
  if (typeof value !== "string" || !value.trim()) {
    throw new Error(`Missing page config string: ${context}`);
  }

  return value.trim();
}

function optionalString(value: unknown, context: string): string | undefined {
  if (value === undefined || value === null) {
    return undefined;
  }

  return requireString(value, context);
}

function requireArray(value: unknown, context: string): unknown[] {
  if (!Array.isArray(value)) {
    throw new Error(`Missing page config list: ${context}`);
  }

  return value;
}

function requireStringList(value: unknown, context: string): string[] {
  const list = requireArray(value, context).map((item, index) =>
    requireString(item, `${context}[${index}]`)
  );

  if (!list.length) {
    throw new Error(`Missing page config list entries: ${context}`);
  }

  return list;
}

function requireLocalizedText(value: unknown, context: string): LocalizedText {
  const record = requireRecord(value, context);

  return {
    en: requireString(record.en, `${context}.en`),
    zh: requireString(record.zh, `${context}.zh`)
  };
}

function readYamlPage(filename: string): Record<string, unknown> {
  const source = fs.readFileSync(path.join(contentRoot, "pages", filename), "utf8");
  const parsed = matter(`---\n${source}\n---\n`);

  return requireRecord(parsed.data, filename);
}

function normalizeLinks(value: unknown, context: string): PageLinkConfig[] {
  return requireArray(value ?? [], context).map((item, index) => {
    const record = requireRecord(item, `${context}[${index}]`);
    return {
      label: requireLocalizedText(record.label, `${context}[${index}].label`),
      href: requireString(record.href, `${context}[${index}].href`)
    };
  });
}

function normalizeProjectCard(value: unknown, context: string): ProjectCardConfig {
  const record = requireRecord(value, context);
  const key = requireString(record.key, `${context}.key`);

  return {
    key,
    title: requireString(record.title, `${context}.title`),
    tag: requireLocalizedText(record.tag, `${context}.tag`),
    href: requireString(record.href, `${context}.href`),
    image: requireString(record.image, `${context}.image`),
    imageAlt: requireString(record.image_alt, `${context}.image_alt`),
    description: requireLocalizedText(record.description, `${context}.description`),
    links: normalizeLinks(record.links, `${context}.links`)
  };
}

function normalizeProjectGroup(value: unknown, context: string): ProjectGroupConfig {
  const record = requireRecord(value, context);
  const key = requireString(record.key, `${context}.key`);

  return {
    key,
    title: requireLocalizedText(record.title, `${context}.title`),
    intro: requireLocalizedText(record.intro, `${context}.intro`),
    projectKeys: requireStringList(record.project_keys, `${context}.project_keys`)
  };
}

function normalizeProjectCatalog(record: Record<string, unknown>, context: string): ProjectCatalogConfig {
  return {
    projectGroups: requireArray(record.project_groups, `${context}.project_groups`).map((item, index) =>
      normalizeProjectGroup(item, `${context}.project_groups[${index}]`)
    ),
    projects: requireArray(record.projects, `${context}.projects`).map((item, index) =>
      normalizeProjectCard(item, `${context}.projects[${index}]`)
    )
  };
}

function normalizeProjectsPageConfig(record: Record<string, unknown>): ProjectsLandingConfig {
  const variant = requireString(record.variant, "projects.variant");
  if (variant !== "project-index") {
    throw new Error(`Unsupported projects.variant: ${variant}`);
  }
  const sectionLabels = requireRecord(record.section_labels, "projects.section_labels");

  return {
    variant,
    title: requireLocalizedText(record.title, "projects.title"),
    description: requireLocalizedText(record.description, "projects.description"),
    summary: requireLocalizedText(record.summary, "projects.summary"),
    sectionLabels: {
      featured: requireLocalizedText(sectionLabels.featured, "projects.section_labels.featured"),
      groups: requireLocalizedText(sectionLabels.groups, "projects.section_labels.groups"),
      groupCount: requireLocalizedText(sectionLabels.group_count, "projects.section_labels.group_count"),
      entryCount: requireLocalizedText(sectionLabels.entry_count, "projects.section_labels.entry_count")
    },
    featuredProjectKeys: requireStringList(record.featured_projects, "projects.featured_projects"),
    ...normalizeProjectCatalog(record, "projects")
  };
}

function normalizeHomePageConfig(record: Record<string, unknown>): HomePageConfig {
  const hero = requireRecord(record.hero, "home.hero");

  return {
    hero: {
      kicker: requireLocalizedText(hero.kicker, "home.hero.kicker"),
      title: requireLocalizedText(hero.title, "home.hero.title"),
      summary: requireLocalizedText(hero.summary, "home.hero.summary"),
      primaryCta: requireLocalizedText(hero.primary_cta, "home.hero.primary_cta"),
      primaryHref: requireString(hero.primary_href, "home.hero.primary_href"),
      secondaryCta: requireLocalizedText(hero.secondary_cta, "home.hero.secondary_cta"),
      secondaryHref: requireString(hero.secondary_href, "home.hero.secondary_href"),
      proofPoints: requireArray(hero.proof_points, "home.hero.proof_points").map((item, index) => {
        const proofPoint = requireRecord(item, `home.hero.proof_points[${index}]`);
        return {
          value: requireString(proofPoint.value, `home.hero.proof_points[${index}].value`),
          label: requireLocalizedText(proofPoint.label, `home.hero.proof_points[${index}].label`)
        };
      })
    },
    capabilitiesTitle: requireLocalizedText(record.capabilities_title, "home.capabilities_title"),
    capabilitiesIntro: requireLocalizedText(record.capabilities_intro, "home.capabilities_intro"),
    capabilities: requireArray(record.capabilities, "home.capabilities").map((item, index) => {
      const capability = requireRecord(item, `home.capabilities[${index}]`);
      const key = requireString(capability.key, `home.capabilities[${index}].key`);
      return {
        key,
        eyebrow: requireLocalizedText(capability.eyebrow, `home.capabilities.${key}.eyebrow`),
        title: requireLocalizedText(capability.title, `home.capabilities.${key}.title`),
        description: requireLocalizedText(capability.description, `home.capabilities.${key}.description`)
      };
    }),
    projectsTitle: requireLocalizedText(record.projects_title, "home.projects_title"),
    projectsIntro: requireLocalizedText(record.projects_intro, "home.projects_intro"),
    projectGroupKeys: requireStringList(record.project_group_keys, "home.project_group_keys"),
    latestTitle: requireLocalizedText(record.latest_title, "home.latest_title"),
    allPostsLabel: requireLocalizedText(record.all_posts_label, "home.all_posts_label")
  };
}

function normalizeAiDirection(value: unknown, context: string): AiDirectionConfig {
  const record = requireRecord(value, context);
  const key = requireString(record.key, `${context}.key`);

  return {
    key,
    eyebrow: requireLocalizedText(record.eyebrow, `${context}.${key}.eyebrow`),
    title: requireLocalizedText(record.title, `${context}.${key}.title`),
    description: requireLocalizedText(record.description, `${context}.${key}.description`),
    points: requireArray(record.points, `${context}.${key}.points`).map((item, index) => {
      const point = requireRecord(item, `${context}.${key}.points[${index}]`);
      return {
        label: requireLocalizedText(point.label, `${context}.${key}.points[${index}].label`)
      };
    })
  };
}

function normalizeAiUseCaseGroup(value: unknown, context: string): AiUseCaseGroupConfig {
  const record = requireRecord(value, context);
  const key = requireString(record.key, `${context}.key`);

  return {
    key,
    eyebrow: requireLocalizedText(record.eyebrow, `${context}.${key}.eyebrow`),
    title: requireLocalizedText(record.title, `${context}.${key}.title`),
    description: requireLocalizedText(record.description, `${context}.${key}.description`),
    items: requireArray(record.items, `${context}.${key}.items`).map((item, index) => {
      const useCase = requireRecord(item, `${context}.${key}.items[${index}]`);
      const href = optionalString(useCase.href, `${context}.${key}.items[${index}].href`);
      const normalized = {
        title: requireLocalizedText(useCase.title, `${context}.${key}.items[${index}].title`),
        description: requireLocalizedText(useCase.description, `${context}.${key}.items[${index}].description`)
      };

      if (!href) {
        return normalized;
      }

      return {
        ...normalized,
        href
      };
    })
  };
}

function normalizeAiEbpfPageConfig(record: Record<string, unknown>): AiEbpfLandingConfig {
  const variant = requireString(record.variant, "ai-ebpf.variant");
  if (variant !== "ai-ebpf") {
    throw new Error(`Unsupported ai-ebpf.variant: ${variant}`);
  }
  const sectionLabels = requireRecord(record.section_labels, "ai-ebpf.section_labels");

  return {
    variant,
    title: requireLocalizedText(record.title, "ai-ebpf.title"),
    description: requireLocalizedText(record.description, "ai-ebpf.description"),
    summary: requireLocalizedText(record.summary, "ai-ebpf.summary"),
    heroImage: requireString(record.hero_image, "ai-ebpf.hero_image"),
    sectionLabels: {
      directions: requireLocalizedText(sectionLabels.directions, "ai-ebpf.section_labels.directions"),
      projects: requireLocalizedText(sectionLabels.projects, "ai-ebpf.section_labels.projects"),
      useCases: requireLocalizedText(sectionLabels.use_cases, "ai-ebpf.section_labels.use_cases"),
      references: requireLocalizedText(sectionLabels.references, "ai-ebpf.section_labels.references")
    },
    directions: requireArray(record.directions, "ai-ebpf.directions").map((item, index) =>
      normalizeAiDirection(item, `ai-ebpf.directions[${index}]`)
    ),
    featuredProjects: requireArray(record.featured_projects, "ai-ebpf.featured_projects").map((item, index) =>
      normalizeProjectCard(item, `ai-ebpf.featured_projects[${index}]`)
    ),
    useCaseGroups: requireArray(record.use_case_groups, "ai-ebpf.use_case_groups").map((item, index) =>
      normalizeAiUseCaseGroup(item, `ai-ebpf.use_case_groups[${index}]`)
    ),
    references: normalizeLinks(record.references, "ai-ebpf.references")
  };
}

function normalizeBlogPageConfig(record: Record<string, unknown>): BlogLandingConfig {
  const variant = requireString(record.variant, "blog.variant");
  if (variant !== "blog-index") {
    throw new Error(`Unsupported blog.variant: ${variant}`);
  }
  const sectionLabels = requireRecord(record.section_labels, "blog.section_labels");

  return {
    variant,
    title: requireLocalizedText(record.title, "blog.title"),
    description: requireLocalizedText(record.description, "blog.description"),
    sectionLabels: {
      featured: requireLocalizedText(sectionLabels.featured, "blog.section_labels.featured"),
      archive: requireLocalizedText(sectionLabels.archive, "blog.section_labels.archive"),
      empty: requireLocalizedText(sectionLabels.empty, "blog.section_labels.empty")
    }
  };
}

export function readHomePageConfig(): HomePageConfig {
  if (!useContentCache || !homePageConfigCache) {
    homePageConfigCache = normalizeHomePageConfig(readYamlPage("home.yaml"));
  }

  return homePageConfigCache;
}

export function readProjectsPageConfig(): ProjectsLandingConfig {
  if (!useContentCache || !projectsPageConfigCache) {
    projectsPageConfigCache = normalizeProjectsPageConfig(readYamlPage("projects.yaml"));
  }

  return projectsPageConfigCache;
}

export function readAiEbpfPageConfig(): AiEbpfLandingConfig {
  if (!useContentCache || !aiEbpfPageConfigCache) {
    aiEbpfPageConfigCache = normalizeAiEbpfPageConfig(readYamlPage("ai-ebpf.yaml"));
  }

  return aiEbpfPageConfigCache;
}

export function readBlogLandingConfig(): BlogLandingConfig {
  if (!useContentCache || !blogPageConfigCache) {
    blogPageConfigCache = normalizeBlogPageConfig(readYamlPage("blog.yaml"));
  }

  return blogPageConfigCache;
}

export function readSectionLandingPage(section: string): PageLandingConfig | null {
  if (section === "projects") {
    return readProjectsPageConfig();
  }
  if (section === "GPTtrace") {
    return readAiEbpfPageConfig();
  }

  return null;
}

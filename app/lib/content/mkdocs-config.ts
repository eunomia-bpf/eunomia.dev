import fs from "node:fs";
import path from "node:path";

import { useContentCache } from "./cache";
import { appRoot, mkdocsConfigPath } from "./roots";

export type MkdocsSiteMetadata = {
  siteName: string;
  repoUrl: string;
  siteUrl: string;
  copyright: string;
  siteDescription: string;
  remoteBranch: string;
  editUri: string;
};

export type MkdocsTopLevelNavSection = {
  key: string;
  label: string;
  order: number;
  source?: string;
};

export type MkdocsSiteSectionConfig = {
  labels?: Partial<Record<"en" | "zh", string>>;
  published?: Partial<{
    nav: boolean;
    homeTrack: boolean;
    homeExplore: boolean;
    footerExplore: boolean;
    footerProject: boolean;
  }>;
  order?: number;
};

export type MkdocsLocalizedText = Record<"en" | "zh", string>;

export type MkdocsSectionPageConfig = {
  source: string;
  title: MkdocsLocalizedText;
  description: MkdocsLocalizedText;
  body: MkdocsLocalizedText;
};

export type MkdocsHomeProjectLink = {
  label: MkdocsLocalizedText;
  href: string;
};

export type MkdocsHomeProject = {
  key: string;
  title: string;
  tag: MkdocsLocalizedText;
  href: string;
  image: string;
  imageAlt: string;
  description: MkdocsLocalizedText;
  links: MkdocsHomeProjectLink[];
};

export type MkdocsHomeHeroConfig = {
  kicker: MkdocsLocalizedText;
  title: MkdocsLocalizedText;
  summary: MkdocsLocalizedText;
  primaryCta: MkdocsLocalizedText;
  primaryHref: string;
  secondaryCta: MkdocsLocalizedText;
  secondaryHref: string;
};

export type MkdocsHomeCapability = {
  key: string;
  eyebrow: MkdocsLocalizedText;
  title: MkdocsLocalizedText;
  description: MkdocsLocalizedText;
};

export type MkdocsHomeProjectGroup = {
  key: string;
  title: MkdocsLocalizedText;
  intro: MkdocsLocalizedText;
  projectKeys: string[];
};

export type MkdocsHomeConfig = {
  hero: MkdocsHomeHeroConfig;
  capabilitiesTitle: MkdocsLocalizedText;
  capabilitiesIntro: MkdocsLocalizedText;
  capabilities: MkdocsHomeCapability[];
  projectGroups: MkdocsHomeProjectGroup[];
  projectsTitle: MkdocsLocalizedText;
  projectsIntro: MkdocsLocalizedText;
  projects: MkdocsHomeProject[];
  latestTitle: MkdocsLocalizedText;
  allPostsLabel: MkdocsLocalizedText;
};

const metadataKeyMap = {
  site_name: "siteName",
  repo_url: "repoUrl",
  site_url: "siteUrl",
  copyright: "copyright",
  site_description: "siteDescription",
  remote_branch: "remoteBranch",
  edit_uri: "editUri"
} as const;

const navSourcePattern = /^\s*-\s+(?:[^:]+:\s+)?(.+?\.md)\s*$/;
const topLevelNavPattern = /^\s{2}-\s+([^:]+):\s*(.*)$/;
const nestedMarkdownPattern = /^\s{4}-\s+(?:[^:]+:\s+)?(.+?\.md)\s*$/;
const siteSectionPublishedKeyMap = {
  nav: "nav",
  home_track: "homeTrack",
  home_explore: "homeExplore",
  footer_explore: "footerExplore",
  footer_project: "footerProject"
} as const;

let metadataCache: MkdocsSiteMetadata | null = null;
let navSourcesCache: string[] | null = null;
let topLevelNavSectionsCache: MkdocsTopLevelNavSection[] | null = null;
let siteSectionsCache: Map<string, MkdocsSiteSectionConfig> | null = null;
let sectionPagesCache: Map<string, MkdocsSectionPageConfig> | null = null;
let homeConfigCache: MkdocsHomeConfig | null = null;

export const generatedSiteConfigModulePath = path.join(appRoot, "lib", "site-config.generated.ts");

function readMkdocsConfigText(): string {
  return fs.readFileSync(mkdocsConfigPath, "utf8");
}

function parseScalar(value: string): string {
  const trimmed = value.trim();
  if (
    (trimmed.startsWith("'") && trimmed.endsWith("'")) ||
    (trimmed.startsWith("\"") && trimmed.endsWith("\""))
  ) {
    return trimmed.slice(1, -1);
  }

  return trimmed;
}

function parseBooleanScalar(value: string): boolean | null {
  const normalized = parseScalar(value).toLowerCase();
  if (normalized === "true") {
    return true;
  }
  if (normalized === "false") {
    return false;
  }

  return null;
}

function baseMarkdownPath(relativePath: string): string {
  return relativePath.replace(/\.(zh|en)\.md$/, ".md");
}

function indentation(line: string): number {
  return line.match(/^\s*/)?.[0].length ?? 0;
}

function isTopLevelConfigLine(line: string): boolean {
  return /^[A-Za-z_][A-Za-z0-9_-]*:\s*/.test(line);
}

function matchIndentedKey(line: string, spaces: number): [string, string] | null {
  const match = line.match(new RegExp(`^\\s{${spaces}}([A-Za-z0-9_-]+):\\s*(.*)$`));
  return match ? [match[1], match[2] ?? ""] : null;
}

function matchIndentedAnyKey(line: string, spaces: number): [string, string] | null {
  const match = line.match(new RegExp(`^\\s{${spaces}}(\\S[^:]*):\\s*(.*)$`));
  return match ? [parseScalar(match[1] ?? ""), match[2] ?? ""] : null;
}

function matchIndentedListKey(line: string, spaces: number): [string, string] | null {
  const match = line.match(new RegExp(`^\\s{${spaces}}-\\s+([A-Za-z0-9_-]+):\\s*(.*)$`));
  return match ? [match[1], match[2] ?? ""] : null;
}

function readTopLevelMetadata(): MkdocsSiteMetadata {
  const values: Partial<MkdocsSiteMetadata> = {};

  for (const line of readMkdocsConfigText().split(/\r?\n/)) {
    const match = line.match(/^([A-Za-z_][A-Za-z0-9_-]*):\s*(.*)$/);
    if (!match) {
      continue;
    }

    const sourceKey = match[1] as keyof typeof metadataKeyMap;
    const targetKey = metadataKeyMap[sourceKey];
    if (!targetKey) {
      continue;
    }

    const value = parseScalar(match[2] ?? "");
    if (value) {
      values[targetKey] = value;
    }
  }

  return {
    siteName: requireScalar(values.siteName, "site_name"),
    repoUrl: requireScalar(values.repoUrl, "repo_url"),
    siteUrl: requireScalar(values.siteUrl, "site_url"),
    copyright: requireScalar(values.copyright, "copyright"),
    siteDescription: requireScalar(values.siteDescription, "site_description"),
    remoteBranch: requireScalar(values.remoteBranch, "remote_branch"),
    editUri: requireScalar(values.editUri, "edit_uri")
  };
}

export function readMkdocsSiteMetadata(): MkdocsSiteMetadata {
  if (!useContentCache || !metadataCache) {
    metadataCache = readTopLevelMetadata();
  }

  return metadataCache;
}

function ensureSiteSectionConfig(
  sections: Map<string, MkdocsSiteSectionConfig>,
  key: string
): MkdocsSiteSectionConfig {
  const existing = sections.get(key);
  if (existing) {
    return existing;
  }

  const config: MkdocsSiteSectionConfig = {};
  sections.set(key, config);
  return config;
}

function readSiteSections(): Map<string, MkdocsSiteSectionConfig> {
  const sections = new Map<string, MkdocsSiteSectionConfig>();
  let inSiteSections = false;
  let baseIndent = 0;
  let currentKey: string | null = null;
  let currentNestedField: "labels" | "published" | null = null;

  for (const line of readMkdocsConfigText().split(/\r?\n/)) {
    if (!inSiteSections) {
      const siteSectionsMatch = line.match(/^(\s*)site_sections:\s*$/);
      if (siteSectionsMatch) {
        inSiteSections = true;
        baseIndent = siteSectionsMatch[1]?.length ?? 0;
      }
      continue;
    }

    if (line.trim() && indentation(line) <= baseIndent) {
      break;
    }

    if (!line.trim() || line.trim().startsWith("#")) {
      continue;
    }

    const sectionMatch = matchIndentedKey(line, baseIndent + 2);
    if (sectionMatch) {
      const [sectionKey, value] = sectionMatch;
      if (value) {
        throw new Error(`Inline site_sections.${sectionKey} values are not supported`);
      }
      currentKey = sectionKey;
      currentNestedField = null;
      ensureSiteSectionConfig(sections, currentKey);
      continue;
    }

    if (!currentKey) {
      throw new Error(`Invalid site_sections entry before a section key: ${line.trim()}`);
    }

    const config = ensureSiteSectionConfig(sections, currentKey);
    const topLevelFieldMatch = matchIndentedKey(line, baseIndent + 4);
    if (topLevelFieldMatch) {
      const [field, rawValue] = topLevelFieldMatch;
      const value = parseScalar(rawValue);
      if (field !== "labels" && field !== "published" && field !== "order") {
        throw new Error(`Unsupported site_sections key for "${currentKey}": ${field}`);
      }

      if (field === "order") {
        const order = Number(value);
        if (!Number.isInteger(order)) {
          throw new Error(`Invalid site_sections order for "${currentKey}": ${value}`);
        }
        config.order = order;
        currentNestedField = null;
        continue;
      }

      if (value) {
        throw new Error(`Inline site_sections.${currentKey}.${field} values are not supported`);
      }
      currentNestedField = field;
      continue;
    }

    const nestedFieldMatch = matchIndentedKey(line, baseIndent + 6);
    if (!nestedFieldMatch || !currentNestedField) {
      throw new Error(`Invalid site_sections entry for "${currentKey}": ${line.trim()}`);
    }

    const [sourceKey, rawValue] = nestedFieldMatch;
    const value = parseScalar(rawValue);
    if (currentNestedField === "labels") {
      if (sourceKey !== "en" && sourceKey !== "zh") {
        throw new Error(`Unsupported site_sections label locale for "${currentKey}": ${sourceKey}`);
      }
      config.labels = config.labels ?? {};
      config.labels[sourceKey] = value;
      continue;
    }

    const targetKey = siteSectionPublishedKeyMap[sourceKey as keyof typeof siteSectionPublishedKeyMap];
    if (!targetKey) {
      throw new Error(`Unsupported site_sections published key for "${currentKey}": ${sourceKey}`);
    }

    const booleanValue = parseBooleanScalar(value);
    if (booleanValue === null) {
      throw new Error(`Invalid site_sections published value for "${currentKey}.${sourceKey}": ${value}`);
    }
    config.published = config.published ?? {};
    config.published[targetKey] = booleanValue;
  }

  return sections;
}

export function readMkdocsSiteSections(): Map<string, MkdocsSiteSectionConfig> {
  if (!useContentCache || !siteSectionsCache) {
    siteSectionsCache = readSiteSections();
  }

  return siteSectionsCache;
}

type PartialLocalizedText = Partial<MkdocsLocalizedText>;

type PartialSectionPageConfig = {
  source: string;
  title: PartialLocalizedText;
  description: PartialLocalizedText;
  body: PartialLocalizedText;
};

type PartialHomeProjectLink = {
  label?: string;
  labelZh?: string;
  href?: string;
};

type PartialHomeProject = {
  key?: string;
  title?: string;
  tag: PartialLocalizedText;
  href?: string;
  image?: string;
  imageAlt?: string;
  description: PartialLocalizedText;
  links: PartialHomeProjectLink[];
};

type PartialHomeHeroConfig = {
  kicker: PartialLocalizedText;
  title: PartialLocalizedText;
  summary: PartialLocalizedText;
  primaryCta: PartialLocalizedText;
  primaryHref?: string;
  secondaryCta: PartialLocalizedText;
  secondaryHref?: string;
};

type PartialHomeCapability = {
  key?: string;
  eyebrow: PartialLocalizedText;
  title: PartialLocalizedText;
  description: PartialLocalizedText;
};

type PartialHomeProjectGroup = {
  key?: string;
  title: PartialLocalizedText;
  intro: PartialLocalizedText;
  projectKeys: string[];
};

type PartialHomeConfig = {
  hero: PartialHomeHeroConfig;
  capabilitiesTitle: PartialLocalizedText;
  capabilitiesIntro: PartialLocalizedText;
  capabilities: PartialHomeCapability[];
  projectGroups: PartialHomeProjectGroup[];
  projectsTitle: PartialLocalizedText;
  projectsIntro: PartialLocalizedText;
  projects: PartialHomeProject[];
  latestTitle: PartialLocalizedText;
  allPostsLabel: PartialLocalizedText;
};

function setLocalizedValue(target: PartialLocalizedText, key: string, value: string, context: string) {
  if (key !== "en" && key !== "zh") {
    throw new Error(`Unsupported locale for ${context}: ${key}`);
  }

  target[key] = value;
}

function requireScalar(value: string | undefined, context: string): string {
  const trimmed = value?.trim();
  if (!trimmed) {
    throw new Error(`Missing mkdocs config value: ${context}`);
  }

  return trimmed;
}

function requireLocalizedText(value: PartialLocalizedText, context: string): MkdocsLocalizedText {
  const en = requireScalar(value.en, `${context}.en`);
  return {
    en,
    zh: requireScalar(value.zh, `${context}.zh`)
  };
}

function parseStringList(value: string): string[] {
  const normalized = parseScalar(value);
  const content = normalized.startsWith("[") && normalized.endsWith("]")
    ? normalized.slice(1, -1)
    : normalized;

  return content
    .split(",")
    .map((entry) => parseScalar(entry).trim())
    .filter(Boolean);
}

function requireStringList(value: string[], context: string): string[] {
  if (!value.length) {
    throw new Error(`Missing mkdocs home config list: ${context}`);
  }
  return value;
}

function normalizeSectionPageConfig(page: PartialSectionPageConfig): MkdocsSectionPageConfig {
  return {
    source: page.source,
    title: requireLocalizedText(page.title, `section_pages.${page.source}.title`),
    description: requireLocalizedText(page.description, `section_pages.${page.source}.description`),
    body: requireLocalizedText(page.body, `section_pages.${page.source}.body`)
  };
}

function ensureSectionPageConfig(
  pages: Map<string, PartialSectionPageConfig>,
  source: string
): PartialSectionPageConfig {
  const existing = pages.get(source);
  if (existing) {
    return existing;
  }

  const page: PartialSectionPageConfig = {
    source,
    title: {},
    description: {},
    body: {}
  };
  pages.set(source, page);
  return page;
}

function readSectionPages(): Map<string, MkdocsSectionPageConfig> {
  const pages = new Map<string, PartialSectionPageConfig>();
  let inSectionPages = false;
  let baseIndent = 0;
  let currentPage: PartialSectionPageConfig | null = null;
  let currentLocalizedField: "title" | "description" | "body" | null = null;

  for (const line of readMkdocsConfigText().split(/\r?\n/)) {
    if (!inSectionPages) {
      const sectionPagesMatch = line.match(/^(\s*)section_pages:\s*$/);
      if (sectionPagesMatch) {
        inSectionPages = true;
        baseIndent = sectionPagesMatch[1]?.length ?? 0;
      }
      continue;
    }

    if (line.trim() && indentation(line) <= baseIndent) {
      break;
    }

    if (!line.trim() || line.trim().startsWith("#")) {
      continue;
    }

    const pageMatch = matchIndentedAnyKey(line, baseIndent + 2);
    if (pageMatch) {
      const [source, rawValue] = pageMatch;
      if (rawValue) {
        throw new Error(`Inline section_pages.${source} values are not supported`);
      }
      currentPage = ensureSectionPageConfig(pages, baseMarkdownPath(source));
      currentLocalizedField = null;
      continue;
    }

    if (!currentPage) {
      throw new Error(`Invalid section_pages entry before a page key: ${line.trim()}`);
    }

    if (currentLocalizedField) {
      const localizedMatch = matchIndentedKey(line, baseIndent + 6);
      if (localizedMatch) {
        setLocalizedValue(
          currentPage[currentLocalizedField],
          localizedMatch[0],
          parseScalar(localizedMatch[1]),
          `section_pages.${currentPage.source}.${currentLocalizedField}`
        );
        continue;
      }
    }

    const fieldMatch = matchIndentedKey(line, baseIndent + 4);
    if (!fieldMatch) {
      throw new Error(`Invalid section_pages entry for "${currentPage.source}": ${line.trim()}`);
    }

    const [field, rawValue] = fieldMatch;
    const value = parseScalar(rawValue);
    currentLocalizedField = null;

    if (field === "title" || field === "description" || field === "body") {
      if (value) {
        throw new Error(`Inline section_pages.${currentPage.source}.${field} values are not supported`);
      }
      currentLocalizedField = field;
      continue;
    }

    throw new Error(`Unsupported section_pages key for "${currentPage.source}": ${field}`);
  }

  return new Map([...pages].map(([source, page]) => [source, normalizeSectionPageConfig(page)]));
}

export function readMkdocsSectionPages(): Map<string, MkdocsSectionPageConfig> {
  if (!useContentCache || !sectionPagesCache) {
    sectionPagesCache = readSectionPages();
  }

  return sectionPagesCache;
}

export function readMkdocsSectionPage(sourceRelative: string): MkdocsSectionPageConfig | null {
  return readMkdocsSectionPages().get(baseMarkdownPath(sourceRelative)) ?? null;
}

function normalizeHomeHero(hero: PartialHomeHeroConfig): MkdocsHomeHeroConfig {
  return {
    kicker: requireLocalizedText(hero.kicker, "home.hero.kicker"),
    title: requireLocalizedText(hero.title, "home.hero.title"),
    summary: requireLocalizedText(hero.summary, "home.hero.summary"),
    primaryCta: requireLocalizedText(hero.primaryCta, "home.hero.primary_cta"),
    primaryHref: requireScalar(hero.primaryHref, "home.hero.primary_href"),
    secondaryCta: requireLocalizedText(hero.secondaryCta, "home.hero.secondary_cta"),
    secondaryHref: requireScalar(hero.secondaryHref, "home.hero.secondary_href")
  };
}

function normalizeHomeCapability(capability: PartialHomeCapability, index: number): MkdocsHomeCapability {
  const key = requireScalar(capability.key, `home.capabilities[${index}].key`);
  return {
    key,
    eyebrow: requireLocalizedText(capability.eyebrow, `home.capabilities.${key}.eyebrow`),
    title: requireLocalizedText(capability.title, `home.capabilities.${key}.title`),
    description: requireLocalizedText(capability.description, `home.capabilities.${key}.description`)
  };
}

function normalizeHomeProjectGroup(group: PartialHomeProjectGroup, index: number): MkdocsHomeProjectGroup {
  const key = requireScalar(group.key, `home.project_groups[${index}].key`);
  return {
    key,
    title: requireLocalizedText(group.title, `home.project_groups.${key}.title`),
    intro: requireLocalizedText(group.intro, `home.project_groups.${key}.intro`),
    projectKeys: requireStringList(group.projectKeys, `home.project_groups.${key}.project_keys`)
  };
}

function normalizeHomeProject(project: PartialHomeProject, index: number): MkdocsHomeProject {
  const key = requireScalar(project.key, `home.projects[${index}].key`);
  return {
    key,
    title: requireScalar(project.title, `home.projects.${key}.title`),
    tag: requireLocalizedText(project.tag, `home.projects.${key}.tag`),
    href: requireScalar(project.href, `home.projects.${key}.href`),
    image: requireScalar(project.image, `home.projects.${key}.image`),
    imageAlt: requireScalar(project.imageAlt, `home.projects.${key}.image_alt`),
    description: requireLocalizedText(project.description, `home.projects.${key}.description`),
    links: project.links.map((link, linkIndex) => {
      const label = requireScalar(link.label, `home.projects.${key}.links[${linkIndex}].label`);
      return {
        label: {
          en: label,
          zh: requireScalar(link.labelZh, `home.projects.${key}.links[${linkIndex}].label_zh`)
        },
        href: requireScalar(link.href, `home.projects.${key}.links[${linkIndex}].href`)
      };
    })
  };
}

function readHomeConfig(): MkdocsHomeConfig {
  const config: PartialHomeConfig = {
    hero: {
      kicker: {},
      title: {},
      summary: {},
      primaryCta: {},
      secondaryCta: {}
    },
    capabilitiesTitle: {},
    capabilitiesIntro: {},
    capabilities: [],
    projectGroups: [],
    projectsTitle: {},
    projectsIntro: {},
    projects: [],
    latestTitle: {},
    allPostsLabel: {}
  };
  let inHome = false;
  let baseIndent = 0;
  let currentHomeLocalizedField:
    | "capabilitiesTitle"
    | "capabilitiesIntro"
    | "projectsTitle"
    | "projectsIntro"
    | "latestTitle"
    | "allPostsLabel"
    | null = null;
  let inHero = false;
  let currentHeroLocalizedField: "kicker" | "title" | "summary" | "primaryCta" | "secondaryCta" | null = null;
  let inCapabilities = false;
  let currentCapability: PartialHomeCapability | null = null;
  let currentCapabilityLocalizedField: "eyebrow" | "title" | "description" | null = null;
  let inProjectGroups = false;
  let currentProjectGroup: PartialHomeProjectGroup | null = null;
  let currentProjectGroupLocalizedField: "title" | "intro" | null = null;
  let inProjects = false;
  let currentProject: PartialHomeProject | null = null;
  let currentProjectLocalizedField: "tag" | "description" | null = null;
  let inLinks = false;
  let currentLink: PartialHomeProjectLink | null = null;

  function resetHomeState() {
    currentHomeLocalizedField = null;
    inHero = false;
    currentHeroLocalizedField = null;
    inCapabilities = false;
    currentCapability = null;
    currentCapabilityLocalizedField = null;
    inProjectGroups = false;
    currentProjectGroup = null;
    currentProjectGroupLocalizedField = null;
    inProjects = false;
    currentProject = null;
    currentProjectLocalizedField = null;
    inLinks = false;
    currentLink = null;
  }

  for (const line of readMkdocsConfigText().split(/\r?\n/)) {
    if (!inHome) {
      const homeMatch = line.match(/^(\s*)home:\s*$/);
      if (homeMatch) {
        inHome = true;
        baseIndent = homeMatch[1]?.length ?? 0;
      }
      continue;
    }

    if (line.trim() && indentation(line) <= baseIndent) {
      break;
    }

    if (!line.trim() || line.trim().startsWith("#")) {
      continue;
    }

    const topLevelMatch = matchIndentedKey(line, baseIndent + 2);
    if (topLevelMatch) {
      const [field, rawValue] = topLevelMatch;
      const value = parseScalar(rawValue);
      resetHomeState();

      if (field === "hero") {
        if (value) {
          throw new Error("Inline home.hero values are not supported");
        }
        inHero = true;
        continue;
      }

      const localizedHomeFields = {
        capabilities_title: "capabilitiesTitle",
        capabilities_intro: "capabilitiesIntro",
        projects_title: "projectsTitle",
        projects_intro: "projectsIntro",
        latest_title: "latestTitle",
        all_posts_label: "allPostsLabel"
      } as const;
      const localizedHomeField = localizedHomeFields[field as keyof typeof localizedHomeFields];
      if (localizedHomeField) {
        if (value) {
          throw new Error(`Inline home.${field} values are not supported`);
        }
        currentHomeLocalizedField = localizedHomeField;
        continue;
      }

      if (field === "capabilities") {
        if (value) {
          throw new Error("Inline home.capabilities values are not supported");
        }
        inCapabilities = true;
        continue;
      }

      if (field === "project_groups") {
        if (value) {
          throw new Error("Inline home.project_groups values are not supported");
        }
        inProjectGroups = true;
        continue;
      }

      if (field === "projects") {
        if (value) {
          throw new Error("Inline home.projects values are not supported");
        }
        inProjects = true;
        continue;
      }

      throw new Error(`Unsupported home config key: ${field}`);
    }

    if (currentHomeLocalizedField) {
      const localizedMatch = matchIndentedKey(line, baseIndent + 4);
      if (!localizedMatch) {
        throw new Error(`Invalid home.${currentHomeLocalizedField} entry: ${line.trim()}`);
      }
      setLocalizedValue(config[currentHomeLocalizedField], localizedMatch[0], parseScalar(localizedMatch[1]), currentHomeLocalizedField);
      continue;
    }

    if (inHero) {
      if (currentHeroLocalizedField) {
        const localizedMatch = matchIndentedKey(line, baseIndent + 6);
        if (localizedMatch) {
          setLocalizedValue(
            config.hero[currentHeroLocalizedField],
            localizedMatch[0],
            parseScalar(localizedMatch[1]),
            `home.hero.${currentHeroLocalizedField}`
          );
          continue;
        }
      }

      const heroFieldMatch = matchIndentedKey(line, baseIndent + 4);
      if (!heroFieldMatch) {
        throw new Error(`Invalid home.hero entry: ${line.trim()}`);
      }

      const [field, rawValue] = heroFieldMatch;
      const value = parseScalar(rawValue);
      currentHeroLocalizedField = null;

      const localizedHeroFields = {
        kicker: "kicker",
        title: "title",
        summary: "summary",
        primary_cta: "primaryCta",
        secondary_cta: "secondaryCta"
      } as const;
      const localizedHeroField = localizedHeroFields[field as keyof typeof localizedHeroFields];
      if (localizedHeroField) {
        if (value) {
          throw new Error(`Inline home.hero.${field} values are not supported`);
        }
        currentHeroLocalizedField = localizedHeroField;
        continue;
      }

      if (field === "primary_href") {
        config.hero.primaryHref = value;
        continue;
      }
      if (field === "secondary_href") {
        config.hero.secondaryHref = value;
        continue;
      }

      throw new Error(`Unsupported home.hero key: ${field}`);
    }

    if (inCapabilities) {
      const capabilityMatch = matchIndentedListKey(line, baseIndent + 4);
      if (capabilityMatch) {
        const [field, rawValue] = capabilityMatch;
        if (field !== "key") {
          throw new Error(`home.capabilities entries must start with key, found: ${field}`);
        }

        currentCapability = {
          key: parseScalar(rawValue),
          eyebrow: {},
          title: {},
          description: {}
        };
        config.capabilities.push(currentCapability);
        currentCapabilityLocalizedField = null;
        continue;
      }

      if (!currentCapability) {
        throw new Error(`Invalid home.capabilities entry before a capability key: ${line.trim()}`);
      }

      if (currentCapabilityLocalizedField) {
        const localizedMatch = matchIndentedKey(line, baseIndent + 8);
        if (localizedMatch) {
          setLocalizedValue(
            currentCapability[currentCapabilityLocalizedField],
            localizedMatch[0],
            parseScalar(localizedMatch[1]),
            `home.capabilities.${currentCapability.key}.${currentCapabilityLocalizedField}`
          );
          continue;
        }
      }

      const capabilityFieldMatch = matchIndentedKey(line, baseIndent + 6);
      if (!capabilityFieldMatch) {
        throw new Error(`Invalid home.capabilities entry for "${currentCapability.key ?? "unknown"}": ${line.trim()}`);
      }

      const [field, rawValue] = capabilityFieldMatch;
      const value = parseScalar(rawValue);
      currentCapabilityLocalizedField = null;

      if (field === "eyebrow" || field === "title" || field === "description") {
        if (value) {
          throw new Error(`Inline home.capabilities.${field} values are not supported`);
        }
        currentCapabilityLocalizedField = field;
        continue;
      }

      throw new Error(`Unsupported home.capabilities key for "${currentCapability.key ?? "unknown"}": ${field}`);
    }

    if (inProjectGroups) {
      const groupMatch = matchIndentedListKey(line, baseIndent + 4);
      if (groupMatch) {
        const [field, rawValue] = groupMatch;
        if (field !== "key") {
          throw new Error(`home.project_groups entries must start with key, found: ${field}`);
        }

        currentProjectGroup = {
          key: parseScalar(rawValue),
          title: {},
          intro: {},
          projectKeys: []
        };
        config.projectGroups.push(currentProjectGroup);
        currentProjectGroupLocalizedField = null;
        continue;
      }

      if (!currentProjectGroup) {
        throw new Error(`Invalid home.project_groups entry before a group key: ${line.trim()}`);
      }

      if (currentProjectGroupLocalizedField) {
        const localizedMatch = matchIndentedKey(line, baseIndent + 8);
        if (localizedMatch) {
          setLocalizedValue(
            currentProjectGroup[currentProjectGroupLocalizedField],
            localizedMatch[0],
            parseScalar(localizedMatch[1]),
            `home.project_groups.${currentProjectGroup.key}.${currentProjectGroupLocalizedField}`
          );
          continue;
        }
      }

      const groupFieldMatch = matchIndentedKey(line, baseIndent + 6);
      if (!groupFieldMatch) {
        throw new Error(`Invalid home.project_groups entry for "${currentProjectGroup.key ?? "unknown"}": ${line.trim()}`);
      }

      const [field, rawValue] = groupFieldMatch;
      const value = parseScalar(rawValue);
      currentProjectGroupLocalizedField = null;

      if (field === "title" || field === "intro") {
        if (value) {
          throw new Error(`Inline home.project_groups.${field} values are not supported`);
        }
        currentProjectGroupLocalizedField = field;
        continue;
      }

      if (field === "project_keys") {
        currentProjectGroup.projectKeys = parseStringList(value);
        continue;
      }

      throw new Error(`Unsupported home.project_groups key for "${currentProjectGroup.key ?? "unknown"}": ${field}`);
    }

    if (!inProjects) {
      throw new Error(`Invalid home config entry: ${line.trim()}`);
    }

    const projectMatch = matchIndentedListKey(line, baseIndent + 4);
    if (projectMatch) {
      const [field, rawValue] = projectMatch;
      if (field !== "key") {
        throw new Error(`home.projects entries must start with key, found: ${field}`);
      }

      currentProject = {
        key: parseScalar(rawValue),
        tag: {},
        description: {},
        links: []
      };
      config.projects.push(currentProject);
      currentProjectLocalizedField = null;
      inLinks = false;
      currentLink = null;
      continue;
    }

    if (!currentProject) {
      throw new Error(`Invalid home.projects entry before a project key: ${line.trim()}`);
    }

    const projectFieldMatch = matchIndentedKey(line, baseIndent + 6);
    if (projectFieldMatch) {
      const [field, rawValue] = projectFieldMatch;
      const value = parseScalar(rawValue);
      currentProjectLocalizedField = null;
      inLinks = false;
      currentLink = null;

      if (field === "title" || field === "href" || field === "image") {
        currentProject[field] = value;
        continue;
      }

      if (field === "image_alt") {
        currentProject.imageAlt = value;
        continue;
      }

      if (field === "tag" || field === "description") {
        if (value) {
          throw new Error(`Inline home.projects.${field} values are not supported`);
        }
        currentProjectLocalizedField = field;
        continue;
      }

      if (field === "links") {
        if (value) {
          throw new Error("Inline home.projects.links values are not supported");
        }
        inLinks = true;
        continue;
      }

      throw new Error(`Unsupported home.projects key for "${currentProject.key ?? "unknown"}": ${field}`);
    }

    if (currentProjectLocalizedField) {
      const localizedMatch = matchIndentedKey(line, baseIndent + 8);
      if (!localizedMatch) {
        throw new Error(`Invalid home.projects.${currentProjectLocalizedField} entry: ${line.trim()}`);
      }
      setLocalizedValue(
        currentProject[currentProjectLocalizedField],
        localizedMatch[0],
        parseScalar(localizedMatch[1]),
        `home.projects.${currentProject.key}.${currentProjectLocalizedField}`
      );
      continue;
    }

    if (!inLinks) {
      throw new Error(`Invalid home.projects entry for "${currentProject.key ?? "unknown"}": ${line.trim()}`);
    }

    const linkMatch = matchIndentedListKey(line, baseIndent + 8);
    if (linkMatch) {
      const [field, rawValue] = linkMatch;
      if (field !== "label") {
        throw new Error(`home.projects links must start with label, found: ${field}`);
      }
      currentLink = {
        label: parseScalar(rawValue)
      };
      currentProject.links.push(currentLink);
      continue;
    }

    if (!currentLink) {
      throw new Error(`Invalid home.projects.links entry before a link label: ${line.trim()}`);
    }

    const linkFieldMatch = matchIndentedKey(line, baseIndent + 10);
    if (!linkFieldMatch) {
      throw new Error(`Invalid home.projects.links entry: ${line.trim()}`);
    }

    const [field, rawValue] = linkFieldMatch;
    const value = parseScalar(rawValue);
    if (field === "href") {
      currentLink.href = value;
      continue;
    }
    if (field === "label_zh") {
      currentLink.labelZh = value;
      continue;
    }

    throw new Error(`Unsupported home.projects link key: ${field}`);
  }

  if (!inHome) {
    throw new Error("Missing required mkdocs config section: extra.home");
  }

  return {
    hero: normalizeHomeHero(config.hero),
    capabilitiesTitle: requireLocalizedText(config.capabilitiesTitle, "home.capabilities_title"),
    capabilitiesIntro: requireLocalizedText(config.capabilitiesIntro, "home.capabilities_intro"),
    capabilities: config.capabilities.map(normalizeHomeCapability),
    projectGroups: config.projectGroups.map(normalizeHomeProjectGroup),
    projectsTitle: requireLocalizedText(config.projectsTitle, "home.projects_title"),
    projectsIntro: requireLocalizedText(config.projectsIntro, "home.projects_intro"),
    projects: config.projects.map(normalizeHomeProject),
    latestTitle: requireLocalizedText(config.latestTitle, "home.latest_title"),
    allPostsLabel: requireLocalizedText(config.allPostsLabel, "home.all_posts_label")
  };
}

export function readMkdocsHomeConfig(): MkdocsHomeConfig {
  if (!useContentCache || !homeConfigCache) {
    homeConfigCache = readHomeConfig();
  }

  return homeConfigCache;
}

function renderSiteConfigModule(metadata: MkdocsSiteMetadata): string {
  return `export const generatedSiteConfig = {
  name: ${JSON.stringify(metadata.siteName)},
  description: ${JSON.stringify(metadata.siteDescription)},
  siteUrl: ${JSON.stringify(metadata.siteUrl)},
  repoUrl: ${JSON.stringify(metadata.repoUrl)},
  copyright: ${JSON.stringify(metadata.copyright)},
  remoteBranch: ${JSON.stringify(metadata.remoteBranch)},
  editUri: ${JSON.stringify(metadata.editUri)}
} as const;
`;
}

export function writeGeneratedSiteConfig(outputPath: string = generatedSiteConfigModulePath) {
  const metadata = readMkdocsSiteMetadata();
  const content = renderSiteConfigModule(metadata);

  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, content, "utf8");

  return {
    filePath: outputPath,
    metadata
  };
}

export function readMkdocsNavSources(): string[] {
  if (useContentCache && navSourcesCache) {
    return navSourcesCache;
  }

  const orderedSources: string[] = [];

  for (const line of readMkdocsConfigText().split(/\r?\n/)) {
    if (line.includes("... |")) {
      continue;
    }

    const match = line.match(navSourcePattern);
    const candidate = match?.[1]?.trim();
    if (candidate) {
      orderedSources.push(baseMarkdownPath(candidate));
    }
  }

  navSourcesCache = [...new Set(orderedSources)];
  return navSourcesCache;
}

type PendingTopLevelNavItem = {
  label: string;
  target?: string;
  source?: string;
};

function inferSectionKey(item: PendingTopLevelNavItem): string | null {
  const explicitTarget = item.target && !item.target.endsWith(".md") ? item.target : undefined;
  const target = explicitTarget ?? item.source ?? item.target;
  if (!target) {
    return null;
  }

  if (target === "blog") {
    return "blog";
  }

  const source = baseMarkdownPath(target);
  if (source === "index.md") {
    return null;
  }

  return source.split("/")[0] ?? null;
}

function readTopLevelNavSections(): MkdocsTopLevelNavSection[] {
  const sections: MkdocsTopLevelNavSection[] = [];
  let inNav = false;
  let pending: PendingTopLevelNavItem | null = null;

  function flushPending() {
    if (!pending) {
      return;
    }

    const key = inferSectionKey(pending);
    if (key) {
      sections.push({
        key,
        label: pending.label,
        order: (sections.length + 1) * 10,
        source: pending.source
      });
    }

    pending = null;
  }

  for (const line of readMkdocsConfigText().split(/\r?\n/)) {
    if (!inNav) {
      if (/^nav:\s*$/.test(line)) {
        inNav = true;
      }
      continue;
    }

    if (line.trim() && isTopLevelConfigLine(line)) {
      break;
    }

    const topLevelMatch = line.match(topLevelNavPattern);
    if (topLevelMatch) {
      flushPending();

      const target = parseScalar(topLevelMatch[2] ?? "");
      pending = {
        label: topLevelMatch[1].trim(),
        target: target || undefined,
        source: target.endsWith(".md") ? baseMarkdownPath(target) : undefined
      };
      continue;
    }

    if (!pending || pending.source || line.includes("... |")) {
      continue;
    }

    const nestedMatch = line.match(nestedMarkdownPattern);
    const candidate = nestedMatch?.[1]?.trim();
    if (candidate) {
      pending.source = baseMarkdownPath(candidate);
    }
  }

  flushPending();
  return sections;
}

export function readMkdocsTopLevelNavSections(): MkdocsTopLevelNavSection[] {
  if (!useContentCache || !topLevelNavSectionsCache) {
    topLevelNavSectionsCache = readTopLevelNavSections();
  }

  return topLevelNavSectionsCache;
}

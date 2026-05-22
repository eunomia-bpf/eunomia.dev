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

const fallbackMetadata: MkdocsSiteMetadata = {
  siteName: "eunomia",
  repoUrl: "https://github.com/eunomia-bpf/eunomia.dev",
  siteUrl: "https://eunomia.dev",
  copyright: "Copyright (c) 2025 eunomia-bpf org.",
  siteDescription: "Unlock the potential of eBPF",
  remoteBranch: "docs",
  editUri: "https://github.com/eunomia-bpf/eunomia.dev/tree/main/docs"
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
    ...fallbackMetadata,
    ...values
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

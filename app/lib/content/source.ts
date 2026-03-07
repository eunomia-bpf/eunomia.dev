import path from "node:path";

import type { Locale } from "../site-data";
import { getDocsFileSet, getTopLevelSections } from "./fs-index";

export function isLocalizedMarkdown(relativePath: string): boolean {
  return relativePath.endsWith(".zh.md");
}

export function isEnglishMarkdown(relativePath: string): boolean {
  return relativePath.endsWith(".en.md");
}

export function baseMarkdownPath(relativePath: string): string {
  return relativePath.replace(/\.(zh|en)\.md$/, ".md");
}

export function englishVariant(relativePath: string): string {
  return baseMarkdownPath(relativePath).replace(/\.md$/, ".en.md");
}

export function localizedVariant(relativePath: string, locale: Locale): string {
  const basePath = baseMarkdownPath(relativePath);
  if (locale === "en") {
    return englishVariant(basePath);
  }
  return basePath.replace(/\.md$/, ".zh.md");
}

export function resolveLocalizedSource(relativePath: string, locale: Locale): string | null {
  const docsFiles = getDocsFileSet();
  const candidates =
    locale === "zh"
      ? [localizedVariant(relativePath, "zh"), englishVariant(relativePath), baseMarkdownPath(relativePath)]
      : [englishVariant(relativePath), baseMarkdownPath(relativePath)];

  for (const candidate of candidates) {
    if (docsFiles.has(candidate)) {
      return candidate;
    }
  }

  return null;
}

export function formatGithubSourcePath(relativePath: string): string {
  return `https://github.com/eunomia-bpf/eunomia.dev/tree/main/docs/${relativePath}`;
}

export function slugifyTitle(value: string): string {
  const slug = value
    .toLowerCase()
    .normalize("NFKD")
    .replace(/\p{Mark}+/gu, "")
    .replace(/[^\p{Letter}\p{Number}]+/gu, "-")
    .replace(/^-+|-+$/g, "")
    .replace(/-{2,}/g, "-");

  return slug || value.trim().replace(/\s+/g, "-");
}

export function sortNaturally(values: string[]): string[] {
  return [...values].sort((left, right) =>
    left.localeCompare(right, "en", {
      numeric: true,
      sensitivity: "base"
    })
  );
}

export function isSupportedSection(section: string): boolean {
  return getTopLevelSections().includes(section);
}

export function makeAlternates(pathname: string): { en: string; zh: string } {
  const englishPath = pathname.startsWith("/zh/") ? pathname.replace(/^\/zh/, "") || "/" : pathname;
  const zhPath = pathname.startsWith("/zh/") ? pathname : pathname === "/" ? "/zh/" : `/zh${pathname}`;

  return {
    en: englishPath,
    zh: zhPath
  };
}

export function tutorialSourceToSlugSegments(sourceRelative: string): string[] {
  const normalized = baseMarkdownPath(sourceRelative)
    .replace(/^tutorials\//, "")
    .replace(/\.md$/, "");
  const pieces = normalized.split("/");
  const trailing = pieces.at(-1);
  return trailing === "README" || trailing === "index" ? pieces.slice(0, -1) : pieces;
}

export function sectionSourceToSlugSegments(sourceRelative: string, section: string): string[] {
  const withoutExt = sourceRelative.replace(new RegExp(`^${section}/`), "").replace(/\.md$/, "");
  const pieces = withoutExt.split("/");
  const trailing = pieces.at(-1);
  return trailing === "README" || trailing === "index" ? pieces.slice(0, -1) : pieces;
}

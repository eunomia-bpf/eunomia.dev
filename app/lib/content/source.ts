import { siteConfig, type Locale } from "../site-data";
import { localizePath } from "../paths";
import { getDocsFileSet, getTopLevelSections } from "./fs-index";

export const supportedLocales: Locale[] = ["en", "zh"];

export function isLocalizedMarkdown(relativePath: string): boolean {
  return /\.(zh|zh-CN)\.md$/.test(relativePath);
}

export function isEnglishMarkdown(relativePath: string): boolean {
  return relativePath.endsWith(".en.md");
}

export function baseMarkdownPath(relativePath: string): string {
  return relativePath.replace(/\.(zh|zh-CN|en)\.md$/, ".md");
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

export function zhCnVariant(relativePath: string): string {
  return baseMarkdownPath(relativePath).replace(/\.md$/, ".zh-CN.md");
}

export function resolveLocalizedSource(relativePath: string, locale: Locale): string | null {
  const docsFiles = getDocsFileSet();
  const candidates =
    locale === "zh"
      ? [localizedVariant(relativePath, "zh"), zhCnVariant(relativePath), englishVariant(relativePath), baseMarkdownPath(relativePath)]
      : [englishVariant(relativePath), baseMarkdownPath(relativePath)];

  for (const candidate of candidates) {
    if (docsFiles.has(candidate)) {
      return candidate;
    }
  }

  return null;
}

export function formatGithubSourcePath(relativePath: string): string {
  return `${siteConfig.editUri.replace(/\/+$/, "")}/${relativePath.replace(/^\/+/, "")}`;
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
  const englishPath = localizePath(pathname, "en");
  const zhPath = localizePath(pathname, "zh");

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
  const prefix = `${section}/`;
  const withoutSection = sourceRelative.startsWith(prefix) ? sourceRelative.slice(prefix.length) : sourceRelative;
  const withoutExt = withoutSection.replace(/\.md$/, "");
  const pieces = withoutExt.split("/");
  const trailing = pieces.at(-1);
  return trailing === "README" || trailing === "index" ? pieces.slice(0, -1) : pieces;
}

export function resolveSectionPageSource(
  section: string,
  slugSegments: string[] | undefined,
  locale: Locale
): string | null {
  const joined = slugSegments?.join("/");

  if (joined) {
    return (
      resolveLocalizedSource(`${section}/${joined}.md`, locale) ??
      resolveLocalizedSource(`${section}/${joined}/README.md`, locale) ??
      resolveLocalizedSource(`${section}/${joined}/index.md`, locale)
    );
  }

  return resolveLocalizedSource(`${section}/index.md`, locale) ?? resolveLocalizedSource(`${section}/README.md`, locale);
}

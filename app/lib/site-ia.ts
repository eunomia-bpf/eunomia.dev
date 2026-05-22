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

export type { SiteSectionKey } from "./site-ia-source";

import siteSectionPayload from "../.generated/content/site-sections.json";

import { localizePath } from "./paths";
import type { SidebarGroup } from "./content/types";
import type { Locale } from "./site-data";
import type {
  SerializedPublishedSiteSection,
  SerializedSiteNavLink,
  SerializedSiteSidebarGroup,
  SerializedSiteSectionDefinition,
  SiteSectionKey
} from "./site-ia-source";

export type SiteSectionDefinition = SerializedSiteSectionDefinition & SerializedPublishedSiteSection & {
  href: (locale: Locale) => string;
};

export type PrimaryNavChild = {
  label: string;
  href: string;
};

export type PrimaryNavItem = PrimaryNavChild & {
  children?: PrimaryNavChild[];
};

const sectionDefinitions: SiteSectionDefinition[] = siteSectionPayload.sections.map((section) => ({
  ...section,
  ...section.published,
  href: (locale) => section.hrefByLocale[locale]
}));

const primaryNavChildrenBySection =
  (siteSectionPayload as {
    primaryNavChildrenBySection?: Record<SiteSectionKey, SerializedSiteNavLink[]>;
  }).primaryNavChildrenBySection ?? {};

const sectionSidebarsBySection =
  (siteSectionPayload as {
    sectionSidebarsBySection?: Record<SiteSectionKey, SerializedSiteSidebarGroup[]>;
  }).sectionSidebarsBySection ?? {};

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

function localizeConfiguredHref(href: string, locale: Locale): string {
  return href.startsWith("/") ? localizePath(href, locale) : href;
}

function serializeNavChild(link: SerializedSiteNavLink, locale: Locale): PrimaryNavChild {
  return {
    label: link.labels[locale],
    href: localizeConfiguredHref(link.href, locale)
  };
}

function getConfiguredNavChildren(section: SiteSectionKey, locale: Locale): PrimaryNavChild[] | undefined {
  const children = primaryNavChildrenBySection[section]?.map((link) => serializeNavChild(link, locale)) ?? [];
  return children.length ? children : undefined;
}

export function getPrimaryNav(locale: Locale): PrimaryNavItem[] {
  return sectionDefinitions
    .filter((section) => section.nav)
    .map((section) => {
      const children = getConfiguredNavChildren(section.key, locale);

      return {
        label: section.labels[locale],
        href: section.href(locale),
        ...(children ? { children } : {})
      };
    });
}

export function getSectionSidebarOverride(section: SiteSectionKey, locale: Locale): SidebarGroup[] | null {
  const groups = sectionSidebarsBySection[section];
  if (!groups?.length) {
    return null;
  }

  return groups.map((group) => ({
    title: group.titles[locale],
    items: group.items.map((item) => ({
      title: item.labels[locale],
      href: localizeConfiguredHref(item.href, locale)
    }))
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

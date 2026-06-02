import type { Locale } from "../site-data";
import { buildCollectionContinuation, buildIndexLink } from "./navigation";
import { buildSectionPath } from "./route-paths";
import { buildSectionSidebar } from "./sidebar";
import { resolveSectionPageSource } from "./source";
import type { DocsPage } from "./types";
import { resolveDocument } from "./documents";
import { loadDocumentPage, withContinuation } from "./page-loader-utils";
import { readMkdocsHomeConfig, readMkdocsSectionLandingPages } from "./mkdocs-config";

export async function loadSectionPage(
  section: string,
  slugSegments: string[] | undefined,
  locale: Locale
): Promise<DocsPage | null> {
  const sourceRelative = resolveSectionPageSource(section, slugSegments, locale);

  if (!sourceRelative) {
    return null;
  }

  const publicPath = buildSectionPath(section, slugSegments ?? [], locale);
  const page = await loadDocumentPage(sourceRelative, publicPath, locale);
  const sectionIndexSource =
    tryResolveSource(`${section}/index.md`, locale) ??
    tryResolveSource(`${section}/README.md`, locale);
  const sectionIndexHref = buildSectionPath(section, [], locale);
  const continuation = buildCollectionContinuation({
    kind: "section-page",
    locale,
    currentPath: publicPath,
    section,
    index: buildIndexLink(sectionIndexSource, sectionIndexHref)
  });

  return {
    ...withContinuation(page, continuation),
    ...buildSectionLandingProps(section, locale, slugSegments ?? []),
    ...(page.reactPage ? buildProjectCatalogProps() : {}),
    sidebar: buildSectionSidebar(section, locale)
  };
}

function buildSectionLandingProps(section: string, locale: Locale, slugSegments: string[]): Partial<DocsPage> {
  if (slugSegments.length > 0) {
    return {};
  }

  const landingPage = readMkdocsSectionLandingPages().get(section);
  if (!landingPage) {
    return {};
  }

  return {
    title: landingPage.title[locale],
    description: landingPage.description[locale],
    landingPage,
    projectCatalog: buildProjectCatalogProps().projectCatalog
  };
}

function buildProjectCatalogProps(): Partial<DocsPage> {
  const home = readMkdocsHomeConfig();

  return {
    projectCatalog: {
      projectGroups: home.projectGroups,
      projects: home.projects
    }
  };
}

function tryResolveSource(relativePath: string, locale: Locale): string | null {
  return resolveDocument(relativePath, locale)?.sourceRelative ?? null;
}

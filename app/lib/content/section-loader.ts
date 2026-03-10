import type { Locale } from "../site-data";
import { buildCollectionContinuation, buildIndexLink } from "./navigation";
import { buildSectionPath } from "./route-paths";
import { buildSectionSidebar } from "./sidebar";
import { resolveSectionPageSource } from "./source";
import type { DocsPage } from "./types";
import { resolveDocument } from "./documents";
import { loadDocumentPage, withContinuation } from "./page-loader-utils";

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
    sidebar: buildSectionSidebar(section, locale)
  };
}

function tryResolveSource(relativePath: string, locale: Locale): string | null {
  return resolveDocument(relativePath, locale)?.sourceRelative ?? null;
}

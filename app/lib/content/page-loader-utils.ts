import type { Locale } from "../site-data";
import { getDocumentBySource, resolveDocument } from "./documents";
import { getGitMetadata } from "./git";
import { resolveAlternatesFromDocSource } from "./manifest";
import { renderMarkdownBody, renderMarkdownDocumentBody } from "./render";
import { formatGithubSourcePath } from "./source";
import type { DocsPage, LandingCard } from "./types";

export function requireDocument(relativePath: string, locale: Locale) {
  const document = resolveDocument(relativePath, locale);
  if (!document) {
    throw new Error(`missing document for ${relativePath} (${locale})`);
  }

  return document;
}

export async function loadDocumentPage(
  relativePath: string,
  publicPath: string,
  locale: Locale
): Promise<DocsPage> {
  const document = requireDocument(relativePath, locale);
  const sourceRelative = document.sourceRelative;
  const rendered = await renderMarkdownDocumentBody(document.body, sourceRelative, locale);

  return {
    layout: "document",
    title: document.title,
    description: document.description,
    bodyHtml: rendered.html,
    headings: rendered.headings,
    sourcePath: formatGithubSourcePath(sourceRelative),
    metadata: getGitMetadata(sourceRelative),
    path: publicPath,
    alternates: resolveAlternatesFromDocSource(sourceRelative, locale, publicPath)
  };
}

export async function loadDirectoryPage({
  sourceRelative,
  publicPath,
  locale,
  cards,
  sidebar
}: {
  sourceRelative: string;
  publicPath: string;
  locale: Locale;
  cards: LandingCard[];
  sidebar?: DocsPage["sidebar"];
}): Promise<DocsPage> {
  const document = getDocumentBySource(sourceRelative) ?? requireDocument(sourceRelative, locale);
  const bodyHtml = await renderMarkdownBody(document.body, sourceRelative, locale);

  return {
    layout: "directory",
    title: document.title,
    description: document.description,
    bodyHtml,
    sourcePath: formatGithubSourcePath(sourceRelative),
    metadata: getGitMetadata(sourceRelative),
    path: publicPath,
    alternates: resolveAlternatesFromDocSource(sourceRelative, locale, publicPath),
    cards,
    sidebar
  };
}

export function withContinuation(
  page: DocsPage,
  continuation: DocsPage["continuation"]
): DocsPage {
  if (!continuation) {
    return page;
  }

  return {
    ...page,
    continuation
  };
}

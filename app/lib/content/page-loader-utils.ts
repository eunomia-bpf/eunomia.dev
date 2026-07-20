import type { Locale } from "../site-data";
import { localizePath } from "../paths";
import { getDocumentBySource, resolveDocument } from "./documents";
import { getGitMetadata } from "./git";
import { resolveAlternatesFromDocSource } from "./manifest";
import { readMkdocsSectionPage } from "./mkdocs-config";
import { renderMarkdownBody, renderMarkdownDocumentBody } from "./render";
import { formatGithubSourcePath } from "./source";
import type { DocsPage, LandingCard, ReactPageLink } from "./types";

export function requireDocument(relativePath: string, locale: Locale) {
  const document = resolveDocument(relativePath, locale);
  if (!document) {
    throw new Error(`missing document for ${relativePath} (${locale})`);
  }

  return document;
}

function localizeConfiguredHref(href: string, locale: Locale): string {
  return href.startsWith("/") ? localizePath(href, locale) : href;
}

function localizeReactLinks(
  links: NonNullable<ReturnType<typeof readMkdocsSectionPage>>["links"] | undefined,
  locale: Locale
): ReactPageLink[] | undefined {
  if (!links?.length) {
    return undefined;
  }

  return links.map((link) => ({
    key: link.key,
    label: link.label[locale],
    href: localizeConfiguredHref(link.href, locale),
    ...(link.variant ? { variant: link.variant } : {})
  }));
}

export async function loadDocumentPage(
  relativePath: string,
  publicPath: string,
  locale: Locale
): Promise<DocsPage> {
  const document = requireDocument(relativePath, locale);
  const sourceRelative = document.sourceRelative;
  const rendered = await renderMarkdownDocumentBody(document.body, sourceRelative, locale);
  const configuredPage = readMkdocsSectionPage(sourceRelative);

  return {
    layout: "document",
    title: document.title,
    description: document.description,
    descriptionIsExcerpt: !configuredPage && document.description === document.excerpt,
    bodyHtml: rendered.html,
    tags: document.tags,
    ...(configuredPage?.reactPage ? { reactPage: configuredPage.reactPage } : {}),
    ...(configuredPage?.links.length ? { reactLinks: localizeReactLinks(configuredPage.links, locale) } : {}),
    ...(document.date ? { date: document.date } : {}),
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
  const configuredPage = readMkdocsSectionPage(sourceRelative);

  return {
    layout: "directory",
    title: document.title,
    description: document.description,
    descriptionIsExcerpt: !configuredPage && document.description === document.excerpt,
    bodyHtml,
    ...(configuredPage?.reactPage ? { reactPage: configuredPage.reactPage } : {}),
    ...(configuredPage?.links.length ? { reactLinks: localizeReactLinks(configuredPage.links, locale) } : {}),
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

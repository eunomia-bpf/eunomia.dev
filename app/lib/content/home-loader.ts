import { getHomePath } from "../site-ia";
import type { Locale } from "../site-data";
import type { HomePageData } from "../page-factories";
import { getGitMetadata } from "./git";
import { renderMarkdownBody } from "./render";
import { makeAlternates, formatGithubSourcePath } from "./source";
import { requireDocument } from "./page-loader-utils";

function extractInlineStyles(markdown: string) {
  const styles: string[] = [];
  const body = markdown.replace(/<style>([\s\S]*?)<\/style>/gi, (_match, css: string) => {
    const trimmed = css.trim();
    if (trimmed) {
      styles.push(trimmed);
    }

    return "";
  });

  return {
    body,
    inlineStyles: styles
  };
}

export async function loadHomePage(locale: Locale): Promise<HomePageData> {
  const home = requireDocument("index.md", locale);
  const extracted = extractInlineStyles(home.body);
  const bodyHtml = await renderMarkdownBody(extracted.body, home.sourceRelative, locale);

  return {
    title: home.title,
    description: home.description,
    intro: home.excerpt || home.description,
    bodyHtml,
    inlineStyles: extracted.inlineStyles,
    sourcePath: formatGithubSourcePath("index.md"),
    metadata: getGitMetadata("index.md"),
    path: getHomePath(locale),
    alternates: makeAlternates(getHomePath(locale))
  };
}

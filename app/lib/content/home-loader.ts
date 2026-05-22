import { getHomePath } from "../site-ia";
import type { Locale } from "../site-data";
import type { HomePageData } from "../page-factories";
import { getRecentBlogEntriesForLocale } from "./collections";
import { getGitMetadata } from "./git";
import { renderMarkdownBody } from "./render";
import { makeAlternates, formatGithubSourcePath } from "./source";
import { requireDocument } from "./page-loader-utils";

export async function loadHomePage(locale: Locale): Promise<HomePageData> {
  const home = requireDocument("index.md", locale);
  const bodyHtml = await renderMarkdownBody(home.body, home.sourceRelative, locale);

  return {
    title: home.title,
    description: home.description,
    intro: home.excerpt || home.description,
    bodyHtml,
    sourcePath: formatGithubSourcePath(home.sourceRelative),
    metadata: getGitMetadata(home.sourceRelative),
    path: getHomePath(locale),
    alternates: makeAlternates(getHomePath(locale)),
    recentPosts: getRecentBlogEntriesForLocale(locale)
  };
}

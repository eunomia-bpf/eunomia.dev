import { getHomePath } from "../site-ia";
import type { Locale } from "../site-data";
import type { HomePageData } from "../page-factories";
import { getGitMetadata } from "./git";
import { renderMarkdownBody } from "./render";
import { makeAlternates, formatGithubSourcePath } from "./source";
import { requireDocument } from "./page-loader-utils";
import { getBlogEntries } from "./collections";

export async function loadHomePage(locale: Locale): Promise<HomePageData> {
  const home = requireDocument("index.md", locale);
  const bodyHtml = await renderMarkdownBody(home.body, home.sourceRelative, locale);

  // Include up to 8 most recent English blog posts for the homepage
  const allEntries = getBlogEntries();
  const recentPosts = allEntries
    .filter((entry) => entry.sourceByLocale.en)
    .slice(0, 8);

  return {
    title: home.title,
    description: home.description,
    intro: home.excerpt || home.description,
    bodyHtml,
    sourcePath: formatGithubSourcePath(home.sourceRelative),
    metadata: getGitMetadata(home.sourceRelative),
    path: getHomePath(locale),
    alternates: makeAlternates(getHomePath(locale)),
    recentPosts
  };
}

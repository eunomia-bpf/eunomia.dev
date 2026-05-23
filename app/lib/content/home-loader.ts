import { getHomePath } from "../site-ia";
import type { Locale } from "../site-data";
import type { HomePageData } from "../page-factories";
import { getRecentBlogEntriesForLocale } from "./collections";
import { getGitMetadata } from "./git";
import { readMkdocsHomeConfig } from "./mkdocs-config";
import { makeAlternates, formatGithubSourcePath } from "./source";
import { requireDocument } from "./page-loader-utils";

export async function loadHomePage(locale: Locale): Promise<HomePageData> {
  const home = requireDocument("index.md", locale);

  return {
    title: home.title,
    description: home.description,
    intro: home.excerpt || home.description,
    sourcePath: formatGithubSourcePath(home.sourceRelative),
    metadata: getGitMetadata(home.sourceRelative),
    path: getHomePath(locale),
    alternates: makeAlternates(getHomePath(locale)),
    recentPosts: getRecentBlogEntriesForLocale(locale),
    home: readMkdocsHomeConfig()
  };
}

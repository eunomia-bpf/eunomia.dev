import {
  getFeaturedHomeSections,
  getHomeBadge,
  getHomeExploreSections,
  getHomePath,
  getHomeStats
} from "../site-ia";
import type { Locale } from "../site-data";
import type { HomePageData } from "../page-factories";
import { getBlogEntries, getLegacyBlogEntries, getTutorialReadmeSources } from "./collections";
import { getGitMetadata } from "./git";
import { buildBlogIndexPath, buildBlogPath } from "./route-paths";
import { makeAlternates, formatGithubSourcePath } from "./source";
import type { LandingCard } from "./types";
import { requireDocument } from "./page-loader-utils";

export async function loadHomePage(locale: Locale): Promise<HomePageData> {
  const home = requireDocument("index.md", locale);
  const counts = {
    tutorials: getTutorialReadmeSources().length,
    blogs: getBlogEntries().length,
    legacyBlogs: getLegacyBlogEntries().length
  };
  const cards: LandingCard[] = getFeaturedHomeSections().map((section) => {
    const document = requireDocument(section.source, locale);
    return {
      title: document.title,
      description: document.description,
      href: section.href(locale),
      badge: getHomeBadge(section.key, counts, locale)
    };
  });
  const moreLinks: LandingCard[] = getHomeExploreSections().map((section) => {
    const document = requireDocument(section.indexSource, locale);
    return {
      title: document.title,
      description: document.description,
      href: section.href(locale)
    };
  });

  const latestBlog = getBlogEntries()[0];

  return {
    title: home.title,
    description: home.description,
    intro: home.excerpt || home.description,
    cards,
    moreLinks,
    stats: getHomeStats(locale, counts),
    spotlight: latestBlog
      ? {
          title: latestBlog.title,
          description: latestBlog.excerpt || latestBlog.description,
          href: buildBlogPath(latestBlog.year, latestBlog.month, latestBlog.day, latestBlog.slug, locale),
          badge: `${latestBlog.year}-${latestBlog.month}-${latestBlog.day}`
        }
      : {
          title: locale === "zh" ? "查看 Blog" : "Visit the blog",
          description: home.description,
          href: buildBlogIndexPath(locale),
          badge: locale === "zh" ? "博客" : "Blog"
        },
    sourcePath: formatGithubSourcePath("index.md"),
    metadata: getGitMetadata("index.md"),
    path: getHomePath(locale),
    alternates: makeAlternates(getHomePath(locale))
  };
}

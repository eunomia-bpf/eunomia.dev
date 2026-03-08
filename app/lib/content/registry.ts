import { routeRolloutPolicies } from "../rollout";
import type { Locale } from "../site-data";
import { getBlogEntries, getLegacyBlogEntries, getTutorialDocSources } from "./collections";
import {
  buildBlogIndexPath,
  buildBlogPath,
  buildLegacyBlogIndexPath,
  buildLegacyBlogPath,
  buildTutorialIndexPath,
  buildTutorialPath
} from "./route-paths";
import {
  resolveLocalizedSource,
  tutorialSourceToSlugSegments
} from "./source";
import type { ContentManifestKind, ContentManifestRecord } from "./types";

export type CollectionFamilyId = "tutorial" | "blog" | "legacy-blog";

export type PublishedSiteSectionFlags = {
  nav: boolean;
  homeTrack: boolean;
  homeExplore: boolean;
  footerExplore: boolean;
  footerProject: boolean;
};

type CollectionPageDescriptor = {
  kind: ContentManifestKind;
  key: string;
  slug: string[];
  resolveSource: (locale: Locale) => string | null;
  buildPath: (locale: Locale) => string;
  getSourceAliases: () => string[];
};

export type CollectionFamilyDefinition = {
  id: CollectionFamilyId;
  indexKind: ContentManifestKind;
  pageKind: ContentManifestKind;
  indexKey: string;
  indexRouteClass: ContentManifestRecord["routeClass"];
  indexSitemapStage: ContentManifestRecord["sitemapStage"];
  pageRouteClass: ContentManifestRecord["routeClass"];
  pageSitemapStage: ContentManifestRecord["sitemapStage"];
  indexSource: string;
  buildIndexPath: (locale: Locale) => string;
  eyebrow: (locale: Locale) => string;
  siteSection: {
    key: string;
    topLevelDir: string;
    defaults: PublishedSiteSectionFlags;
  };
  getPages: () => CollectionPageDescriptor[];
};

const collectionFamilies: CollectionFamilyDefinition[] = [
  {
    id: "tutorial",
    indexKind: "tutorial-index",
    pageKind: "tutorial-page",
    indexKey: "tutorials/index",
    indexRouteClass: routeRolloutPolicies.tutorial.routeClass,
    indexSitemapStage: routeRolloutPolicies.tutorial.sitemapStage,
    pageRouteClass: routeRolloutPolicies.tutorial.routeClass,
    pageSitemapStage: routeRolloutPolicies.tutorial.sitemapStage,
    indexSource: "tutorials/index.md",
    buildIndexPath: buildTutorialIndexPath,
    eyebrow: (locale) => (locale === "zh" ? "教程" : "Tutorials"),
    siteSection: {
      key: "tutorials",
      topLevelDir: "tutorials",
      defaults: {
        nav: true,
        homeTrack: true,
        homeExplore: false,
        footerExplore: true,
        footerProject: false
      }
    },
    getPages: () =>
      getTutorialDocSources().map((sourceRelative) => {
        const slug = tutorialSourceToSlugSegments(sourceRelative);
        const tutorialBase = `tutorials/${slug.join("/")}`;

        return {
          kind: "tutorial-page",
          key: `tutorial:${slug.join("/")}`,
          slug,
          resolveSource: (locale) => resolveLocalizedSource(sourceRelative, locale),
          buildPath: (locale) => buildTutorialPath(slug, locale),
          getSourceAliases: () => [`${tutorialBase}.md`, `${tutorialBase}/README.md`, `${tutorialBase}/index.md`]
        };
      })
  },
  {
    id: "blog",
    indexKind: "blog-index",
    pageKind: "blog-page",
    indexKey: "blog/index",
    indexRouteClass: routeRolloutPolicies["blog-index"].routeClass,
    indexSitemapStage: routeRolloutPolicies["blog-index"].sitemapStage,
    pageRouteClass: routeRolloutPolicies.blog.routeClass,
    pageSitemapStage: routeRolloutPolicies.blog.sitemapStage,
    indexSource: "blog/index.md",
    buildIndexPath: buildBlogIndexPath,
    eyebrow: (locale) => (locale === "zh" ? "博客" : "Blog"),
    siteSection: {
      key: "blog",
      topLevelDir: "blog",
      defaults: {
        nav: true,
        homeTrack: false,
        homeExplore: false,
        footerExplore: true,
        footerProject: false
      }
    },
    getPages: () =>
      getBlogEntries().map((entry) => ({
        kind: "blog-page",
        key: `blog:${entry.year}-${entry.month}-${entry.day}:${entry.slug}`,
        slug: [entry.year, entry.month, entry.day, entry.slug],
        resolveSource: (locale) => entry.sourceByLocale[locale] ?? entry.sourceByLocale.en ?? entry.sourceByLocale.zh ?? null,
        buildPath: (locale) => buildBlogPath(entry.year, entry.month, entry.day, entry.slug, locale),
        getSourceAliases: () => Object.values(entry.sourceByLocale).filter(Boolean) as string[]
      }))
  },
  {
    id: "legacy-blog",
    indexKind: "legacy-blog-index",
    pageKind: "legacy-blog-page",
    indexKey: "blogs/index",
    indexRouteClass: routeRolloutPolicies["legacy-blog"].routeClass,
    indexSitemapStage: routeRolloutPolicies["legacy-blog"].sitemapStage,
    pageRouteClass: routeRolloutPolicies["legacy-blog"].routeClass,
    pageSitemapStage: routeRolloutPolicies["legacy-blog"].sitemapStage,
    indexSource: "blogs/index.md",
    buildIndexPath: buildLegacyBlogIndexPath,
    eyebrow: (locale) => (locale === "zh" ? "旧博客" : "Legacy Blog"),
    siteSection: {
      key: "legacy-blog",
      topLevelDir: "blogs",
      defaults: {
        nav: false,
        homeTrack: false,
        homeExplore: true,
        footerExplore: true,
        footerProject: false
      }
    },
    getPages: () =>
      getLegacyBlogEntries().map((entry) => ({
        kind: "legacy-blog-page",
        key: `legacy-blog:${entry.key}`,
        slug: [entry.key],
        resolveSource: (locale) => entry.sourceByLocale[locale] ?? entry.sourceByLocale.en ?? entry.sourceByLocale.zh ?? null,
        buildPath: (locale) => buildLegacyBlogPath(entry.key, locale),
        getSourceAliases: () => Object.values(entry.sourceByLocale).filter(Boolean) as string[]
      }))
  }
];

export function getCollectionFamilies(): CollectionFamilyDefinition[] {
  return collectionFamilies;
}

export function getCollectionFamilyById(id: CollectionFamilyId): CollectionFamilyDefinition | null {
  return collectionFamilies.find((family) => family.id === id) ?? null;
}

export function getCollectionFamilyByKind(kind: ContentManifestKind): CollectionFamilyDefinition | null {
  return (
    collectionFamilies.find((family) => family.indexKind === kind || family.pageKind === kind) ?? null
  );
}

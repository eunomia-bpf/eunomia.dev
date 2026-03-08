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
  baseMarkdownPath,
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

export type CollectionPageDescriptor = {
  kind: ContentManifestKind;
  key: string;
  slug: string[];
  sourceByLocale: Partial<Record<Locale, string>>;
  buildPath: (locale: Locale) => string;
  sourceAliases: string[];
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
  buildPagePath: (slug: string[], locale: Locale) => string;
  eyebrow: (locale: Locale) => string;
  siteSection: {
    key: string;
    topLevelDir: string;
    defaults: PublishedSiteSectionFlags;
  };
  getPageDescriptorsFromSource: () => CollectionPageDescriptor[];
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
    buildPagePath: buildTutorialPath,
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
    getPageDescriptorsFromSource: () =>
      getTutorialDocSources().map((sourceRelative) => {
        const slug = tutorialSourceToSlugSegments(sourceRelative);
        const tutorialBase = `tutorials/${slug.join("/")}`;

        return {
          kind: "tutorial-page",
          key: `tutorial:${slug.join("/")}`,
          slug,
          sourceByLocale: {
            en: resolveLocalizedSource(sourceRelative, "en") ?? undefined,
            zh: resolveLocalizedSource(sourceRelative, "zh") ?? undefined
          },
          buildPath: (locale) => buildTutorialPath(slug, locale),
          sourceAliases: [`${tutorialBase}.md`, `${tutorialBase}/README.md`, `${tutorialBase}/index.md`].map(baseMarkdownPath)
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
    buildPagePath: (slug, locale) => {
      const [year, month, day, key] = slug;
      return buildBlogPath(year ?? "", month ?? "", day ?? "", key ?? "", locale);
    },
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
    getPageDescriptorsFromSource: () =>
      getBlogEntries().map((entry) => ({
        kind: "blog-page",
        key: `blog:${entry.year}-${entry.month}-${entry.day}:${entry.slug}`,
        slug: [entry.year, entry.month, entry.day, entry.slug],
        sourceByLocale: entry.sourceByLocale,
        buildPath: (locale) => buildBlogPath(entry.year, entry.month, entry.day, entry.slug, locale),
        sourceAliases: (Object.values(entry.sourceByLocale).filter(Boolean) as string[]).map(baseMarkdownPath)
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
    buildPagePath: (slug, locale) => buildLegacyBlogPath(slug[0] ?? "", locale),
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
    getPageDescriptorsFromSource: () =>
      getLegacyBlogEntries().map((entry) => ({
        kind: "legacy-blog-page",
        key: `legacy-blog:${entry.key}`,
        slug: [entry.key],
        sourceByLocale: entry.sourceByLocale,
        buildPath: (locale) => buildLegacyBlogPath(entry.key, locale),
        sourceAliases: (Object.values(entry.sourceByLocale).filter(Boolean) as string[]).map(baseMarkdownPath)
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
  return collectionFamilies.find((family) => family.indexKind === kind || family.pageKind === kind) ?? null;
}

export function getCollectionPageDescriptors(familyId: CollectionFamilyId): CollectionPageDescriptor[] {
  const family = getCollectionFamilyById(familyId);
  if (!family) {
    throw new Error(`Unknown collection family: ${familyId}`);
  }

  return family.getPageDescriptorsFromSource();
}

export function resolveCollectionPageDescriptor(
  familyId: CollectionFamilyId,
  slugSegments: string[] | undefined
): CollectionPageDescriptor | null {
  if (!slugSegments?.length) {
    return null;
  }

  const slugKey = slugSegments.join("/");
  return (
    getCollectionPageDescriptors(familyId).find((descriptor) => descriptor.slug.join("/") === slugKey) ?? null
  );
}

export function resolveCollectionPageSource(
  familyId: CollectionFamilyId,
  slugSegments: string[] | undefined,
  locale: Locale
): string | null {
  const descriptor = resolveCollectionPageDescriptor(familyId, slugSegments);
  if (!descriptor) {
    return null;
  }

  return descriptor.sourceByLocale[locale] ?? descriptor.sourceByLocale.en ?? descriptor.sourceByLocale.zh ?? null;
}

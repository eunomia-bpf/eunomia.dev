import type { GetServerSideProps, GetStaticProps } from "next";

import {
  loadBlogIndex,
  loadBlogPage,
  loadHomePage,
  loadLegacyBlogIndex,
  loadLegacyBlogPage,
  loadSectionPage,
  loadTutorialIndex,
  loadTutorialPage,
  resolveManifestRecordFromRoute
} from "./content";
import { renderFeed } from "./content/feed";
import { buildSearchSidebar } from "./content/sidebar";
import { searchContent } from "./content/search";
import type { SearchResult, SidebarGroup } from "./content/types";
import type { HomePageData } from "./page-factories";
import type { ContentPageProps } from "./page-builders";
import { localizePath } from "./paths";
import type { Locale } from "./site-data";

type CollectionKind = "tutorial" | "blog" | "legacy-blog";

type SearchPageProps = {
  query: string;
  results: SearchResult[];
  sidebar: SidebarGroup[];
};

function collectionEyebrow(kind: CollectionKind, locale: Locale): string {
  if (locale === "zh") {
    switch (kind) {
      case "tutorial":
        return "教程";
      case "blog":
        return "博客";
      case "legacy-blog":
        return "旧博客";
      default: {
        const unreachable: never = kind;
        throw new Error(`unsupported collection kind: ${unreachable}`);
      }
    }
  }

  switch (kind) {
    case "tutorial":
      return "Tutorials";
    case "blog":
      return "Blog";
    case "legacy-blog":
      return "Legacy Blog";
    default: {
      const unreachable: never = kind;
      throw new Error(`unsupported collection kind: ${unreachable}`);
    }
  }
}

function normalizeQuery(value: string | string[] | undefined): string {
  if (Array.isArray(value)) {
    return value[0] ?? "";
  }

  return value ?? "";
}

function serializeResults(results: SearchResult[]): SearchResult[] {
  return results.map((result) =>
    result.section
      ? result
      : {
          title: result.title,
          description: result.description,
          href: result.href,
          locale: result.locale,
          kind: result.kind
      }
  );
}

function pathnameFromSlug(locale: Locale, slug: string[] | undefined): string {
  const pathname = slug?.length ? `/${slug.join("/")}/` : "/";
  return localizePath(pathname, locale);
}

export function createHomePageRoute(locale: Locale) {
  const getStaticProps: GetStaticProps<{ page: HomePageData }> = async () => ({
    props: {
      page: await loadHomePage(locale)
    }
  });

  return {
    getStaticProps
  };
}

export function createContentPageRoute(locale: Locale) {
  const getServerSideProps: GetServerSideProps<ContentPageProps> = async ({ params }) => {
    const slug = Array.isArray(params?.slug) ? params.slug : [];
    if (!slug.length) {
      return {
        notFound: true
      };
    }

    const path = pathnameFromSlug(locale, slug);
    const record = resolveManifestRecordFromRoute(path);
    if (!record) {
      return {
        notFound: true
      };
    }

    switch (record.kind) {
      case "tutorial-index":
        return {
          props: {
            routeKind: "collection",
            eyebrow: collectionEyebrow("tutorial", locale),
            content: {
              kind: "index",
              page: await loadTutorialIndex(locale)
            }
          }
        };
      case "tutorial-page": {
        const page = await loadTutorialPage(record.slug, locale);
        if (!page) {
          return { notFound: true };
        }
        return {
          props: {
            routeKind: "collection",
            eyebrow: collectionEyebrow("tutorial", locale),
            content: {
              kind: "article",
              page
            }
          }
        };
      }
      case "blog-index":
        return {
          props: {
            routeKind: "collection",
            eyebrow: collectionEyebrow("blog", locale),
            content: {
              kind: "index",
              page: await loadBlogIndex(locale)
            }
          }
        };
      case "blog-page": {
        const page = await loadBlogPage(record.slug, locale);
        if (!page) {
          return { notFound: true };
        }
        return {
          props: {
            routeKind: "collection",
            eyebrow: collectionEyebrow("blog", locale),
            content: {
              kind: "article",
              page
            }
          }
        };
      }
      case "legacy-blog-index":
        return {
          props: {
            routeKind: "collection",
            eyebrow: collectionEyebrow("legacy-blog", locale),
            content: {
              kind: "index",
              page: await loadLegacyBlogIndex(locale)
            }
          }
        };
      case "legacy-blog-page": {
        const page = await loadLegacyBlogPage(record.slug, locale);
        if (!page) {
          return { notFound: true };
        }
        return {
          props: {
            routeKind: "collection",
            eyebrow: collectionEyebrow("legacy-blog", locale),
            content: {
              kind: "article",
              page
            }
          }
        };
      }
      case "section-page": {
        const section = record.section ?? "";
        if (!section) {
          return { notFound: true };
        }

        const page = await loadSectionPage(section, record.slug ?? [], locale);
        if (!page) {
          return { notFound: true };
        }

        return {
          props: {
            routeKind: "section",
            section,
            page
          }
        };
      }
      default:
        return {
          notFound: true
        };
    }
  };

  return {
    getServerSideProps
  };
}

export function createSearchPageRoute(locale: Locale) {
  const getServerSideProps: GetServerSideProps<SearchPageProps> = async ({ query }) => {
    const q = normalizeQuery(query.q);

    return {
      props: {
        query: q,
        results: serializeResults(searchContent(q, locale, 24)),
        sidebar: buildSearchSidebar(locale)
      }
    };
  };

  return {
    getServerSideProps
  };
}

export function createFeedPageRoute(locale: Locale) {
  const getServerSideProps: GetServerSideProps = async ({ res }) => {
    res.setHeader("Content-Type", "application/rss+xml; charset=utf-8");
    res.write(renderFeed(locale));
    res.end();

    return {
      props: {}
    };
  };

  return {
    getServerSideProps
  };
}

import type { GetServerSideProps, GetStaticPaths, GetStaticProps } from "next";

import {
  getGenericSectionRoutes,
  loadBlogIndex,
  loadBlogPage,
  loadHomePage,
  loadLegacyBlogIndex,
  loadLegacyBlogPage,
  loadSectionPage,
  loadTutorialIndex,
  loadTutorialPage
} from "./content";
import { renderFeed } from "./content/feed";
import { buildSearchSidebar } from "./content/sidebar";
import { searchContent } from "./content/search";
import type { LandingPageData, MarkdownPage, SearchResult, SidebarGroup } from "./content/types";
import { buildSectionStaticPaths, loadSectionStaticProps, type CollectionPageProps, type HomePageData } from "./page-factories";
import type { Locale } from "./site-data";

type CollectionKind = "tutorial" | "blog" | "legacy-blog";

type SearchPageProps = {
  query: string;
  results: SearchResult[];
  sidebar: SidebarGroup[];
};

type CollectionIndexProps<IndexPage extends LandingPageData, ArticlePage extends MarkdownPage> = Extract<
  CollectionPageProps<IndexPage, ArticlePage>,
  { kind: "index" }
>;

type CollectionArticleProps<IndexPage extends LandingPageData, ArticlePage extends MarkdownPage> = Extract<
  CollectionPageProps<IndexPage, ArticlePage>,
  { kind: "article" }
>;

type SectionPageProps = {
  page: MarkdownPage;
  section: string;
};

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

type CollectionRouteConfig<IndexPage extends LandingPageData, ArticlePage extends MarkdownPage> = {
  loadIndex: () => Promise<IndexPage>;
  loadArticle: (slug: string[]) => Promise<ArticlePage | null>;
};

function getCollectionRouteConfig(
  kind: CollectionKind,
  locale: Locale
): CollectionRouteConfig<LandingPageData, MarkdownPage> {
  switch (kind) {
    case "tutorial":
      return {
        loadIndex: () => loadTutorialIndex(locale),
        loadArticle: (slug) => loadTutorialPage(slug, locale)
      };
    case "blog":
      return {
        loadIndex: () => loadBlogIndex(locale),
        loadArticle: (slug) => loadBlogPage(slug, locale)
      };
    case "legacy-blog":
      return {
        loadIndex: () => loadLegacyBlogIndex(locale),
        loadArticle: (slug) => loadLegacyBlogPage(slug, locale)
      };
    default: {
      const unreachable: never = kind;
      throw new Error(`unsupported collection route kind: ${unreachable}`);
    }
  }
}

function createCollectionIndexRoute<IndexPage extends LandingPageData, ArticlePage extends MarkdownPage>({
  loadIndex,
}: CollectionRouteConfig<IndexPage, ArticlePage>) {
  type Props = CollectionIndexProps<IndexPage, ArticlePage>;

  const getStaticProps: GetStaticProps<Props> = async () => ({
    props: {
      kind: "index",
      page: await loadIndex()
    }
  });

  return {
    getStaticProps
  };
}

function createCollectionArticleRoute<IndexPage extends LandingPageData, ArticlePage extends MarkdownPage>({
  loadArticle
}: CollectionRouteConfig<IndexPage, ArticlePage>) {
  type Props = CollectionArticleProps<IndexPage, ArticlePage>;

  const getServerSideProps: GetServerSideProps<Props> = async ({ params }) => {
    const slug = Array.isArray(params?.slug) ? params.slug : [];
    if (!slug.length) {
      return {
        notFound: true
      };
    }

    const page = await loadArticle(slug);
    if (!page) {
      return {
        notFound: true
      };
    }

    return {
      props: {
        kind: "article",
        page
      }
    };
  };

  return {
    getServerSideProps
  };
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

export function createTutorialPageRoute(locale: Locale) {
  return createCollectionIndexRoute(getCollectionRouteConfig("tutorial", locale));
}

export function createTutorialArticleRoute(locale: Locale) {
  return createCollectionArticleRoute(getCollectionRouteConfig("tutorial", locale));
}

export function createBlogPageRoute(locale: Locale) {
  return createCollectionIndexRoute(getCollectionRouteConfig("blog", locale));
}

export function createBlogArticleRoute(locale: Locale) {
  return createCollectionArticleRoute(getCollectionRouteConfig("blog", locale));
}

export function createLegacyBlogPageRoute(locale: Locale) {
  return createCollectionIndexRoute(getCollectionRouteConfig("legacy-blog", locale));
}

export function createLegacyBlogArticleRoute(locale: Locale) {
  return createCollectionArticleRoute(getCollectionRouteConfig("legacy-blog", locale));
}

export function createSectionPageRoute(locale: Locale) {
  const rootRoutes = getGenericSectionRoutes(locale).filter((route) => route.slug.length === 0);
  const getStaticPaths: GetStaticPaths = async () =>
    buildSectionStaticPaths(rootRoutes.map((route) => ({ section: route.section, slug: [] })));

  const getStaticProps: GetStaticProps<SectionPageProps> = async ({ params }) =>
    loadSectionStaticProps(params, (section) => loadSectionPage(section, [], locale));

  return {
    getStaticPaths,
    getStaticProps
  };
}

export function createSectionArticleRoute(locale: Locale) {
  const getServerSideProps: GetServerSideProps<SectionPageProps> = async ({ params }) => {
    const section = typeof params?.section === "string" ? params.section : "";
    const slug = Array.isArray(params?.slug) ? params.slug : [];
    if (!section || !slug.length) {
      return {
        notFound: true
      };
    }

    const page = await loadSectionPage(section, slug, locale);
    if (!page) {
      return {
        notFound: true
      };
    }

    return {
      props: {
        page,
        section
      }
    };
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

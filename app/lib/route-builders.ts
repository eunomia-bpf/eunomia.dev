import type { GetServerSideProps, GetStaticPaths, GetStaticProps } from "next";

import {
  getBlogRoutes,
  getGenericSectionRoutes,
  getLegacyBlogRoutes,
  getTutorialRoutes,
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
import { searchContent } from "./content/search";
import type { LandingPageData, MarkdownPage, SearchResult } from "./content/types";
import { buildSectionStaticPaths, buildSlugStaticPaths, loadCollectionStaticProps, loadSectionStaticProps, type CollectionPageProps, type HomePageData } from "./page-factories";
import type { Locale } from "./site-data";

type SearchPageProps = {
  query: string;
  results: SearchResult[];
};

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
  getRoutes: () => string[][];
  loadIndex: () => Promise<IndexPage>;
  loadArticle: (slug: string[]) => Promise<ArticlePage | null>;
};

function createCollectionRoute<IndexPage extends LandingPageData, ArticlePage extends MarkdownPage>({
  getRoutes,
  loadIndex,
  loadArticle
}: CollectionRouteConfig<IndexPage, ArticlePage>) {
  type Props = CollectionPageProps<IndexPage, ArticlePage>;

  const getStaticPaths: GetStaticPaths = async () => buildSlugStaticPaths(getRoutes());

  const getStaticProps: GetStaticProps<Props> = async ({ params }) =>
    loadCollectionStaticProps(params, {
      loadIndex,
      loadArticle
    });

  return {
    getStaticPaths,
    getStaticProps
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
  return createCollectionRoute({
    getRoutes: () => getTutorialRoutes(locale),
    loadIndex: () => loadTutorialIndex(locale),
    loadArticle: (slug) => loadTutorialPage(slug, locale)
  });
}

export function createBlogPageRoute(locale: Locale) {
  return createCollectionRoute({
    getRoutes: () => getBlogRoutes(),
    loadIndex: () => loadBlogIndex(locale),
    loadArticle: (slug) => loadBlogPage(slug, locale)
  });
}

export function createLegacyBlogPageRoute(locale: Locale) {
  return createCollectionRoute({
    getRoutes: () => getLegacyBlogRoutes(),
    loadIndex: () => loadLegacyBlogIndex(locale),
    loadArticle: (slug) => loadLegacyBlogPage(slug, locale)
  });
}

export function createSectionPageRoute(locale: Locale) {
  const getStaticPaths: GetStaticPaths = async () => buildSectionStaticPaths(getGenericSectionRoutes(locale));

  const getStaticProps: GetStaticProps<SectionPageProps> = async ({ params }) =>
    loadSectionStaticProps(params, (section, slug) => loadSectionPage(section, slug, locale));

  return {
    getStaticPaths,
    getStaticProps
  };
}

export function createSearchPageRoute(locale: Locale) {
  const getServerSideProps: GetServerSideProps<SearchPageProps> = async ({ query }) => {
    const q = normalizeQuery(query.q);

    return {
      props: {
        query: q,
        results: serializeResults(searchContent(q, locale, 24))
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

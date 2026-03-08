import type { GetServerSideProps, GetStaticProps } from "next";

import { loadHomePage, resolveContentPage } from "./content";
import { renderFeed } from "./content/feed";
import { buildSearchSidebar } from "./content/sidebar";
import { searchContent } from "./content/search";
import type { SearchResult, SidebarGroup } from "./content/types";
import type { HomePageData } from "./page-factories";
import type { ContentPageProps } from "./page-builders";
import { localizePath } from "./paths";
import type { Locale } from "./site-data";

type SearchPageProps = {
  query: string;
  results: SearchResult[];
  sidebar: SidebarGroup[];
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

    const resolved = await resolveContentPage(pathnameFromSlug(locale, slug), locale);
    if (!resolved) {
      return { notFound: true };
    }

    return {
      props: {
        eyebrow: resolved.eyebrow,
        page: resolved.page
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

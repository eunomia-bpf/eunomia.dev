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
import { getCollectionFamilyByKind, getContentEyebrow, type CollectionFamilyId } from "./content/registry";
import { buildSearchSidebar } from "./content/sidebar";
import { searchContent } from "./content/search";
import type { DocsPage, SearchResult, SidebarGroup } from "./content/types";
import type { HomePageData } from "./page-factories";
import type { ContentPageProps } from "./page-builders";
import { localizePath } from "./paths";
import type { Locale } from "./site-data";

type SearchPageProps = {
  query: string;
  results: SearchResult[];
  sidebar: SidebarGroup[];
};

const collectionLoaders: Record<
  CollectionFamilyId,
  {
    loadIndex: (locale: Locale) => Promise<DocsPage>;
    loadPage: (slug: string[] | undefined, locale: Locale) => Promise<DocsPage | null>;
  }
> = {
  tutorial: {
    loadIndex: loadTutorialIndex,
    loadPage: loadTutorialPage
  },
  blog: {
    loadIndex: loadBlogIndex,
    loadPage: loadBlogPage
  },
  "legacy-blog": {
    loadIndex: loadLegacyBlogIndex,
    loadPage: loadLegacyBlogPage
  }
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

    const path = pathnameFromSlug(locale, slug);
    const record = resolveManifestRecordFromRoute(path);
    if (!record) {
      return {
        notFound: true
      };
    }

    const family = getCollectionFamilyByKind(record.kind);
    if (family) {
      const loader = collectionLoaders[family.id];
      const page =
        record.kind === family.indexKind ? await loader.loadIndex(locale) : await loader.loadPage(record.slug, locale);

      if (!page) {
        return { notFound: true };
      }

      return {
        props: {
          eyebrow: family.eyebrow(locale),
          page
        }
      };
    }

    if (record.kind !== "section-page") {
      return {
        notFound: true
      };
    }

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
        eyebrow: getContentEyebrow(record, locale),
        page
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

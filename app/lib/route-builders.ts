import type { GetStaticPaths, GetStaticProps } from "next";

import { loadHomePage, resolveContentPage } from "./content";
import { listRenderableRoutesForLocale } from "./content/manifest";
import { buildSearchSidebar } from "./content/sidebar";
import type { SidebarGroup } from "./content/types";
import type { HomePageData } from "./page-factories";
import type { ContentPageProps } from "./page-builders";
import { localizePath } from "./paths";
import type { Locale } from "./site-data";

type SearchPageProps = {
  sidebar: SidebarGroup[];
};

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
  const homeRoute = pathnameFromSlug(locale, []);

  const getStaticPaths: GetStaticPaths = async () => {
    const paths = listRenderableRoutesForLocale(locale)
      .filter((route) => route !== homeRoute)
      .map((route) => {
        const localizedRoute =
          locale === "zh" && route.startsWith("/zh/")
            ? route.slice("/zh".length)
            : route;

        return {
          params: {
            slug: localizedRoute.replace(/^\/+|\/+$/g, "").split("/").filter(Boolean)
          }
        };
      });

    return {
      paths,
      fallback: false
    };
  };

  const getStaticProps: GetStaticProps<ContentPageProps> = async ({ params }) => {
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
    getStaticPaths,
    getStaticProps
  };
}

export function createSearchPageRoute(locale: Locale) {
  const getStaticProps: GetStaticProps<SearchPageProps> = async () => {
    return {
      props: {
        sidebar: buildSearchSidebar(locale)
      }
    };
  };

  return {
    getStaticProps
  };
}

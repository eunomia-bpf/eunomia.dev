import type { GetStaticPropsContext, GetStaticPropsResult } from "next";

import { ArticleLayout } from "../components/ArticleLayout";
import { CardGrid } from "../components/CardGrid";
import { MarkdownContent } from "../components/MarkdownContent";
import { PageFooter } from "../components/PageFooter";
import { SeoHead } from "../components/SeoHead";
import { SiteChrome } from "../components/SiteChrome";
import { canonicalAlternates } from "./seo";
import type { GitMetadata, LandingCard, LandingPageData, MarkdownPage } from "./content/types";
import type { Locale } from "./site-data";

export type HomePageData = {
  title: string;
  description: string;
  intro: string;
  cards: LandingCard[];
  sourcePath: string;
  metadata?: GitMetadata | null;
  path: string;
  alternates: {
    en: string;
    zh: string;
  };
};

export type CollectionPageProps<IndexPage extends LandingPageData, ArticlePage extends MarkdownPage> =
  | {
      kind: "index";
      page: IndexPage;
    }
  | {
      kind: "article";
      page: ArticlePage;
    };

type CollectionStaticPropsConfig<IndexPage extends LandingPageData, ArticlePage extends MarkdownPage> = {
  loadIndex: () => Promise<IndexPage>;
  loadArticle: (slug: string[]) => Promise<ArticlePage | null>;
};

type CollectionPageViewProps<IndexPage extends LandingPageData, ArticlePage extends MarkdownPage> =
  CollectionPageProps<IndexPage, ArticlePage> & {
    locale: Locale;
    eyebrow: string;
  };

type SectionPageViewProps = {
  page: MarkdownPage;
  section: string;
  locale: Locale;
};

function getTocTitle(locale: Locale): string {
  return locale === "zh" ? "本页目录" : "On this page";
}

function renderCollectionContent<IndexPage extends LandingPageData, ArticlePage extends MarkdownPage>(
  kind: CollectionPageProps<IndexPage, ArticlePage>["kind"],
  page: IndexPage | ArticlePage,
  locale: Locale
) {
  if (kind === "index") {
    const indexPage = page as IndexPage;
    return (
      <>
        <section className="mx-auto max-w-4xl px-5 pb-10">
          <article className="rounded-[2rem] border border-white/70 bg-white/90 p-8 shadow-panel md:p-10">
            <MarkdownContent html={indexPage.introHtml} />
            <PageFooter
              locale={locale}
              title={indexPage.title}
              path={indexPage.path}
              sourceHref={indexPage.sourcePath}
              metadata={indexPage.metadata}
            />
          </article>
        </section>
        <CardGrid cards={indexPage.cards} />
      </>
    );
  }

  const articlePage = page as ArticlePage;
  return (
    <ArticleLayout
      locale={locale}
      path={articlePage.path}
      title={articlePage.title}
      description={articlePage.description}
      sourceHref={articlePage.sourcePath}
      metadata={articlePage.metadata}
      headings={articlePage.headings}
      tocTitle={getTocTitle(locale)}
    >
      <MarkdownContent html={articlePage.html} />
    </ArticleLayout>
  );
}

export function buildSlugStaticPaths(slugs: string[][]) {
  return {
    paths: slugs.map((slug) => ({
      params: {
        slug
      }
    })),
    fallback: "blocking" as const
  };
}

export function buildSectionStaticPaths(routes: Array<{ section: string; slug: string[] }>) {
  return {
    paths: routes.map((route) => ({
      params: {
        section: route.section,
        slug: route.slug
      }
    })),
    fallback: "blocking" as const
  };
}

export async function loadCollectionStaticProps<IndexPage extends LandingPageData, ArticlePage extends MarkdownPage>(
  params: GetStaticPropsContext["params"],
  config: CollectionStaticPropsConfig<IndexPage, ArticlePage>
): Promise<GetStaticPropsResult<CollectionPageProps<IndexPage, ArticlePage>>> {
  const slug = Array.isArray(params?.slug) ? params.slug : [];

  if (!slug.length) {
    return {
      props: {
        kind: "index",
        page: await config.loadIndex()
      }
    };
  }

  const page = await config.loadArticle(slug);
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
}

export async function loadSectionStaticProps(
  params: GetStaticPropsContext["params"],
  loadPage: (section: string, slug: string[]) => Promise<MarkdownPage | null>
): Promise<GetStaticPropsResult<{ page: MarkdownPage; section: string }>> {
  const section = typeof params?.section === "string" ? params.section : "";
  const slug = Array.isArray(params?.slug) ? params.slug : [];
  const page = await loadPage(section, slug);

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
}

export function CollectionPageView<IndexPage extends LandingPageData, ArticlePage extends MarkdownPage>({
  kind,
  page,
  locale,
  eyebrow
}: CollectionPageViewProps<IndexPage, ArticlePage>) {
  return (
    <>
      <SeoHead
        title={page.title}
        description={page.description}
        path={page.path}
        alternates={canonicalAlternates(page.alternates.en, page.alternates.zh)}
        article={kind === "article"}
        metadata={page.metadata}
      />
      <SiteChrome locale={locale} eyebrow={eyebrow} title={page.title} intro={page.description}>
        {renderCollectionContent(kind, page, locale)}
      </SiteChrome>
    </>
  );
}

export function SectionPageView({ page, section, locale }: SectionPageViewProps) {
  return (
    <>
      <SeoHead
        title={page.title}
        description={page.description}
        path={page.path}
        alternates={canonicalAlternates(page.alternates.en, page.alternates.zh)}
        article
        metadata={page.metadata}
      />
      <SiteChrome locale={locale} eyebrow={section} title={page.title} intro={page.description}>
        <ArticleLayout
          locale={locale}
          path={page.path}
          title={page.title}
          description={page.description}
          sourceHref={page.sourcePath}
          metadata={page.metadata}
          headings={page.headings}
          tocTitle={getTocTitle(locale)}
        >
          <MarkdownContent html={page.html} />
        </ArticleLayout>
      </SiteChrome>
    </>
  );
}

export function HomePageView({
  page,
  locale,
  eyebrow
}: {
  page: HomePageData;
  locale: Locale;
  eyebrow: string;
}) {
  return (
    <>
      <SeoHead
        title={page.title}
        description={page.description}
        path={page.path}
        alternates={canonicalAlternates(page.alternates.en, page.alternates.zh)}
        metadata={page.metadata}
      />
      <SiteChrome locale={locale} eyebrow={eyebrow} title={page.title} intro={page.intro}>
        <section className="mx-auto max-w-4xl px-5 pb-10">
          <article className="rounded-[2rem] border border-white/70 bg-white/90 p-8 shadow-panel md:p-10">
            <PageFooter
              locale={locale}
              title={page.title}
              path={page.path}
              sourceHref={page.sourcePath}
              metadata={page.metadata}
            />
          </article>
        </section>
        <CardGrid cards={page.cards} />
      </SiteChrome>
    </>
  );
}

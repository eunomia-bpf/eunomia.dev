import { ArticleLayout } from "../components/ArticleLayout";
import { BlogListing } from "../components/BlogListing";
import { CardGrid } from "../components/CardGrid";
import { HomePageHero, HomePageLanding } from "../components/HomePageLanding";
import { SeoHead } from "../components/SeoHead";
import { SiteChrome } from "../components/SiteChrome";
import { canonicalAlternates } from "./seo";
import { MarkdownContent } from "../components/MarkdownContent";
import type { BlogEntry, DocsPage, GitMetadata, LocaleAlternates } from "./content/types";
import type { Locale } from "./site-data";

export type HomePageData = {
  title: string;
  description: string;
  intro: string;
  sourcePath: string;
  metadata?: GitMetadata | null;
  path: string;
  alternates: LocaleAlternates;
  recentPosts: BlogEntry[];
};

function getTocTitle(locale: Locale): string {
  return locale === "zh" ? "本页目录" : "On this page";
}

function renderDocsBody(page: DocsPage, locale: Locale) {
  // Blog index: render the React blog listing component instead of markdown.
  if (page.blogEntries) {
    return (
      <BlogListing
        title={page.title}
        description={page.description}
        entries={page.blogEntries}
        locale={locale}
      />
    );
  }

  return (
    <ArticleLayout
      locale={locale}
      path={page.path}
      title={page.title}
      description={page.description}
      sourceHref={page.sourcePath}
      metadata={page.metadata}
      headings={page.layout === "document" ? page.headings : []}
      continuation={page.layout === "document" ? page.continuation : undefined}
      tocTitle={getTocTitle(locale)}
      showBreadcrumbs={page.layout === "document"}
    >
      <MarkdownContent html={page.bodyHtml} />
      {page.cards?.length ? (
        <section className="mt-12">
          <CardGrid cards={page.cards} compact />
        </section>
      ) : null}
    </ArticleLayout>
  );
}

export function DocsPageView({
  page,
  locale,
  eyebrow
}: {
  page: DocsPage;
  locale: Locale;
  eyebrow: string;
}) {
  return (
    <>
      <SeoHead
        title={page.title}
        description={page.description}
        path={page.path}
        alternates={canonicalAlternates(page.alternates)}
        article={page.layout === "document"}
        publishedAt={page.date}
        metadata={page.metadata}
      />
      <SiteChrome
        locale={locale}
        eyebrow={eyebrow}
        title={page.title}
        intro={page.description}
        leadMode="none"
        currentPath={page.path}
        sidebar={page.sidebar}
        alternates={page.alternates}
      >
        {renderDocsBody(page, locale)}
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
        alternates={canonicalAlternates(page.alternates)}
        metadata={page.metadata}
      />
      <SiteChrome
        locale={locale}
        eyebrow={eyebrow}
        title={page.title}
        intro={page.intro}
        currentPath={page.path}
        leadMode="none"
        hero={<HomePageHero locale={locale} />}
        alternates={page.alternates}
      >
        <HomePageLanding locale={locale} recentPosts={page.recentPosts} />
      </SiteChrome>
    </>
  );
}

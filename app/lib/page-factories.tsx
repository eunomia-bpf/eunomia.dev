import { ArticleLayout } from "../components/ArticleLayout";
import { AboutLandingPage } from "../components/AboutLandingPage";
import { AgentSystemLayerPresentation } from "../components/AgentSystemLayerPresentation";
import { BlogListing } from "../components/BlogListing";
import { CardGrid } from "../components/CardGrid";
import { DailyReportResearchDirections } from "../components/DailyReportEnhancements";
import { DailyReportFooterNote } from "../components/DailyReportFooterNote";
import { HomePageHero, HomePageLanding } from "../components/HomePageLanding";
import {
  AgentRuntimeInfrastructurePage,
  BpftimeProductPage,
  ProductsLandingPage,
  ServicesProductPage
} from "../components/ProductPages";
import { ProjectLandingPage } from "../components/ProjectLandingPage";
import { SeoHead } from "../components/SeoHead";
import { SiteChrome } from "../components/SiteChrome";
import { canonicalAlternates } from "./seo";
import { MarkdownContent } from "../components/MarkdownContent";
import type {
  BlogEntry,
  DocsPage,
  GitMetadata,
  LocaleAlternates
} from "./content/types";
import type { MkdocsHomeConfig } from "./content/mkdocs-config";
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
  home: MkdocsHomeConfig;
};

function getTocTitle(locale: Locale): string {
  return locale === "zh" ? "本页目录" : "On this page";
}

function getDocsRobots(path: string): string | undefined {
  return /\/(blogs|zh\/blogs)\//.test(path) || path === "/about/" || path === "/zh/about/"
    ? "noindex,follow"
    : undefined;
}

function isDailyReportPath(path: string): boolean {
  return /^\/(?:zh\/)?research(?:\/|$)/.test(path);
}

function getDailyReportLabel(locale: Locale): string {
  return locale === "zh" ? "每日报告" : "Daily Report";
}

function getDailyReportTags(tags: string[] | undefined, locale: Locale): string[] {
  const label = getDailyReportLabel(locale);
  return (tags ?? []).map((tag) => (tag === "Research" ? label : tag));
}

function renderCustomReactPage(kind: NonNullable<DocsPage["reactPage"]>, locale: Locale, page: DocsPage) {
  const links = page.reactLinks ?? [];
  const projects = page.projectCatalog?.projects ?? [];

  if (kind === "products") {
    return <ProductsLandingPage locale={locale} links={links} projects={projects} />;
  }
  if (kind === "bpftime-product") {
    return <BpftimeProductPage locale={locale} links={links} projects={projects} />;
  }
  if (kind === "agent-runtime-infrastructure") {
    return <AgentRuntimeInfrastructurePage locale={locale} links={links} projects={projects} />;
  }
  if (kind === "agent-system-layer-presentation") {
    return <AgentSystemLayerPresentation links={links} />;
  }
  if (kind === "services") {
    return <ServicesProductPage locale={locale} links={links} projects={projects} />;
  }
  return <AboutLandingPage locale={locale} links={links} projects={projects} />;
}

function renderDocsBody(page: DocsPage, locale: Locale) {
  if (page.reactPage) {
    return renderCustomReactPage(page.reactPage, locale, page);
  }

  if (
    page.landingPage &&
    page.projectCatalog &&
    (page.landingPage.variant === "project-index" || page.landingPage.variant === "project-focus")
  ) {
    return (
      <ProjectLandingPage
        landing={page.landingPage}
        projectGroups={page.projectCatalog.projectGroups}
        projects={page.projectCatalog.projects}
        locale={locale}
      />
    );
  }

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

  const dailyReport = isDailyReportPath(page.path);
  const headings = page.layout === "document" ? page.headings : [];

  return (
    <ArticleLayout
      locale={locale}
      path={page.path}
      title={page.title}
      description={page.descriptionIsExcerpt ? "" : page.description}
      tags={dailyReport ? getDailyReportTags(page.tags, locale) : page.tags}
      publishedAt={page.date}
      sourceHref={page.sourcePath}
      metadata={dailyReport ? null : page.metadata}
      headings={headings}
      continuation={page.layout === "document" ? page.continuation : undefined}
      tocTitle={getTocTitle(locale)}
      showBreadcrumbs={page.layout === "document"}
      footerNote={dailyReport ? <DailyReportFooterNote locale={locale} /> : undefined}
    >
      <MarkdownContent html={page.bodyHtml} />
      {dailyReport ? <DailyReportResearchDirections locale={locale} path={page.path} /> : null}
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
  const dailyReport = isDailyReportPath(page.path);

  return (
    <>
      <SeoHead
        title={page.title}
        description={page.description}
        path={page.path}
        alternates={canonicalAlternates(page.alternates)}
        article={page.layout === "document"}
        publishedAt={page.date}
        metadata={dailyReport ? null : page.metadata}
        robots={getDocsRobots(page.path)}
        isTutorial={/\/(tutorials|zh\/tutorials)\//.test(page.path)}
        isCodeProject={/\/(bpftime|eunomia-bpf|GPTtrace)\/?/.test(page.path)}
        repoUrl={
          page.path.includes("/bpftime")
            ? "https://github.com/eunomia-bpf/bpftime"
            : page.path.includes("/eunomia-bpf")
              ? "https://github.com/eunomia-bpf/eunomia-bpf"
              : page.path.includes("/GPTtrace")
                ? "https://github.com/eunomia-bpf/GPTtrace"
                : undefined
        }
      />
      <SiteChrome
        locale={locale}
        eyebrow={dailyReport ? getDailyReportLabel(locale) : eyebrow}
        title={page.title}
        intro={page.description}
        leadMode="none"
        currentPath={page.path}
        sidebar={page.reactPage ? undefined : page.sidebar}
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
        hero={<HomePageHero home={page.home} locale={locale} />}
        alternates={page.alternates}
      >
        <HomePageLanding locale={locale} recentPosts={page.recentPosts} home={page.home} />
      </SiteChrome>
    </>
  );
}

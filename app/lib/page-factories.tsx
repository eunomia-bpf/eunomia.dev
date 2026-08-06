import { ArticleLayout } from "../components/ArticleLayout";
import { AboutLandingPage } from "../components/AboutLandingPage";
import { AgentSystemLayerPresentation } from "../components/AgentSystemLayerPresentation";
import { BlogListing } from "../components/BlogListing";
import { CardGrid } from "../components/CardGrid";
import { DailyReportResearchDirections } from "../components/DailyReportEnhancements";
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
import {
  getDailyReportLabel,
  isDailyReportPath,
  prepareDailyReportPage
} from "./daily-report";
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
  const renderedPage = prepareDailyReportPage(page, locale);

  if (renderedPage.reactPage) {
    return renderCustomReactPage(renderedPage.reactPage, locale, renderedPage);
  }

  if (
    renderedPage.landingPage &&
    renderedPage.projectCatalog &&
    (renderedPage.landingPage.variant === "project-index" ||
      renderedPage.landingPage.variant === "project-focus")
  ) {
    return (
      <ProjectLandingPage
        landing={renderedPage.landingPage}
        projectGroups={renderedPage.projectCatalog.projectGroups}
        projects={renderedPage.projectCatalog.projects}
        locale={locale}
      />
    );
  }

  // Blog index: render the React blog listing component instead of markdown.
  if (renderedPage.blogEntries) {
    return (
      <BlogListing
        title={renderedPage.title}
        description={renderedPage.description}
        entries={renderedPage.blogEntries}
        locale={locale}
      />
    );
  }

  const dailyReport = isDailyReportPath(renderedPage.path);
  const headings = renderedPage.layout === "document" ? renderedPage.headings : [];

  return (
    <ArticleLayout
      locale={locale}
      path={renderedPage.path}
      title={renderedPage.title}
      description={renderedPage.descriptionIsExcerpt ? "" : renderedPage.description}
      tags={renderedPage.tags}
      publishedAt={renderedPage.date}
      sourceHref={renderedPage.sourcePath}
      metadata={renderedPage.metadata}
      headings={headings}
      continuation={renderedPage.layout === "document" ? renderedPage.continuation : undefined}
      tocTitle={getTocTitle(locale)}
      showBreadcrumbs={renderedPage.layout === "document"}
    >
      <MarkdownContent html={renderedPage.bodyHtml} />
      {dailyReport ? (
        <DailyReportResearchDirections locale={locale} path={renderedPage.path} />
      ) : null}
      {renderedPage.cards?.length ? (
        <section className="mt-12">
          <CardGrid cards={renderedPage.cards} compact />
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
  const renderedPage = prepareDailyReportPage(page, locale);
  const dailyReport = isDailyReportPath(renderedPage.path);

  return (
    <>
      <SeoHead
        title={renderedPage.title}
        description={renderedPage.description}
        path={renderedPage.path}
        alternates={canonicalAlternates(renderedPage.alternates)}
        article={renderedPage.layout === "document"}
        publishedAt={renderedPage.date}
        metadata={renderedPage.metadata}
        robots={getDocsRobots(renderedPage.path)}
        isTutorial={/\/(tutorials|zh\/tutorials)\//.test(renderedPage.path)}
        isCodeProject={/\/(bpftime|eunomia-bpf|GPTtrace)\/?/.test(renderedPage.path)}
        repoUrl={
          renderedPage.path.includes("/bpftime")
            ? "https://github.com/eunomia-bpf/bpftime"
            : renderedPage.path.includes("/eunomia-bpf")
              ? "https://github.com/eunomia-bpf/eunomia-bpf"
              : renderedPage.path.includes("/GPTtrace")
                ? "https://github.com/eunomia-bpf/GPTtrace"
                : undefined
        }
      />
      <SiteChrome
        locale={locale}
        eyebrow={dailyReport ? getDailyReportLabel(locale) : eyebrow}
        title={renderedPage.title}
        intro={renderedPage.description}
        leadMode="none"
        currentPath={renderedPage.path}
        sidebar={renderedPage.reactPage ? undefined : renderedPage.sidebar}
        alternates={renderedPage.alternates}
      >
        {renderDocsBody(renderedPage, locale)}
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

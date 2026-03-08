import {
  getFeaturedHomeSections,
  getHomeBadge,
  getHomeExploreSections,
  getHomePath,
  getHomeStats
} from "../site-ia";
import type { Locale } from "../site-data";
import { getBlogEntries, getLegacyBlogEntries, getTutorialReadmeSources } from "./collections";
import { getDocumentBySource, resolveDocument } from "./documents";
import { getGitMetadata } from "./git";
import { resolveManifestRecordFromRoute } from "./manifest";
import { buildCollectionContinuation, buildIndexLink } from "./navigation";
import { resolveAlternatesFromDocSource } from "./manifest";
import { getCollectionFamilyByKind, getContentEyebrow, type CollectionFamilyId } from "./registry";
import {
  buildBlogIndexPath,
  buildBlogPath,
  buildLegacyBlogIndexPath,
  buildLegacyBlogPath,
  buildSectionPath,
  buildTutorialIndexPath,
  buildTutorialPath
} from "./route-paths";
import {
  buildBlogSidebar,
  buildLegacyBlogSidebar,
  buildSectionSidebar,
  buildTutorialSidebar
} from "./sidebar";
import { renderMarkdownBody, renderMarkdownDocumentBody } from "./render";
import {
  formatGithubSourcePath,
  makeAlternates,
  resolveSectionPageSource,
  tutorialSourceToSlugSegments
} from "./source";
import type { DocsPage, LandingCard } from "./types";
import type { HomePageData } from "../page-factories";

export type ResolvedContentPage = {
  page: DocsPage;
  eyebrow: string;
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

function requireDocument(relativePath: string, locale: Locale) {
  const document = resolveDocument(relativePath, locale);
  if (!document) {
    throw new Error(`missing document for ${relativePath} (${locale})`);
  }

  return document;
}

async function loadDocumentPage(relativePath: string, publicPath: string, locale: Locale): Promise<DocsPage> {
  const document = requireDocument(relativePath, locale);
  const sourceRelative = document.sourceRelative;
  const rendered = await renderMarkdownDocumentBody(document.body, sourceRelative, locale);

  return {
    layout: "document",
    title: document.title,
    description: document.description,
    bodyHtml: rendered.html,
    headings: rendered.headings,
    sourcePath: formatGithubSourcePath(sourceRelative),
    metadata: getGitMetadata(sourceRelative),
    path: publicPath,
    alternates: resolveAlternatesFromDocSource(sourceRelative, locale, publicPath)
  };
}

async function loadDirectoryPage({
  sourceRelative,
  publicPath,
  locale,
  cards,
  sidebar
}: {
  sourceRelative: string;
  publicPath: string;
  locale: Locale;
  cards: LandingCard[];
  sidebar?: DocsPage["sidebar"];
}): Promise<DocsPage> {
  const document = getDocumentBySource(sourceRelative) ?? requireDocument(sourceRelative, locale);
  const bodyHtml = await renderMarkdownBody(document.body, sourceRelative, locale);

  return {
    layout: "directory",
    title: document.title,
    description: document.description,
    bodyHtml,
    sourcePath: formatGithubSourcePath(sourceRelative),
    metadata: getGitMetadata(sourceRelative),
    path: publicPath,
    alternates: resolveAlternatesFromDocSource(sourceRelative, locale, publicPath),
    cards,
    sidebar
  };
}

function withContinuation(page: DocsPage, continuation: DocsPage["continuation"]): DocsPage {
  if (!continuation) {
    return page;
  }

  return {
    ...page,
    continuation
  };
}

export async function loadHomePage(locale: Locale): Promise<HomePageData> {
  const home = requireDocument("index.md", locale);
  const counts = {
    tutorials: getTutorialReadmeSources().length,
    blogs: getBlogEntries().length,
    legacyBlogs: getLegacyBlogEntries().length
  };
  const cards: LandingCard[] = getFeaturedHomeSections().map((section) => {
    const document = requireDocument(section.source, locale);
    return {
      title: document.title,
      description: document.description,
      href: section.href(locale),
      badge: getHomeBadge(section.key, counts, locale)
    };
  });
  const moreLinks: LandingCard[] = getHomeExploreSections().map((section) => {
    const document = requireDocument(section.indexSource, locale);
    return {
      title: document.title,
      description: document.description,
      href: section.href(locale)
    };
  });

  const latestBlog = getBlogEntries()[0];

  return {
    title: home.title,
    description: home.description,
    intro: home.excerpt || home.description,
    cards,
    moreLinks,
    stats: getHomeStats(locale, counts),
    spotlight: latestBlog
      ? {
          title: latestBlog.title,
          description: latestBlog.excerpt || latestBlog.description,
          href: buildBlogPath(latestBlog.year, latestBlog.month, latestBlog.day, latestBlog.slug, locale),
          badge: `${latestBlog.year}-${latestBlog.month}-${latestBlog.day}`
        }
      : {
          title: locale === "zh" ? "查看 Blog" : "Visit the blog",
          description: home.description,
          href: buildBlogIndexPath(locale),
          badge: locale === "zh" ? "博客" : "Blog"
        },
    sourcePath: formatGithubSourcePath("index.md"),
    metadata: getGitMetadata("index.md"),
    path: getHomePath(locale),
    alternates: makeAlternates(getHomePath(locale))
  };
}

export async function loadTutorialIndex(locale: Locale): Promise<DocsPage> {
  const sourceRelative = requireDocument("tutorials/index.md", locale).sourceRelative;
  const cards = getTutorialReadmeSources().map((source) => {
    const tutorial = requireDocument(source, locale);
    const slugSegments = tutorialSourceToSlugSegments(source);

    return {
      title: tutorial.title,
      description: tutorial.description,
      href: buildTutorialPath(slugSegments, locale),
      badge: slugSegments.join("/")
    };
  });

  return loadDirectoryPage({
    sourceRelative,
    publicPath: buildTutorialIndexPath(locale),
    locale,
    cards,
    sidebar: buildTutorialSidebar(locale)
  });
}

export async function loadTutorialPage(
  slugSegments: string[] | undefined,
  locale: Locale
): Promise<DocsPage | null> {
  if (!slugSegments?.length) {
    return null;
  }

  const baseCandidate = `tutorials/${slugSegments.join("/")}`;
  const sourceRelative =
    resolveDocument(`${baseCandidate}.md`, locale)?.sourceRelative ??
    resolveDocument(`${baseCandidate}/README.md`, locale)?.sourceRelative;

  if (!sourceRelative) {
    return null;
  }

  const publicPath = buildTutorialPath(slugSegments, locale);
  const page = await loadDocumentPage(sourceRelative, publicPath, locale);
  const continuation = buildCollectionContinuation({
    kind: "tutorial-page",
    locale,
    currentPath: publicPath,
    index: buildIndexLink(requireDocument("tutorials/index.md", locale).sourceRelative, buildTutorialIndexPath(locale))
  });

  return {
    ...withContinuation(page, continuation),
    sidebar: buildTutorialSidebar(locale)
  };
}

export async function loadBlogIndex(locale: Locale): Promise<DocsPage> {
  const sourceRelative = requireDocument("blog/index.md", locale).sourceRelative;
  const cards = getBlogEntries().map((entry) => ({
    title: entry.title,
    description: entry.description,
    href: buildBlogPath(entry.year, entry.month, entry.day, entry.slug, locale),
    badge: `${entry.year}-${entry.month}-${entry.day}`
  }));

  return loadDirectoryPage({
    sourceRelative,
    publicPath: buildBlogIndexPath(locale),
    locale,
    cards,
    sidebar: buildBlogSidebar(locale)
  });
}

export async function loadBlogPage(
  slugSegments: string[] | undefined,
  locale: Locale
): Promise<DocsPage | null> {
  if (!slugSegments || slugSegments.length !== 4) {
    return null;
  }

  const [year, month, day, slug] = slugSegments;
  const entry = getBlogEntries().find(
    (candidate) =>
      candidate.year === year &&
      candidate.month === month &&
      candidate.day === day &&
      candidate.slug === slug
  );

  if (!entry) {
    return null;
  }

  const sourceRelative = entry.sourceByLocale[locale] ?? entry.sourceByLocale.en ?? entry.sourceByLocale.zh;
  if (!sourceRelative) {
    return null;
  }

  const publicPath = buildBlogPath(entry.year, entry.month, entry.day, entry.slug, locale);
  const page = await loadDocumentPage(sourceRelative, publicPath, locale);
  const continuation = buildCollectionContinuation({
    kind: "blog-page",
    locale,
    currentPath: publicPath,
    index: buildIndexLink(requireDocument("blog/index.md", locale).sourceRelative, buildBlogIndexPath(locale))
  });

  return {
    ...withContinuation(page, continuation),
    sidebar: buildBlogSidebar(locale)
  };
}

export async function loadLegacyBlogIndex(locale: Locale): Promise<DocsPage> {
  const sourceRelative = requireDocument("blogs/index.md", locale).sourceRelative;
  const cards = getLegacyBlogEntries().map((entry) => ({
    title: entry.title,
    description: entry.description,
    href: buildLegacyBlogPath(entry.key, locale),
    badge: "Legacy"
  }));

  return loadDirectoryPage({
    sourceRelative,
    publicPath: buildLegacyBlogIndexPath(locale),
    locale,
    cards,
    sidebar: buildLegacyBlogSidebar(locale)
  });
}

export async function loadLegacyBlogPage(
  slugSegments: string[] | undefined,
  locale: Locale
): Promise<DocsPage | null> {
  if (!slugSegments || slugSegments.length !== 1) {
    return null;
  }

  const [slug] = slugSegments;
  const entry = getLegacyBlogEntries().find((candidate) => candidate.key === slug);
  if (!entry) {
    return null;
  }

  const sourceRelative = entry.sourceByLocale[locale] ?? entry.sourceByLocale.en ?? entry.sourceByLocale.zh;
  if (!sourceRelative) {
    return null;
  }

  const publicPath = buildLegacyBlogPath(slug, locale);
  const page = await loadDocumentPage(sourceRelative, publicPath, locale);
  const continuation = buildCollectionContinuation({
    kind: "legacy-blog-page",
    locale,
    currentPath: publicPath,
    index: buildIndexLink(
      requireDocument("blogs/index.md", locale).sourceRelative,
      buildLegacyBlogIndexPath(locale)
    )
  });

  return {
    ...withContinuation(page, continuation),
    sidebar: buildLegacyBlogSidebar(locale)
  };
}

export async function loadSectionPage(
  section: string,
  slugSegments: string[] | undefined,
  locale: Locale
): Promise<DocsPage | null> {
  const sourceRelative = resolveSectionPageSource(section, slugSegments, locale);

  if (!sourceRelative) {
    return null;
  }

  const publicPath = buildSectionPath(section, slugSegments ?? [], locale);
  const page = await loadDocumentPage(sourceRelative, publicPath, locale);
  const sectionIndexSource =
    resolveDocument(`${section}/index.md`, locale)?.sourceRelative ??
    resolveDocument(`${section}/README.md`, locale)?.sourceRelative;
  const sectionIndexHref = buildSectionPath(section, [], locale);
  const continuation = buildCollectionContinuation({
    kind: "section-page",
    locale,
    currentPath: publicPath,
    section,
    index: buildIndexLink(sectionIndexSource ?? null, sectionIndexHref)
  });

  return {
    ...withContinuation(page, continuation),
    sidebar: buildSectionSidebar(section, locale)
  };
}

export async function resolveContentPage(path: string, locale: Locale): Promise<ResolvedContentPage | null> {
  const record = resolveManifestRecordFromRoute(path);
  if (!record) {
    return null;
  }

  const family = getCollectionFamilyByKind(record.kind);
  if (family) {
    const loader = collectionLoaders[family.id];
    const page =
      record.kind === family.indexKind ? await loader.loadIndex(locale) : await loader.loadPage(record.slug, locale);

    if (!page) {
      return null;
    }

    return {
      eyebrow: family.eyebrow(locale),
      page
    };
  }

  if (record.kind !== "section-page") {
    return null;
  }

  const section = record.section ?? "";
  if (!section) {
    return null;
  }

  const page = await loadSectionPage(section, record.slug ?? [], locale);
  if (!page) {
    return null;
  }

  return {
    eyebrow: getContentEyebrow(record, locale),
    page
  };
}

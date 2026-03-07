import type { Locale } from "../site-data";
import {
  getBlogEntries,
  getLegacyBlogEntries,
  getTutorialReadmeSources
} from "./collections";
import { getGitMetadata } from "./git";
import { parseMarkdown } from "./markdown";
import { buildCollectionContinuation, buildIndexLink } from "./navigation";
import { resolveAlternatesFromDocSource } from "./manifest";
import { renderMarkdownBody, renderMarkdownDocumentBody } from "./render";
import {
  formatGithubSourcePath,
  makeAlternates,
  resolveLocalizedSource,
  resolveSectionPageSource,
  tutorialSourceToSlugSegments
} from "./source";
import type { LandingCard, LandingPageData, MarkdownPage } from "./types";
import type { HomePageData } from "../page-factories";

async function loadMarkdownPage(relativePath: string, publicPath: string, locale: Locale): Promise<MarkdownPage> {
  const sourceRelative = resolveLocalizedSource(relativePath, locale) ?? relativePath;
  const parsed = parseMarkdown(sourceRelative);
  const rendered = await renderMarkdownDocumentBody(parsed.body, sourceRelative, locale);

  return {
    title: parsed.title,
    description: parsed.description,
    html: rendered.html,
    headings: rendered.headings,
    sourcePath: formatGithubSourcePath(sourceRelative),
    metadata: getGitMetadata(sourceRelative),
    path: publicPath,
    alternates: resolveAlternatesFromDocSource(sourceRelative, locale, publicPath)
  };
}

function withContinuation(page: MarkdownPage, continuation: MarkdownPage["continuation"]): MarkdownPage {
  if (!continuation) {
    return page;
  }

  return {
    ...page,
    continuation
  };
}

export async function loadHomePage(locale: Locale): Promise<HomePageData> {
  const home = parseMarkdown("index.md");
  const tutorials = parseMarkdown(resolveLocalizedSource("tutorials/index.md", locale) ?? "tutorials/index.md");
  const bpftime = parseMarkdown(resolveLocalizedSource("bpftime/index.md", locale) ?? "bpftime/index.md");
  const eunomiaBpf = parseMarkdown(
    resolveLocalizedSource("eunomia-bpf/index.md", locale) ?? "eunomia-bpf/index.md"
  );

  const cards: LandingCard[] = [
    {
      title: tutorials.title,
      description: tutorials.description,
      href: locale === "zh" ? "/zh/tutorials/" : "/tutorials/",
      badge: `${getTutorialReadmeSources().length} walkthroughs`
    },
    {
      title: bpftime.title,
      description: bpftime.description,
      href: locale === "zh" ? "/zh/bpftime/" : "/bpftime/",
      badge: "Userspace eBPF"
    },
    {
      title: eunomiaBpf.title,
      description: eunomiaBpf.description,
      href: locale === "zh" ? "/zh/eunomia-bpf/" : "/eunomia-bpf/",
      badge: "Toolchain"
    }
  ];
  const latestBlog = getBlogEntries()[0];
  const stats =
    locale === "zh"
      ? [
          {
            label: "教程",
            value: String(getTutorialReadmeSources().length),
            detail: "从基础 trace 到更深的 kernel/runtime 主题。"
          },
          {
            label: "研究文章",
            value: String(getBlogEntries().length),
            detail: "持续整理 eBPF、GPU、AI agent 和系统研究。"
          },
          {
            label: "旧博客",
            value: String(getLegacyBlogEntries().length),
            detail: "旧 `/blogs/*` 路径仍然保留并可访问。"
          }
        ]
      : [
          {
            label: "Tutorials",
            value: String(getTutorialReadmeSources().length),
            detail: "Hands-on walkthroughs from basic tracing to deeper kernel and runtime topics."
          },
          {
            label: "Research posts",
            value: String(getBlogEntries().length),
            detail: "Ongoing writing on eBPF, GPU tooling, AI agents, and systems research."
          },
          {
            label: "Legacy posts",
            value: String(getLegacyBlogEntries().length),
            detail: "The old `/blogs/*` paths remain live while the app cuts over."
          }
        ];

  return {
    title: home.title,
    description: home.description,
    intro: home.excerpt || home.description,
    cards,
    stats,
    spotlight: latestBlog
      ? {
          title: latestBlog.title,
          description: latestBlog.excerpt || latestBlog.description,
          href:
            locale === "zh"
              ? `/zh/blog/${latestBlog.year}/${latestBlog.month}/${latestBlog.day}/${latestBlog.slug}/`
              : `/blog/${latestBlog.year}/${latestBlog.month}/${latestBlog.day}/${latestBlog.slug}/`,
          badge: `${latestBlog.year}-${latestBlog.month}-${latestBlog.day}`
        }
      : {
          title: locale === "zh" ? "查看 Blog" : "Visit the blog",
          description: home.description,
          href: locale === "zh" ? "/zh/blog/" : "/blog/",
          badge: locale === "zh" ? "博客" : "Blog"
        },
    sourcePath: formatGithubSourcePath("index.md"),
    metadata: getGitMetadata("index.md"),
    path: locale === "zh" ? "/zh/" : "/",
    alternates: makeAlternates(locale === "zh" ? "/zh/" : "/")
  };
}

export async function loadTutorialIndex(locale: Locale): Promise<LandingPageData> {
  const sourceRelative = resolveLocalizedSource("tutorials/index.md", locale) ?? "tutorials/index.md";
  const parsed = parseMarkdown(sourceRelative);
  const introHtml = await renderMarkdownBody(parsed.body, sourceRelative, locale);

  const cards = getTutorialReadmeSources().map((source) => {
    const localizedSource = resolveLocalizedSource(source, locale) ?? source;
    const tutorial = parseMarkdown(localizedSource);
    const slugSegments = tutorialSourceToSlugSegments(source);
    const href = locale === "zh" ? `/zh/tutorials/${slugSegments.join("/")}/` : `/tutorials/${slugSegments.join("/")}/`;

    return {
      title: tutorial.title,
      description: tutorial.description,
      href,
      badge: slugSegments.join("/")
    };
  });

  return {
    title: parsed.title,
    description: parsed.description,
    introHtml,
    sourcePath: formatGithubSourcePath(sourceRelative),
    metadata: getGitMetadata(sourceRelative),
    path: locale === "zh" ? "/zh/tutorials/" : "/tutorials/",
    alternates: makeAlternates(locale === "zh" ? "/zh/tutorials/" : "/tutorials/"),
    cards
  };
}

export async function loadTutorialPage(
  slugSegments: string[] | undefined,
  locale: Locale
): Promise<MarkdownPage | null> {
  if (!slugSegments?.length) {
    return null;
  }

  const baseCandidate = `tutorials/${slugSegments.join("/")}`;
  const sourceRelative =
    resolveLocalizedSource(`${baseCandidate}.md`, locale) ??
    resolveLocalizedSource(`${baseCandidate}/README.md`, locale);

  if (!sourceRelative) {
    return null;
  }

  const publicPath = locale === "zh" ? `/zh/tutorials/${slugSegments.join("/")}/` : `/tutorials/${slugSegments.join("/")}/`;
  const page = await loadMarkdownPage(sourceRelative, publicPath, locale);
  const indexSource = resolveLocalizedSource("tutorials/index.md", locale) ?? "tutorials/index.md";
  const continuation = buildCollectionContinuation({
    kind: "tutorial-page",
    locale,
    currentPath: publicPath,
    index: buildIndexLink(indexSource, locale === "zh" ? "/zh/tutorials/" : "/tutorials/")
  });

  return withContinuation(page, continuation);
}

export async function loadBlogIndex(locale: Locale): Promise<LandingPageData> {
  const sourceRelative = resolveLocalizedSource("blog/index.md", locale) ?? "blog/index.md";
  const parsed = parseMarkdown(sourceRelative);
  const introHtml = await renderMarkdownBody(parsed.body, sourceRelative, locale);

  const cards = getBlogEntries().map((entry) => ({
    title: entry.title,
    description: entry.description,
    href:
      locale === "zh"
        ? `/zh/blog/${entry.year}/${entry.month}/${entry.day}/${entry.slug}/`
        : `/blog/${entry.year}/${entry.month}/${entry.day}/${entry.slug}/`,
    badge: `${entry.year}-${entry.month}-${entry.day}`
  }));

  return {
    title: parsed.title,
    description: parsed.description,
    introHtml,
    sourcePath: formatGithubSourcePath(sourceRelative),
    metadata: getGitMetadata(sourceRelative),
    path: locale === "zh" ? "/zh/blog/" : "/blog/",
    alternates: makeAlternates(locale === "zh" ? "/zh/blog/" : "/blog/"),
    cards
  };
}

export async function loadBlogPage(
  slugSegments: string[] | undefined,
  locale: Locale
): Promise<MarkdownPage | null> {
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

  const sourceRelative =
    entry.sourceByLocale[locale] ?? entry.sourceByLocale.en ?? entry.sourceByLocale.zh;
  if (!sourceRelative) {
    return null;
  }

  const publicPath =
    locale === "zh"
      ? `/zh/blog/${entry.year}/${entry.month}/${entry.day}/${entry.slug}/`
      : `/blog/${entry.year}/${entry.month}/${entry.day}/${entry.slug}/`;

  const page = await loadMarkdownPage(sourceRelative, publicPath, locale);
  const indexSource = resolveLocalizedSource("blog/index.md", locale) ?? "blog/index.md";
  const continuation = buildCollectionContinuation({
    kind: "blog-page",
    locale,
    currentPath: publicPath,
    index: buildIndexLink(indexSource, locale === "zh" ? "/zh/blog/" : "/blog/")
  });

  return withContinuation(page, continuation);
}

export async function loadLegacyBlogIndex(locale: Locale): Promise<LandingPageData> {
  const sourceRelative = resolveLocalizedSource("blogs/index.md", locale) ?? "blogs/index.md";
  const parsed = parseMarkdown(sourceRelative);
  const introHtml = await renderMarkdownBody(parsed.body, sourceRelative, locale);

  const cards = getLegacyBlogEntries().map((entry) => ({
    title: entry.title,
    description: entry.description,
    href: locale === "zh" ? `/zh/blogs/${entry.key}/` : `/blogs/${entry.key}/`,
    badge: "Legacy"
  }));

  return {
    title: parsed.title,
    description: parsed.description,
    introHtml,
    sourcePath: formatGithubSourcePath(sourceRelative),
    metadata: getGitMetadata(sourceRelative),
    path: locale === "zh" ? "/zh/blogs/" : "/blogs/",
    alternates: makeAlternates(locale === "zh" ? "/zh/blogs/" : "/blogs/"),
    cards
  };
}

export async function loadLegacyBlogPage(
  slugSegments: string[] | undefined,
  locale: Locale
): Promise<MarkdownPage | null> {
  if (!slugSegments || slugSegments.length !== 1) {
    return null;
  }

  const [slug] = slugSegments;
  const entry = getLegacyBlogEntries().find((candidate) => candidate.key === slug);
  if (!entry) {
    return null;
  }

  const sourceRelative =
    entry.sourceByLocale[locale] ?? entry.sourceByLocale.en ?? entry.sourceByLocale.zh;
  if (!sourceRelative) {
    return null;
  }

  const publicPath = locale === "zh" ? `/zh/blogs/${slug}/` : `/blogs/${slug}/`;
  const page = await loadMarkdownPage(sourceRelative, publicPath, locale);
  const indexSource = resolveLocalizedSource("blogs/index.md", locale) ?? "blogs/index.md";
  const continuation = buildCollectionContinuation({
    kind: "legacy-blog-page",
    locale,
    currentPath: publicPath,
    index: buildIndexLink(indexSource, locale === "zh" ? "/zh/blogs/" : "/blogs/")
  });

  return withContinuation(page, continuation);
}

export async function loadSectionPage(
  section: string,
  slugSegments: string[] | undefined,
  locale: Locale
): Promise<MarkdownPage | null> {
  const joined = slugSegments?.join("/");
  const sourceRelative = resolveSectionPageSource(section, slugSegments, locale);

  if (!sourceRelative) {
    return null;
  }

  const suffix = joined ? `/${joined}/` : "/";
  const publicPath = locale === "zh" ? `/zh/${section}${suffix}` : `/${section}${suffix}`;
  const page = await loadMarkdownPage(sourceRelative, publicPath, locale);
  const sectionIndexSource =
    resolveLocalizedSource(`${section}/index.md`, locale) ?? resolveLocalizedSource(`${section}/README.md`, locale);
  const sectionIndexHref = locale === "zh" ? `/zh/${section}/` : `/${section}/`;
  const continuation = buildCollectionContinuation({
    kind: "section-page",
    locale,
    currentPath: publicPath,
    section,
    index: buildIndexLink(sectionIndexSource, sectionIndexHref)
  });

  return withContinuation(page, continuation);
}

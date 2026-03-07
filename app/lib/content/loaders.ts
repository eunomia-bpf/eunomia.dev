import type { Locale } from "../site-data";
import {
  getBlogEntries,
  getLegacyBlogEntries,
  getTutorialReadmeSources
} from "./collections";
import { parseMarkdown } from "./markdown";
import { renderMarkdown } from "./render";
import {
  formatGithubSourcePath,
  makeAlternates,
  resolveLocalizedSource,
  tutorialSourceToSlugSegments
} from "./source";
import type { LandingCard, LandingPageData, MarkdownPage } from "./types";

async function loadMarkdownPage(relativePath: string, publicPath: string, locale: Locale): Promise<MarkdownPage> {
  const sourceRelative = resolveLocalizedSource(relativePath, locale) ?? relativePath;
  const parsed = parseMarkdown(sourceRelative);

  return {
    title: parsed.title,
    description: parsed.description,
    html: await renderMarkdown(sourceRelative, locale),
    sourcePath: formatGithubSourcePath(sourceRelative),
    path: publicPath,
    alternates: makeAlternates(publicPath)
  };
}

export async function loadHomePage(locale: Locale): Promise<{
  title: string;
  description: string;
  intro: string;
  cards: LandingCard[];
  path: string;
  alternates: {
    en: string;
    zh: string;
  };
}> {
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

  return {
    title: home.title,
    description: home.description,
    intro: home.excerpt || home.description,
    cards,
    path: locale === "zh" ? "/zh/" : "/",
    alternates: makeAlternates(locale === "zh" ? "/zh/" : "/")
  };
}

export async function loadTutorialIndex(locale: Locale): Promise<LandingPageData> {
  const sourceRelative = resolveLocalizedSource("tutorials/index.md", locale) ?? "tutorials/index.md";
  const parsed = parseMarkdown(sourceRelative);
  const introHtml = await renderMarkdown(sourceRelative, locale);

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
  return loadMarkdownPage(sourceRelative, publicPath, locale);
}

export async function loadBlogIndex(locale: Locale): Promise<LandingPageData> {
  const sourceRelative = resolveLocalizedSource("blog/index.md", locale) ?? "blog/index.md";
  const parsed = parseMarkdown(sourceRelative);
  const introHtml = await renderMarkdown(sourceRelative, locale);

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

  return loadMarkdownPage(sourceRelative, publicPath, locale);
}

export async function loadLegacyBlogIndex(locale: Locale): Promise<LandingPageData> {
  const sourceRelative = resolveLocalizedSource("blogs/index.md", locale) ?? "blogs/index.md";
  const parsed = parseMarkdown(sourceRelative);
  const introHtml = await renderMarkdown(sourceRelative, locale);

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
  return loadMarkdownPage(sourceRelative, publicPath, locale);
}

export async function loadSectionPage(
  section: string,
  slugSegments: string[] | undefined,
  locale: Locale
): Promise<MarkdownPage | null> {
  const joined = slugSegments?.join("/");
  const sourceRelative = joined
    ? resolveLocalizedSource(`${section}/${joined}.md`, locale) ??
      resolveLocalizedSource(`${section}/${joined}/README.md`, locale) ??
      resolveLocalizedSource(`${section}/${joined}/index.md`, locale)
    : resolveLocalizedSource(`${section}/index.md`, locale) ?? resolveLocalizedSource(`${section}/README.md`, locale);

  if (!sourceRelative) {
    return null;
  }

  const suffix = joined ? `/${joined}/` : "/";
  const publicPath = locale === "zh" ? `/zh/${section}${suffix}` : `/${section}${suffix}`;
  return loadMarkdownPage(sourceRelative, publicPath, locale);
}

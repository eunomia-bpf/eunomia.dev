export type Locale = "en" | "zh";

export type NavItem = {
  label: string;
  href: string;
};

export type TutorialArticle = {
  slug: string;
  title: string;
  description: string;
  summary: string;
  sourcePath: string;
  body: string[];
};

export type BlogPost = {
  year: string;
  month: string;
  day: string;
  slug: string;
  title: string;
  description: string;
  excerpt: string;
  sourcePath: string;
  body: string[];
};

export const siteConfig = {
  name: "eunomia",
  siteUrl: process.env.NEXT_PUBLIC_SITE_URL ?? "https://eunomia.dev",
  analyticsId: "G-1YVMXGL0MY",
  repoUrl: "https://github.com/eunomia-bpf/eunomia.dev",
  ogImage: "https://eunomia.dev/assets/icon.svg"
};

export const navByLocale: Record<Locale, NavItem[]> = {
  en: [
    { label: "Tutorials", href: "/tutorials/" },
    { label: "Blog", href: "/blog/" },
    { label: "bpftime", href: "/bpftime/" },
    { label: "eunomia-bpf", href: "/eunomia-bpf/" },
    { label: "Ecosystem", href: "/others/" }
  ],
  zh: [
    { label: "教程", href: "/zh/tutorials/" },
    { label: "博客", href: "/zh/blog/" },
    { label: "文档", href: "/zh/bpftime/" },
    { label: "eunomia-bpf", href: "/zh/eunomia-bpf/" },
    { label: "生态", href: "/zh/others/" }
  ]
};

export const homeSections = [
  {
    title: "Learn eBPF by examples",
    description:
      "Practice-driven tutorials for tracing, networking, security, and GPU observability.",
    href: "/tutorials/",
    badge: "48+ walkthroughs"
  },
  {
    title: "Build with bpftime",
    description:
      "Userspace eBPF runtime work with lower overhead Uprobe, Syscall, and extension points.",
    href: "/bpftime/",
    badge: "OSDI 2025"
  },
  {
    title: "Ship with eunomia-bpf",
    description:
      "Package, distribute, and run eBPF tools with JSON and WebAssembly-friendly workflows.",
    href: "/eunomia-bpf/",
    badge: "CO-RE first"
  }
];

export const tutorialArticles: TutorialArticle[] = [
  {
    slug: "0-introduce",
    title: "eBPF Tutorial by Example 0: Introduction to Core Concepts and Tools",
    description:
      "A quick orientation to the eunomia learning path, the toolchain, and the layout of the tutorial collection.",
    summary: "Start here to understand the tutorial structure and the learning path.",
    sourcePath: "docs/tutorials/0-introduce/README.md",
    body: [
      "This compatibility-first frontend keeps the tutorial entry point obvious: start with the concepts, then move directly into runnable examples.",
      "The long-term migration target is to render tutorial metadata from the existing Markdown sources instead of duplicating content in code."
    ]
  },
  {
    slug: "1-helloworld",
    title: "eBPF Tutorial by Example 1: Hello World, Framework and Development",
    description:
      "The smallest useful starting point for running an eBPF tutorial with the eunomia toolchain.",
    summary: "Build a first runnable eBPF example and understand the minimum project shape.",
    sourcePath: "docs/tutorials/1-helloworld/README.md",
    body: [
      "This first implementation focuses on route and SEO compatibility. The final migration will replace these summary pages with Markdown-backed rendering from the source tutorial repository.",
      "For now, this route proves the new frontend can serve static article pages, expose edit links, and preserve stable URLs for tutorials."
    ]
  }
];

export const blogPosts: BlogPost[] = [
  {
    year: "2026",
    month: "02",
    day: "17",
    slug: "agentcgroup-what-happens-when-ai-coding-agents-meet-os-resources",
    title: "Agentcgroup: What Happens When AI Coding Agents Meet OS Resources?",
    description:
      "A compatibility placeholder for the blog route, focused on preserving dated URLs, metadata, and edit-link behavior during the frontend migration.",
    excerpt:
      "This route proves the custom frontend can preserve the dated blog URL shape used by eunomia.dev today.",
    sourcePath: "docs/blog/posts/agentcgroup-characterization.md",
    body: [
      "The goal of this initial custom frontend is not to finish the content migration in one step. It is to prove that the new stack can preserve the public shape of the website without sacrificing crawlability or metadata quality.",
      "This blog route keeps the existing date-based permalink format, emits canonical and alternate tags, and links back to the source Markdown path in GitHub."
    ]
  }
];

export const pageSummaries = {
  en: {
    bpftime: {
      title: "bpftime",
      description:
        "Userspace eBPF runtime documentation, positioning, and migration compatibility page."
    },
    eunomiaBpf: {
      title: "eunomia-bpf",
      description:
        "Packaging and distribution workflow for eBPF applications, preserved as a stable landing route."
    },
    others: {
      title: "Ecosystem",
      description:
        "Projects, talks, and surrounding ecosystem references collected under a stable route."
    },
    blog: {
      title: "Blog",
      description:
        "Research notes, release updates, and technical writing collected under the stable blog path."
    },
    tutorials: {
      title: "Tutorials",
      description:
        "The practical tutorial collection that anchors the eunomia learning experience."
    }
  },
  zh: {
    bpftime: {
      title: "bpftime 文档",
      description: "保留 `bpftime` 路由与基础 SEO 的中文入口页。"
    },
    eunomiaBpf: {
      title: "eunomia-bpf 文档",
      description: "保留 `eunomia-bpf` 路由与基础 SEO 的中文入口页。"
    },
    others: {
      title: "生态",
      description: "保留生态内容入口与基础 SEO 的中文页。"
    },
    blog: {
      title: "博客",
      description: "保留博客列表、日期路径与基础元数据能力的中文入口。"
    },
    tutorials: {
      title: "教程",
      description: "保留教程入口、文章链接与语言路径的中文页。"
    }
  }
};

export function tutorialPath(slug: string): string {
  return `/tutorials/${slug}/`;
}

export function blogPath(post: BlogPost): string {
  return `/blog/${post.year}/${post.month}/${post.day}/${post.slug}/`;
}

export function localizedPath(path: string, locale: Locale): string {
  if (locale === "en") {
    return path;
  }
  return path === "/" ? "/zh/" : `/zh${path}`;
}

export function getTutorial(slug: string): TutorialArticle | undefined {
  return tutorialArticles.find((article) => article.slug === slug);
}

export function getBlogPost(slug: string): BlogPost | undefined {
  return blogPosts.find((post) => post.slug === slug);
}

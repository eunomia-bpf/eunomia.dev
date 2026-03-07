export type Locale = "en" | "zh";

export type NavItem = {
  label: string;
  href: string;
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
    { label: "eBPF×AI/LLMs", href: "/GPTtrace/" },
    { label: "eunomia-bpf", href: "/eunomia-bpf/" },
    { label: "Ecosystem", href: "/others/" }
  ],
  zh: [
    { label: "教程", href: "/zh/tutorials/" },
    { label: "博客", href: "/zh/blog/" },
    { label: "bpftime", href: "/zh/bpftime/" },
    { label: "eBPF×AI/LLMs", href: "/zh/GPTtrace/" },
    { label: "eunomia-bpf", href: "/zh/eunomia-bpf/" },
    { label: "生态", href: "/zh/others/" }
  ]
};

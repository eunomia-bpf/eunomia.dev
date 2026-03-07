import { navByLocale, siteConfig, type Locale } from "../lib/site-data";

type SiteFooterProps = {
  locale: Locale;
};

export function SiteFooter({ locale }: SiteFooterProps) {
  const copy =
    locale === "zh"
      ? {
          explore: "浏览",
          projects: "项目",
          community: "社区",
          copyright: "eunomia.dev 站点前端，保留文档、教程和旧链接兼容。"
        }
      : {
          explore: "Explore",
          projects: "Projects",
          community: "Community",
          copyright: "eunomia.dev frontend preserving docs, tutorials, and legacy link compatibility."
        };
  const exploreLinks = [...navByLocale[locale], { label: locale === "zh" ? "旧博客" : "Legacy blog", href: locale === "zh" ? "/zh/blogs/" : "/blogs/" }];
  const projectLinks = [
    { label: "GPTtrace", href: locale === "zh" ? "/zh/GPTtrace/" : "/GPTtrace/" },
    { label: "wasm-bpf", href: locale === "zh" ? "/zh/wasm-bpf/" : "/wasm-bpf/" },
    { label: "RSS", href: locale === "zh" ? "/zh/feed.xml" : "/feed.xml" }
  ];
  const communityLinks = [
    { label: "GitHub", href: siteConfig.repoUrl },
    { label: "Discussions", href: "https://github.com/orgs/eunomia-bpf/discussions" }
  ];

  return (
    <footer className="mt-16 border-t border-slate-200 bg-white/80">
      <div className="mx-auto grid max-w-6xl gap-8 px-5 py-10 text-sm text-slate-600 md:grid-cols-[minmax(0,1.3fr)_repeat(3,minmax(0,1fr))]">
        <div>
          <p className="text-lg font-semibold tracking-tight text-ink">eunomia</p>
          <p className="mt-3 max-w-md leading-7">{copy.copyright}</p>
        </div>
        <FooterColumn title={copy.explore} links={exploreLinks} />
        <FooterColumn title={copy.projects} links={projectLinks} />
        <FooterColumn title={copy.community} links={communityLinks} />
      </div>
    </footer>
  );
}

function FooterColumn({
  title,
  links
}: {
  title: string;
  links: Array<{ label: string; href: string }>;
}) {
  return (
    <div>
      <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">{title}</p>
      <ul className="mt-4 space-y-3">
        {links.map((link) => (
          <li key={`${title}:${link.href}`}>
            <a href={link.href} className="transition hover:text-azure">
              {link.label}
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
}

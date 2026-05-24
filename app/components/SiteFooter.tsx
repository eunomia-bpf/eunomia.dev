import { localizePath } from "../lib/paths";
import { getFooterExploreSections, getFooterProjectSections } from "../lib/site-ia";
import { siteConfig, type Locale } from "../lib/site-data";
import { siteFooterCopyByLocale } from "../lib/ui-copy";

type SiteFooterProps = {
  locale: Locale;
};

export function SiteFooter({ locale }: SiteFooterProps) {
  const copy = siteFooterCopyByLocale[locale];
  const exploreLinks = getFooterExploreSections().map((section) => ({
    label: section.labels[locale],
    href: section.href(locale)
  }));
  const projectLinks = [
    ...getFooterProjectSections().map((section) => ({
      label: section.labels[locale],
      href: section.href(locale)
    })),
    { label: "RSS", href: localizePath("/feed.xml", locale) }
  ];
  const communityLinks = [
    { label: "GitHub", href: siteConfig.repoUrl },
    { label: "Discussions", href: "https://github.com/orgs/eunomia-bpf/discussions" }
  ];

  return (
    <footer className="mt-16 border-t border-slate-200 bg-white">
      <div className="mx-auto grid max-w-6xl gap-8 px-5 py-10 text-sm text-slate-600 md:grid-cols-[minmax(0,1.3fr)_repeat(3,minmax(0,1fr))]">
        <div>
          <p className="text-lg font-semibold tracking-tight text-ink">eunomia</p>
          <p className="mt-3 max-w-md leading-7">{copy.copyright}</p>
          <p className="mt-3 text-xs text-slate-500">
            © {new Date().getFullYear()} eunomia-bpf · MIT License
          </p>
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
            <a href={link.href} className="transition hover:text-ink">
              {link.label}
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
}

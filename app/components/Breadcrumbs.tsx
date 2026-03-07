import type { PageLink } from "../lib/content/types";
import type { Locale } from "../lib/site-data";

type BreadcrumbsProps = {
  locale: Locale;
  currentTitle: string;
  sectionLink?: PageLink;
};

export function Breadcrumbs({ locale, currentTitle, sectionLink }: BreadcrumbsProps) {
  const home = locale === "zh" ? { label: "主页", href: "/zh/" } : { label: "Home", href: "/" };
  const ariaLabel = locale === "zh" ? "面包屑" : "Breadcrumb";

  return (
    <nav aria-label={ariaLabel} className="mb-5 flex flex-wrap items-center gap-2 text-sm text-slate-500">
      <a href={home.href} className="transition hover:text-azure">
        {home.label}
      </a>
      {sectionLink ? (
        <>
          <span>/</span>
          <a href={sectionLink.href} className="transition hover:text-azure">
            {sectionLink.title}
          </a>
        </>
      ) : null}
      <span>/</span>
      <span className="text-slate-700">{currentTitle}</span>
    </nav>
  );
}

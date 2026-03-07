import type { PageLink } from "../lib/content/types";
import type { Locale } from "../lib/site-data";
import { breadcrumbCopyByLocale } from "../lib/ui-copy";

type BreadcrumbsProps = {
  locale: Locale;
  currentTitle: string;
  sectionLink?: PageLink;
};

export function Breadcrumbs({ locale, currentTitle, sectionLink }: BreadcrumbsProps) {
  const copy = breadcrumbCopyByLocale[locale];

  return (
    <nav aria-label={copy.ariaLabel} className="mb-5 flex flex-wrap items-center gap-2 text-sm text-slate-500">
      <a href={copy.homeHref} className="transition hover:text-azure">
        {copy.homeLabel}
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

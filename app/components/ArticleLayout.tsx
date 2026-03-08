import type { PropsWithChildren } from "react";

import type { GitMetadata, HeadingEntry, PageContinuation } from "../lib/content/types";
import type { Locale } from "../lib/site-data";
import { Breadcrumbs } from "./Breadcrumbs";
import { PageFooter } from "./PageFooter";
import { TableOfContents } from "./TableOfContents";

type ArticleLayoutProps = PropsWithChildren<{
  locale: Locale;
  path: string;
  title: string;
  description: string;
  sourceHref: string;
  metadata?: GitMetadata | null;
  headings?: HeadingEntry[];
  continuation?: PageContinuation;
  tocTitle?: string;
  showBreadcrumbs?: boolean;
}>;

export function ArticleLayout({
  children,
  locale,
  path,
  title,
  description,
  sourceHref,
  metadata,
  headings = [],
  continuation,
  tocTitle = "On this page",
  showBreadcrumbs = true
}: ArticleLayoutProps) {
  const hasToc = headings.length > 0;

  return (
    <section className="pb-16">
      <div className={hasToc ? "grid gap-10 xl:grid-cols-[minmax(0,1fr)_14rem]" : ""}>
        <article className="min-w-0 max-w-3xl">
          {showBreadcrumbs ? (
            <Breadcrumbs locale={locale} currentTitle={title} sectionLink={continuation?.index} />
          ) : null}
          <h1 className="text-3xl font-semibold tracking-tight text-ink md:text-[2.2rem]">{title}</h1>
          <p className="mt-4 text-base leading-7 text-slate-600">{description}</p>
          <TableOfContents
            headings={headings}
            compact
            className="mt-8 xl:hidden"
            title={tocTitle}
            ariaLabel={`${tocTitle} mobile`}
          />
          <div className="mt-8 content-copy text-base">{children}</div>
          <PageFooter
            locale={locale}
            title={title}
            path={path}
            sourceHref={sourceHref}
            metadata={metadata}
            continuation={continuation}
          />
        </article>
        <TableOfContents headings={headings} className="hidden xl:block" title={tocTitle} />
      </div>
    </section>
  );
}

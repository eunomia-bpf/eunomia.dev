import type { GitMetadata, PageContinuation } from "../lib/content/types";
import { localizePath } from "../lib/paths";
import { siteConfig, type Locale } from "../lib/site-data";
import { pageFooterCopyByLocale } from "../lib/ui-copy";
import { FeedbackWidget } from "./FeedbackWidget";

type PageFooterProps = {
  locale: Locale;
  title: string;
  path: string;
  sourceHref: string;
  metadata?: GitMetadata | null;
  continuation?: PageContinuation;
};

function formatDate(value: string | undefined, locale: Locale) {
  if (!value) {
    return null;
  }

  return new Intl.DateTimeFormat(locale === "zh" ? "zh-CN" : "en-US", {
    dateStyle: "medium"
  }).format(new Date(value));
}

function joinAuthors(metadata?: GitMetadata | null) {
  if (!metadata?.authors.length) {
    return null;
  }

  return metadata.authors.map((author) => author.name).join(", ");
}

const MAX_DISPLAYED_AUTHORS = 4;

export function PageFooter({ locale, title, path, sourceHref, metadata, continuation }: PageFooterProps) {
  const labels = pageFooterCopyByLocale[locale];

  const absolutePath = new URL(path, siteConfig.siteUrl).toString();
  const shareTitle = encodeURIComponent(`${title}\n`);
  const shareUrl = encodeURIComponent(absolutePath);
  const updated = formatDate(metadata?.updatedAt, locale);
  const created = formatDate(metadata?.createdAt, locale);
  const authors =
    metadata?.authors.length && metadata.authors.length > MAX_DISPLAYED_AUTHORS
      ? `${metadata.authors
          .slice(0, MAX_DISPLAYED_AUTHORS)
          .map((author) => author.name)
          .join(", ")}, +${metadata.authors.length - MAX_DISPLAYED_AUTHORS} ${labels.overflow}`
      : joinAuthors(metadata);
  const feedHref = localizePath("/feed.xml", locale);
  const navigationCards = [
    continuation?.index
      ? { label: labels.backToIndex, ...continuation.index }
      : null,
    continuation?.previous
      ? { label: labels.previous, ...continuation.previous }
      : null,
    continuation?.next
      ? { label: labels.next, ...continuation.next }
      : null
  ].filter(Boolean) as Array<{ label: string; title: string; description: string; href: string }>;

  return (
    <div className="mt-10 space-y-5 border-t border-slate-200 pt-6">
      {navigationCards.length ? (
        <section className="space-y-3">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">{labels.continue}</p>
          </div>
          <div className="grid gap-3 md:grid-cols-3">
            {navigationCards.map((card) => (
              <a
                key={`${card.label}:${card.href}`}
                href={card.href}
                className="border border-slate-200 px-4 py-4 transition hover:bg-slate-50/60 hover:border-slate-300"
              >
                <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">{card.label}</p>
                <p className="mt-2 text-base font-semibold text-ink">{card.title}</p>
                <p className="mt-2 text-sm leading-6 text-slate-600">{card.description}</p>
              </a>
            ))}
          </div>
        </section>
      ) : null}

      {metadata ? (
        <dl className="grid gap-4 border border-slate-200 p-5 text-sm text-slate-600 md:grid-cols-3">
          <div>
            <dt className="font-semibold text-ink">{labels.updated}</dt>
            <dd className="mt-1">{updated ?? "—"}</dd>
          </div>
          <div>
            <dt className="font-semibold text-ink">{labels.created}</dt>
            <dd className="mt-1">{created ?? "—"}</dd>
          </div>
          <div>
            <dt className="font-semibold text-ink">{labels.authors}</dt>
            <dd className="mt-1">{authors ?? "—"}</dd>
          </div>
        </dl>
      ) : null}

      <div className="flex flex-wrap gap-3">
        <a
          href={sourceHref}
          className="inline-flex rounded-md border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 transition hover:border-slate-400 hover:text-ink"
        >
          {labels.edit}
        </a>
        <a
          href={`https://x.com/intent/tweet?text=${shareTitle}&url=${shareUrl}`}
          target="_blank"
          rel="noreferrer"
          className="inline-flex rounded-md border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 transition hover:border-slate-400 hover:text-ink"
        >
          {labels.shareX}
        </a>
        <a
          href={`https://www.facebook.com/sharer/sharer.php?u=${shareUrl}`}
          target="_blank"
          rel="noreferrer"
          className="inline-flex rounded-md border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 transition hover:border-slate-400 hover:text-ink"
        >
          {labels.shareFacebook}
        </a>
        <a
          href="https://github.com/orgs/eunomia-bpf/discussions"
          target="_blank"
          rel="noreferrer"
          className="inline-flex rounded-md border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 transition hover:border-slate-400 hover:text-ink"
        >
          {labels.discuss}
        </a>
        <a
          href={feedHref}
          className="inline-flex rounded-md border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 transition hover:border-slate-400 hover:text-ink"
        >
          {labels.feed}
        </a>
      </div>

      <FeedbackWidget locale={locale} path={path} title={title} />
    </div>
  );
}

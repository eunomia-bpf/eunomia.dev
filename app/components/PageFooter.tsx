import type { GitMetadata } from "../lib/content/types";
import { siteConfig, type Locale } from "../lib/site-data";
import { FeedbackWidget } from "./FeedbackWidget";

type PageFooterProps = {
  locale: Locale;
  title: string;
  path: string;
  sourceHref: string;
  metadata?: GitMetadata | null;
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

export function PageFooter({ locale, title, path, sourceHref, metadata }: PageFooterProps) {
  const labels =
    locale === "zh"
      ? {
          updated: "最后更新",
          created: "首次发布",
          authors: "贡献者",
          edit: "编辑此页",
          shareX: "分享到 X",
          shareFacebook: "分享到 Facebook",
          discuss: "参与讨论",
          overflow: "更多"
        }
      : {
          updated: "Last updated",
          created: "First published",
          authors: "Contributors",
          edit: "Edit this page",
          shareX: "Share on X",
          shareFacebook: "Share on Facebook",
          discuss: "Join discussion",
          overflow: "more"
        };

  const absolutePath = new URL(path, siteConfig.siteUrl).toString();
  const shareTitle = encodeURIComponent(`${title}\n`);
  const shareUrl = encodeURIComponent(absolutePath);
  const updated = formatDate(metadata?.updatedAt, locale);
  const created = formatDate(metadata?.createdAt, locale);
  const authors =
    metadata?.authors.length && metadata.authors.length > 4
      ? `${metadata.authors
          .slice(0, 4)
          .map((author) => author.name)
          .join(", ")}, +${metadata.authors.length - 4} ${labels.overflow}`
      : joinAuthors(metadata);

  return (
    <div className="mt-10 space-y-5 border-t border-slate-200 pt-6">
      {metadata ? (
        <dl className="grid gap-4 rounded-[1.5rem] border border-slate-200 bg-slate-50/80 p-5 text-sm text-slate-600 md:grid-cols-3">
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
          className="inline-flex rounded-full border border-slate-200 px-4 py-2 text-sm font-medium text-slate-700 transition hover:border-azure hover:text-azure"
        >
          {labels.edit}
        </a>
        <a
          href={`https://x.com/intent/tweet?text=${shareTitle}&url=${shareUrl}`}
          target="_blank"
          rel="noreferrer"
          className="inline-flex rounded-full border border-slate-200 px-4 py-2 text-sm font-medium text-slate-700 transition hover:border-azure hover:text-azure"
        >
          {labels.shareX}
        </a>
        <a
          href={`https://www.facebook.com/sharer/sharer.php?u=${shareUrl}`}
          target="_blank"
          rel="noreferrer"
          className="inline-flex rounded-full border border-slate-200 px-4 py-2 text-sm font-medium text-slate-700 transition hover:border-azure hover:text-azure"
        >
          {labels.shareFacebook}
        </a>
        <a
          href="https://github.com/orgs/eunomia-bpf/discussions"
          target="_blank"
          rel="noreferrer"
          className="inline-flex rounded-full border border-slate-200 px-4 py-2 text-sm font-medium text-slate-700 transition hover:border-azure hover:text-azure"
        >
          {labels.discuss}
        </a>
      </div>

      <FeedbackWidget locale={locale} path={path} title={title} />
    </div>
  );
}

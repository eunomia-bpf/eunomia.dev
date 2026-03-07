import type { PropsWithChildren } from "react";

type ArticleLayoutProps = PropsWithChildren<{
  title: string;
  description: string;
  sourceHref: string;
}>;

export function ArticleLayout({
  children,
  title,
  description,
  sourceHref
}: ArticleLayoutProps) {
  return (
    <section className="mx-auto max-w-4xl px-5 pb-16">
      <article className="rounded-[2rem] border border-white/70 bg-white/90 p-8 shadow-panel md:p-10">
        <h1 className="text-3xl font-semibold tracking-tight text-ink md:text-5xl">{title}</h1>
        <p className="mt-5 text-lg leading-8 text-slate-600">{description}</p>
        <div className="mt-8 content-copy text-base">{children}</div>
        <div className="mt-10 border-t border-slate-200 pt-6">
          <a
            href={sourceHref}
            className="inline-flex rounded-full border border-slate-200 px-4 py-2 text-sm font-medium text-slate-700 transition hover:border-azure hover:text-azure"
          >
            Edit this page
          </a>
        </div>
      </article>
    </section>
  );
}

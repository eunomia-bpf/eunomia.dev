import type { HomePageData } from "../lib/page-factories";
import { localizePath } from "../lib/paths";
import { type Locale, siteConfig } from "../lib/site-data";
import { homeLandingCopyByLocale } from "../lib/ui-copy";
import { PageFooter } from "./PageFooter";

type HomeLandingProps = {
  locale: Locale;
  page: HomePageData;
};

export function HomeLanding({ locale, page }: HomeLandingProps) {
  const copy = homeLandingCopyByLocale[locale];

  return (
    <section className="bg-white pb-12">
      <section className="border-b border-slate-200">
        <div className="mx-auto max-w-[82rem] px-4 py-8 sm:px-6 lg:px-8">
          <div className="border border-slate-200 bg-slate-50/70 px-6 py-10 md:px-8">
            <span className="inline-flex text-[11px] font-semibold uppercase tracking-[0.24em] text-slate-500">
              {copy.badge}
            </span>
            <h1 className="mt-4 max-w-4xl text-4xl font-semibold leading-tight tracking-tight text-ink md:text-[3rem]">
              {copy.headline}
            </h1>
            <p className="mt-4 max-w-3xl text-base leading-8 text-slate-600">{copy.body}</p>
            <div className="mt-7 flex flex-wrap gap-3">
              <a
                href={page.cards[0]?.href ?? localizePath("/tutorials/", locale)}
                className="inline-flex rounded-md border border-slate-900 bg-slate-900 px-5 py-3 text-sm font-semibold text-white transition hover:bg-slate-800"
              >
                {copy.primaryCta}
              </a>
              <a
                href={page.spotlight.href}
                className="inline-flex rounded-md border border-slate-300 px-5 py-3 text-sm font-semibold text-slate-700 transition hover:border-slate-400 hover:text-ink"
              >
                {copy.secondaryCta}
              </a>
              <a
                href={siteConfig.repoUrl}
                className="inline-flex rounded-md border border-slate-300 px-5 py-3 text-sm font-semibold text-slate-700 transition hover:border-slate-400 hover:text-ink"
              >
                {copy.tertiaryCta}
              </a>
            </div>
          </div>
        </div>
      </section>

      <section className="mx-auto grid max-w-[82rem] gap-10 px-4 py-8 sm:px-6 lg:grid-cols-[minmax(0,1fr)_17rem] lg:px-8">
        <div className="min-w-0">
          <div className="mb-4">
            <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-500">{copy.tracksLabel}</p>
          </div>
          <div className="border-t border-slate-200">
            {page.cards.map((card, index) => (
              <a
                key={card.href}
                href={card.href}
                className={`block py-5 transition hover:bg-slate-50/60 ${index > 0 ? "border-t border-slate-200" : ""}`}
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="min-w-0">
                    <h2 className="text-lg font-semibold tracking-tight text-ink">{card.title}</h2>
                    <p className="mt-2 max-w-3xl leading-7 text-slate-600">{card.description}</p>
                    <span className="mt-3 inline-flex text-sm font-medium text-slate-700">{copy.openCard}</span>
                  </div>
                  {card.badge ? <span className="whitespace-nowrap text-xs font-medium text-slate-500">{card.badge}</span> : null}
                </div>
              </a>
            ))}
          </div>

          {page.moreLinks.length ? (
            <section className="mt-10">
              <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-500">{copy.exploreLabel}</p>
              <div className="mt-3 border-t border-slate-200">
                {page.moreLinks.map((card, index) => (
                  <a
                    key={card.href}
                    href={card.href}
                    className={`block py-4 ${index > 0 ? "border-t border-slate-200" : ""}`}
                  >
                    <h3 className="text-base font-semibold text-ink">{card.title}</h3>
                    <p className="mt-1 text-sm leading-6 text-slate-600">{card.description}</p>
                  </a>
                ))}
              </div>
            </section>
          ) : null}
        </div>

        <aside className="space-y-8">
          <section className="border-t border-slate-200 pt-4">
            <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-500">{copy.spotlightLabel}</p>
            <h2 className="mt-3 text-lg font-semibold tracking-tight text-ink">{page.spotlight.title}</h2>
            <p className="mt-2 text-sm leading-7 text-slate-600">{page.spotlight.description}</p>
            <a href={page.spotlight.href} className="mt-4 inline-flex text-sm font-medium text-slate-900 underline underline-offset-4">
              {page.spotlight.badge}
            </a>
          </section>

          <section className="border-t border-slate-200 pt-4">
            <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-500">{copy.signalLabel}</p>
            <p className="mt-3 text-sm leading-7 text-slate-600">{copy.signalTitle}</p>
            <ul className="mt-4 space-y-2 text-sm leading-7 text-slate-600">
              {copy.signals.map((signal) => (
                <li key={signal}>{signal}</li>
              ))}
            </ul>
            <dl className="mt-5 space-y-3 border-t border-slate-200 pt-4">
              {page.stats.map((stat) => (
                <div key={stat.label}>
                  <dt className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">{stat.label}</dt>
                  <dd className="mt-1 text-xl font-semibold text-ink">{stat.value}</dd>
                  <dd className="mt-1 text-sm leading-6 text-slate-600">{stat.detail}</dd>
                </div>
              ))}
            </dl>
          </section>
        </aside>
      </section>

      <section className="mx-auto max-w-[82rem] px-4 sm:px-6 lg:px-8">
        <article className="border-t border-slate-200 pt-6">
          <PageFooter
            locale={locale}
            title={page.title}
            path={page.path}
            sourceHref={page.sourcePath}
            metadata={page.metadata}
          />
        </article>
      </section>
    </section>
  );
}

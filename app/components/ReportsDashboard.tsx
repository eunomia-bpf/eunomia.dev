import type { ReportCadence, ReportEntry } from "../lib/content/reports";
import type { Locale } from "../lib/site-data";

type ReportsDashboardProps = {
  entries: ReportEntry[];
  locale: Locale;
};

function formatDate(date: string, locale: Locale, options?: Intl.DateTimeFormatOptions): string {
  return new Intl.DateTimeFormat(locale === "zh" ? "zh-CN" : "en-US", {
    timeZone: "UTC",
    ...options
  }).format(new Date(`${date}T00:00:00Z`));
}

function formatDuration(duration: string | undefined, locale: Locale, fallback: string): string {
  if (!duration || locale !== "zh") {
    return duration ?? fallback;
  }

  const match = duration.match(/^(\d+)\s+days?,\s*(\d{1,2}):(\d{2}):(\d{2})$/i);
  return match
    ? `${match[1]} 天 ${Number(match[2])} 小时 ${Number(match[3])} 分 ${Number(match[4])} 秒`
    : duration;
}

function ReportRows({ entries, locale }: ReportsDashboardProps) {
  const copy =
    locale === "zh"
      ? { open: "查看报告", empty: "新的 Markdown 报告发布后会自动出现在这里。" }
      : { open: "Open report", empty: "New Markdown reports will appear here automatically." };

  if (!entries.length) {
    return <p className="border-t border-slate-200 py-6 text-sm leading-6 text-slate-500">{copy.empty}</p>;
  }

  return (
    <div className="border-t border-slate-200">
      {entries.map((entry) => (
        <a
          key={entry.key}
          href={entry.href}
          className="grid gap-3 border-b border-slate-200 py-5 transition hover:bg-slate-50/70 sm:grid-cols-[9rem_minmax(0,1fr)_auto] sm:items-center sm:px-3"
        >
          <time className="text-sm font-medium text-slate-500" dateTime={entry.periodEnd}>
            {formatDate(entry.periodEnd, locale, { year: "numeric", month: "short", day: "numeric" })}
          </time>
          <span className="min-w-0">
            <span className="block text-base font-semibold text-ink">{entry.title}</span>
            <span className="mt-1 block text-sm leading-6 text-slate-600">{entry.description}</span>
          </span>
          <span className="text-sm font-medium text-slate-700">{copy.open}</span>
        </a>
      ))}
    </div>
  );
}

function cadenceEntries(entries: ReportEntry[], cadence: ReportCadence): ReportEntry[] {
  return entries.filter((entry) => entry.cadence === cadence);
}

export function ReportsDashboard({ entries, locale }: ReportsDashboardProps) {
  const latest = entries[0];
  const earliest = entries.at(-1);
  const monthly = cadenceEntries(entries, "monthly");
  const weekly = cadenceEntries(entries, "weekly");
  const copy =
    locale === "zh"
      ? {
          kicker: "公开社区看板",
          title: "Eunomia 社区活动",
          intro: "周报和月报直接从公开 Markdown 生成。每次新增报告后，最新进展、覆盖周期和归档会自动更新。",
          reports: "已发布报告",
          coverage: "覆盖周期",
          latest: "最近更新",
          current: "最新报告",
          monthly: "月报",
          weekly: "周报",
          tracked: "记录事项",
          closed: "已关闭",
          closeTime: "平均关闭时间",
          noMetric: "见报告",
          methodology: "这里汇总公开 GitHub 组织活动，包括 issue、pull request、关闭情况和项目进展。报告保留原始链接，便于回到具体工作核对。"
        }
      : {
          kicker: "Public community dashboard",
          title: "Eunomia community activity",
          intro: "Weekly and monthly reports are generated from public Markdown. New reports automatically update the latest results, coverage window, and archive.",
          reports: "Published reports",
          coverage: "Coverage window",
          latest: "Latest update",
          current: "Latest report",
          monthly: "Monthly reports",
          weekly: "Weekly reports",
          tracked: "Tracked items",
          closed: "Closed",
          closeTime: "Average close time",
          noMetric: "See report",
          methodology: "This dashboard summarizes public GitHub organization activity, including issues, pull requests, closure progress, and project work. Every report retains its source links for verification."
        };

  const coverage = latest && earliest
    ? `${formatDate(earliest.periodStart, locale, { year: "numeric", month: "short" })} - ${formatDate(latest.periodEnd, locale, { year: "numeric", month: "short" })}`
    : copy.noMetric;

  return (
    <div className="pb-16">
      <header className="border-b border-slate-200 pb-8">
        <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">{copy.kicker}</p>
        <h1 className="mt-3 max-w-4xl text-4xl font-semibold tracking-normal text-ink md:text-5xl">{copy.title}</h1>
        <p className="mt-5 max-w-3xl text-base leading-7 text-slate-600">{copy.intro}</p>
      </header>

      <dl className="grid border-b border-slate-200 sm:grid-cols-3">
        <div className="py-6 sm:pr-6">
          <dt className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">{copy.reports}</dt>
          <dd className="mt-2 text-3xl font-semibold text-ink">{entries.length}</dd>
        </div>
        <div className="border-t border-slate-200 py-6 sm:border-l sm:border-t-0 sm:px-6">
          <dt className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">{copy.coverage}</dt>
          <dd className="mt-2 text-lg font-semibold text-ink">{coverage}</dd>
        </div>
        <div className="border-t border-slate-200 py-6 sm:border-l sm:border-t-0 sm:pl-6">
          <dt className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">{copy.latest}</dt>
          <dd className="mt-2 text-lg font-semibold text-ink">
            {latest ? formatDate(latest.periodEnd, locale, { year: "numeric", month: "long", day: "numeric" }) : copy.noMetric}
          </dd>
        </div>
      </dl>

      {latest ? (
        <section className="py-9" aria-labelledby="latest-report">
          <p className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">{copy.current}</p>
          <div className="mt-3 grid gap-6 border-y border-slate-200 py-6 lg:grid-cols-[minmax(0,1fr)_22rem] lg:items-end">
            <div>
              <h2 id="latest-report" className="text-2xl font-semibold tracking-normal text-ink">{latest.title}</h2>
              <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-600">{latest.description}</p>
              <a href={latest.href} className="mt-5 inline-flex text-sm font-semibold text-slate-800 underline decoration-slate-300 underline-offset-4 hover:decoration-slate-600">
                {locale === "zh" ? "阅读完整报告" : "Read the full report"}
              </a>
            </div>
            <dl className="grid grid-cols-3 gap-4">
              <div>
                <dt className="text-xs text-slate-500">{copy.tracked}</dt>
                <dd className="mt-1 text-xl font-semibold text-ink">{latest.totalItems ?? copy.noMetric}</dd>
              </div>
              <div>
                <dt className="text-xs text-slate-500">{copy.closed}</dt>
                <dd className="mt-1 text-xl font-semibold text-ink">{latest.closedItems ?? copy.noMetric}</dd>
              </div>
              <div>
                <dt className="text-xs text-slate-500">{copy.closeTime}</dt>
                <dd className="mt-1 text-sm font-semibold leading-6 text-ink">
                  {formatDuration(latest.averageClose, locale, copy.noMetric)}
                </dd>
              </div>
            </dl>
          </div>
        </section>
      ) : null}

      <section className="py-8" aria-labelledby="monthly-reports">
        <h2 id="monthly-reports" className="mb-4 text-2xl font-semibold tracking-normal text-ink">{copy.monthly}</h2>
        <ReportRows entries={monthly} locale={locale} />
      </section>

      <section className="py-8" aria-labelledby="weekly-reports">
        <h2 id="weekly-reports" className="mb-4 text-2xl font-semibold tracking-normal text-ink">{copy.weekly}</h2>
        <ReportRows entries={weekly} locale={locale} />
      </section>

      <p className="border-t border-slate-200 pt-6 text-sm leading-6 text-slate-500">{copy.methodology}</p>
    </div>
  );
}

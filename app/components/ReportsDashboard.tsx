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
    ? `${match[1]} 天 ${Number(match[2])} 小时 ${Number(match[3])} 分`
    : duration;
}

function cadenceEntries(entries: ReportEntry[], cadence: ReportCadence): ReportEntry[] {
  return entries.filter((entry) => entry.cadence === cadence);
}

function closeRate(entry: ReportEntry | undefined): number | undefined {
  if (!entry?.totalItems || entry.closedItems === undefined) {
    return undefined;
  }
  return Math.round((entry.closedItems / entry.totalItems) * 100);
}

function WeeklyActivityChart({ entries, locale }: ReportsDashboardProps) {
  const chartEntries = [...entries]
    .sort((left, right) => left.periodStart.localeCompare(right.periodStart))
    .slice(-12);
  const width = 920;
  const height = 320;
  const margin = { top: 24, right: 16, bottom: 52, left: 42 };
  const chartWidth = width - margin.left - margin.right;
  const chartHeight = height - margin.top - margin.bottom;
  const largest = Math.max(1, ...chartEntries.flatMap((entry) => [entry.totalItems ?? 0, entry.closedItems ?? 0]));
  const yMax = Math.ceil(largest / 20) * 20;
  const groupWidth = chartWidth / Math.max(1, chartEntries.length);
  const barWidth = Math.min(20, groupWidth * 0.28);
  const copy = locale === "zh"
    ? { opened: "新建", closed: "关闭", label: "近 12 周新建和关闭的 issue 与 pull request" }
    : { opened: "Opened", closed: "Closed", label: "Issues and pull requests opened and closed over the last 12 weeks" };

  return (
    <div>
      <div className="mb-4 flex flex-wrap gap-x-6 gap-y-2 text-sm text-slate-600" aria-hidden="true">
        <span className="inline-flex items-center gap-2"><span className="h-2.5 w-2.5 bg-cyan-700" />{copy.opened}</span>
        <span className="inline-flex items-center gap-2"><span className="h-2.5 w-2.5 bg-amber-500" />{copy.closed}</span>
      </div>
      <div className="overflow-x-auto pb-2">
        <svg
          className="min-w-[720px]"
          viewBox={`0 0 ${width} ${height}`}
          role="img"
          aria-label={copy.label}
        >
          <title>{copy.label}</title>
          {[0, 0.25, 0.5, 0.75, 1].map((fraction) => {
            const value = Math.round(yMax * (1 - fraction));
            const y = margin.top + chartHeight * fraction;
            return (
              <g key={fraction}>
                <line x1={margin.left} x2={width - margin.right} y1={y} y2={y} stroke="#e2e8f0" strokeWidth="1" />
                <text x={margin.left - 10} y={y + 4} textAnchor="end" fontSize="11" fill="#64748b">{value}</text>
              </g>
            );
          })}
          {chartEntries.map((entry, index) => {
            const opened = entry.totalItems ?? 0;
            const closed = entry.closedItems ?? 0;
            const center = margin.left + groupWidth * index + groupWidth / 2;
            const openedHeight = (opened / yMax) * chartHeight;
            const closedHeight = (closed / yMax) * chartHeight;
            return (
              <g key={entry.key}>
                <title>{`${formatDate(entry.periodStart, locale, { month: "short", day: "numeric" })}: ${copy.opened} ${opened}, ${copy.closed} ${closed}`}</title>
                <rect x={center - barWidth - 2} y={margin.top + chartHeight - openedHeight} width={barWidth} height={openedHeight} fill="#0e7490" />
                <rect x={center + 2} y={margin.top + chartHeight - closedHeight} width={barWidth} height={closedHeight} fill="#f59e0b" />
                <text x={center} y={height - 24} textAnchor="middle" fontSize="10" fill="#64748b">
                  {formatDate(entry.periodStart, locale, { month: "short", day: "numeric" })}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
}

function MonthlyActivityChart({ entries, locale }: ReportsDashboardProps) {
  const chartEntries = [...entries].sort((left, right) => left.periodStart.localeCompare(right.periodStart));
  const maxItems = Math.max(1, ...chartEntries.map((entry) => entry.totalItems ?? 0));
  const maxStars = Math.max(1, ...chartEntries.map((entry) => entry.newStars ?? 0));
  const copy = locale === "zh"
    ? { activity: "组织活动", stars: "新增 Star", opened: "记录", closed: "关闭", rate: "关闭比例" }
    : { activity: "Organization activity", stars: "New stars", opened: "Recorded", closed: "Closed", rate: "Closure ratio" };

  return (
    <div className="grid gap-10 lg:grid-cols-2">
      <div>
        <h3 className="text-sm font-semibold text-ink">{copy.activity}</h3>
        <div className="mt-5 space-y-5">
          {chartEntries.map((entry) => (
            <div key={entry.key} className="grid grid-cols-[4.5rem_minmax(0,1fr)_3.5rem] items-center gap-3">
              <span className="text-sm font-medium text-slate-600">{formatDate(entry.periodStart, locale, { month: "short", year: "2-digit" })}</span>
              <div className="space-y-1.5" aria-label={`${entry.title}: ${copy.opened} ${entry.totalItems ?? 0}, ${copy.closed} ${entry.closedItems ?? 0}`}>
                <div className="h-2 bg-slate-100"><div className="h-2 bg-cyan-700" style={{ width: `${((entry.totalItems ?? 0) / maxItems) * 100}%` }} /></div>
                <div className="h-2 bg-slate-100"><div className="h-2 bg-amber-500" style={{ width: `${((entry.closedItems ?? 0) / maxItems) * 100}%` }} /></div>
              </div>
              <span className="text-right text-sm font-semibold text-slate-700">{closeRate(entry) ?? 0}%</span>
            </div>
          ))}
        </div>
        <div className="mt-4 flex flex-wrap gap-x-5 gap-y-2 text-xs text-slate-500">
          <span><span className="mr-2 inline-block h-2 w-2 bg-cyan-700" />{copy.opened}</span>
          <span><span className="mr-2 inline-block h-2 w-2 bg-amber-500" />{copy.closed}</span>
          <span>{copy.rate}</span>
        </div>
      </div>

      <div>
        <h3 className="text-sm font-semibold text-ink">{copy.stars}</h3>
        <div className="mt-5 space-y-5">
          {chartEntries.map((entry) => (
            <div key={entry.key} className="grid grid-cols-[4.5rem_minmax(0,1fr)_3.5rem] items-center gap-3">
              <span className="text-sm font-medium text-slate-600">{formatDate(entry.periodStart, locale, { month: "short", year: "2-digit" })}</span>
              <div className="h-4 bg-slate-100" aria-label={`${entry.title}: ${entry.newStars ?? 0} ${copy.stars}`}>
                <div className="h-4 bg-emerald-600" style={{ width: `${((entry.newStars ?? 0) / maxStars) * 100}%` }} />
              </div>
              <span className="text-right text-sm font-semibold text-slate-700">{entry.newStars ?? 0}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function ReportRows({ entries, locale }: ReportsDashboardProps) {
  const copy = locale === "zh"
    ? { open: "查看", empty: "新的 Markdown 报告发布后会自动出现在这里。" }
    : { open: "Open", empty: "New Markdown reports will appear here automatically." };

  if (!entries.length) {
    return <p className="border-t border-slate-200 py-6 text-sm leading-6 text-slate-500">{copy.empty}</p>;
  }

  return (
    <div className="border-t border-slate-200">
      {entries.map((entry) => (
        <a
          key={entry.key}
          href={entry.href}
          className="grid gap-2 border-b border-slate-200 py-4 transition hover:bg-slate-50 sm:grid-cols-[8.5rem_minmax(0,1fr)_auto] sm:items-center sm:px-3"
        >
          <time className="text-sm font-medium text-slate-500" dateTime={entry.periodEnd}>
            {formatDate(entry.periodEnd, locale, { year: "numeric", month: "short", day: "numeric" })}
          </time>
          <span className="min-w-0">
            <span className="block text-sm font-semibold text-ink">{entry.title}</span>
            <span className="mt-1 block text-sm leading-6 text-slate-600">{entry.description}</span>
          </span>
          <span className="text-sm font-medium text-slate-700">{copy.open}</span>
        </a>
      ))}
    </div>
  );
}

export function ReportsDashboard({ entries, locale }: ReportsDashboardProps) {
  const monthly = cadenceEntries(entries, "monthly");
  const weekly = cadenceEntries(entries, "weekly");
  const latestWeekly = weekly[0];
  const latestMonthly = monthly[0];
  const latest = entries[0];
  const earliest = entries.at(-1);
  const latestRate = closeRate(latestWeekly);
  const copy = locale === "zh"
    ? {
        kicker: "公开社区数据",
        title: "Eunomia 社区活动看板",
        intro: "这里把公开 GitHub 活动按周和按月汇总。图表用于观察工作量、关闭进度和社区增长，下面的 Markdown 报告保留每个周期的原始口径。",
        weeks: "周度样本",
        coverage: "覆盖周期",
        closeRate: "最近一周关闭比例",
        stars: "最近一月新增 Star",
        weeklyTrend: "近 12 周活动趋势",
        weeklyNote: "每根柱子代表该周新建或关闭的 issue 与 pull request。",
        monthlyTrend: "月度活动与增长",
        monthlyNote: "组织活动与 Star 使用独立刻度，避免把不同量纲混在同一条轴上。",
        latest: "最近一周",
        tracked: "新建",
        closed: "关闭",
        latestMonth: "最近一月",
        closeTime: "平均关闭时间",
        reports: "报告归档",
        monthly: "月报",
        weekly: "周报",
        noMetric: "见报告",
        methodology: "周度数字来自 GitHub 对 eunomia-bpf 组织的公开检索；月度报告还包含 Star、仓库、提交和具体事项。历史快照可能因事项迁移或删除产生小幅变化。"
      }
    : {
        kicker: "Public community data",
        title: "Eunomia community dashboard",
        intro: "This dashboard aggregates public GitHub activity by week and month. The charts show workload, closure progress, and community growth, while the Markdown reports preserve the source definition for each period.",
        weeks: "Weekly samples",
        coverage: "Coverage window",
        closeRate: "Latest weekly closure ratio",
        stars: "Stars added in latest month",
        weeklyTrend: "Activity over the last 12 weeks",
        weeklyNote: "Each bar represents issues and pull requests opened or closed during that week.",
        monthlyTrend: "Monthly activity and growth",
        monthlyNote: "Organization activity and stars use separate scales so unlike units are not plotted on one axis.",
        latest: "Latest week",
        tracked: "Opened",
        closed: "Closed",
        latestMonth: "Latest month",
        closeTime: "Average close time",
        reports: "Report archive",
        monthly: "Monthly",
        weekly: "Weekly",
        noMetric: "See report",
        methodology: "Weekly counts come from public GitHub searches for the eunomia-bpf organization. Monthly reports also cover stars, repositories, commits, and individual work items. Historical snapshots can shift slightly when items are transferred or deleted."
      };

  const coverage = latest && earliest
    ? `${formatDate(earliest.periodStart, locale, { year: "numeric", month: "short" })} - ${formatDate(latest.periodEnd, locale, { year: "numeric", month: "short" })}`
    : copy.noMetric;

  return (
    <div className="pb-16">
      <header className="border-b border-slate-200 pb-8">
        <p className="text-[11px] font-semibold uppercase text-slate-500">{copy.kicker}</p>
        <h1 className="mt-3 max-w-4xl text-4xl font-semibold tracking-normal text-ink md:text-5xl">{copy.title}</h1>
        <p className="mt-5 max-w-3xl text-base leading-7 text-slate-600">{copy.intro}</p>
      </header>

      <dl className="grid border-b border-slate-200 sm:grid-cols-2 xl:grid-cols-4">
        <div className="py-6 sm:pr-6">
          <dt className="text-xs font-semibold uppercase text-slate-500">{copy.weeks}</dt>
          <dd className="mt-2 text-3xl font-semibold text-ink">{weekly.length}</dd>
        </div>
        <div className="border-t border-slate-200 py-6 sm:border-l sm:border-t-0 sm:pl-6 xl:pr-6">
          <dt className="text-xs font-semibold uppercase text-slate-500">{copy.coverage}</dt>
          <dd className="mt-2 text-lg font-semibold text-ink">{coverage}</dd>
        </div>
        <div className="border-t border-slate-200 py-6 sm:pr-6 xl:border-l xl:border-t-0 xl:pl-6">
          <dt className="text-xs font-semibold uppercase text-slate-500">{copy.closeRate}</dt>
          <dd className="mt-2 text-3xl font-semibold text-ink">{latestRate === undefined ? copy.noMetric : `${latestRate}%`}</dd>
        </div>
        <div className="border-t border-slate-200 py-6 sm:border-l sm:pl-6 xl:border-t-0">
          <dt className="text-xs font-semibold uppercase text-slate-500">{copy.stars}</dt>
          <dd className="mt-2 text-3xl font-semibold text-ink">{latestMonthly?.newStars ?? copy.noMetric}</dd>
        </div>
      </dl>

      <section className="border-b border-slate-200 py-10" aria-labelledby="weekly-activity">
        <div className="mb-7 flex flex-col justify-between gap-2 md:flex-row md:items-end">
          <div>
            <h2 id="weekly-activity" className="text-2xl font-semibold tracking-normal text-ink">{copy.weeklyTrend}</h2>
            <p className="mt-2 text-sm leading-6 text-slate-600">{copy.weeklyNote}</p>
          </div>
          {latestWeekly ? (
            <a href={latestWeekly.href} className="text-sm font-semibold text-slate-700 underline decoration-slate-300 underline-offset-4 hover:decoration-slate-600">
              {copy.latest}: {formatDate(latestWeekly.periodEnd, locale, { month: "short", day: "numeric" })}
            </a>
          ) : null}
        </div>
        <WeeklyActivityChart entries={weekly} locale={locale} />
      </section>

      <section className="border-b border-slate-200 py-10" aria-labelledby="monthly-activity">
        <div className="mb-8">
          <h2 id="monthly-activity" className="text-2xl font-semibold tracking-normal text-ink">{copy.monthlyTrend}</h2>
          <p className="mt-2 text-sm leading-6 text-slate-600">{copy.monthlyNote}</p>
        </div>
        <MonthlyActivityChart entries={monthly} locale={locale} />
      </section>

      <section className="grid gap-6 border-b border-slate-200 py-8 md:grid-cols-2" aria-label={locale === "zh" ? "最近周期摘要" : "Latest period summary"}>
        {latestWeekly ? (
          <div>
            <p className="text-xs font-semibold uppercase text-slate-500">{copy.latest}</p>
            <p className="mt-2 text-lg font-semibold text-ink">{latestWeekly.title}</p>
            <p className="mt-3 text-sm text-slate-600">{copy.tracked} <strong className="text-ink">{latestWeekly.totalItems ?? 0}</strong> · {copy.closed} <strong className="text-ink">{latestWeekly.closedItems ?? 0}</strong></p>
          </div>
        ) : null}
        {latestMonthly ? (
          <div className="md:border-l md:border-slate-200 md:pl-6">
            <p className="text-xs font-semibold uppercase text-slate-500">{copy.latestMonth}</p>
            <p className="mt-2 text-lg font-semibold text-ink">{latestMonthly.title}</p>
            <p className="mt-3 text-sm text-slate-600">{copy.closeTime}: <strong className="text-ink">{formatDuration(latestMonthly.averageClose, locale, copy.noMetric)}</strong></p>
          </div>
        ) : null}
      </section>

      <section className="py-10" aria-labelledby="report-archive">
        <h2 id="report-archive" className="text-2xl font-semibold tracking-normal text-ink">{copy.reports}</h2>
        <div className="mt-7 grid gap-10 xl:grid-cols-2">
          <div>
            <h3 className="mb-4 text-base font-semibold text-ink">{copy.weekly}</h3>
            <ReportRows entries={weekly} locale={locale} />
          </div>
          <div>
            <h3 className="mb-4 text-base font-semibold text-ink">{copy.monthly}</h3>
            <ReportRows entries={monthly} locale={locale} />
          </div>
        </div>
      </section>

      <p className="border-t border-slate-200 pt-6 text-sm leading-6 text-slate-500">{copy.methodology}</p>
    </div>
  );
}

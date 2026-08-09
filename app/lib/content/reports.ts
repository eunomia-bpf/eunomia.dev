import fs from "node:fs";
import path from "node:path";

import type { Locale } from "../site-data";
import { listDocuments, resolveDocument } from "./documents";
import { resolveRouteFromDocSource } from "./manifest";
import { docsRoot } from "./roots";
import { baseMarkdownPath } from "./source";

export type ReportCadence = "weekly" | "monthly";

export type ReportEntry = {
  key: string;
  title: string;
  description: string;
  href: string;
  cadence: ReportCadence;
  periodStart: string;
  periodEnd: string;
  totalItems?: number;
  closedItems?: number;
  averageClose?: string;
  newStars?: number;
};

function cadenceFromSource(source: string): ReportCadence | null {
  if (source.includes("/weekly/")) {
    return "weekly";
  }
  if (source.includes("/monthly/")) {
    return "monthly";
  }
  return null;
}

function periodFromDocument(title: string, source: string): [string, string] | null {
  const titlePeriod = title.match(/(\d{4}-\d{2}-\d{2})\.\.(\d{4}-\d{2}-\d{2})/);
  if (titlePeriod) {
    return [titlePeriod[1], titlePeriod[2]];
  }

  const monthlySource = source.match(/(\d{4})-(\d{2})\.md$/);
  if (!monthlySource) {
    return null;
  }

  const year = Number(monthlySource[1]);
  const month = Number(monthlySource[2]);
  const lastDay = new Date(Date.UTC(year, month, 0)).getUTCDate();
  return [
    `${monthlySource[1]}-${monthlySource[2]}-01`,
    `${monthlySource[1]}-${monthlySource[2]}-${String(lastDay).padStart(2, "0")}`
  ];
}

function metric(body: string, label: string): number | undefined {
  const match = body.match(new RegExp(`${label}:\\s*\\*\\*(\\d+)\\*\\*`, "i"));
  return match ? Number(match[1]) : undefined;
}

function reportTitle(cadence: ReportCadence, periodStart: string, periodEnd: string, locale: Locale): string {
  if (locale === "zh") {
    return cadence === "monthly"
      ? `${periodStart.slice(0, 4)} 年 ${Number(periodStart.slice(5, 7))} 月组织活动报告`
      : `${periodStart} 至 ${periodEnd} 组织活动周报`;
  }

  return cadence === "monthly"
    ? `Organization activity, ${periodStart.slice(0, 7)}`
    : `Organization activity, ${periodStart} to ${periodEnd}`;
}

function reportDescription(
  totalItems: number | undefined,
  closedItems: number | undefined,
  locale: Locale
): string {
  if (totalItems === undefined) {
    return locale === "zh" ? "该周期的公开组织活动与项目进展。" : "Public organization activity and project progress for this period.";
  }

  if (locale === "zh") {
    return closedItems === undefined
      ? `该周期共记录 ${totalItems} 个 issue 和 pull request。`
      : `该周期共记录 ${totalItems} 个 issue 和 pull request，其中 ${closedItems} 个已经关闭。`;
  }

  return closedItems === undefined
    ? `${totalItems} issues and pull requests recorded during this period.`
    : `${totalItems} issues and pull requests recorded, with ${closedItems} closed during the period.`;
}

export function getReportEntries(locale: Locale): ReportEntry[] {
  const baseSources = new Set(
    listDocuments()
      .map((document) => baseMarkdownPath(document.sourceRelative))
      .filter((source) => source.startsWith("reports/") && cadenceFromSource(source))
  );

  const entries: ReportEntry[] = [];

  for (const source of baseSources) {
    const cadence = cadenceFromSource(source);
    const document = resolveDocument(source, locale);
    if (!cadence || !document) {
      continue;
    }

    const period = periodFromDocument(document.title, source);
    const href = resolveRouteFromDocSource(document.sourceRelative, locale);
    if (!period || !href) {
      continue;
    }

    const rawSource = fs.readFileSync(path.join(docsRoot, document.sourceRelative), "utf8");
    const totalItems = metric(rawSource, "Total issues/PRs");
    const closedItems = metric(rawSource, "Closed issues/PRs");
    const averageCloseMatch = rawSource.match(/Average time to close:\s*\*\*([^*]+)\*\*/i);
    const newStarsMatch = rawSource.match(/Total new stars(?: \(non-archived repositories\))?:\s*\*\*(\d+)\*\*/i);

    entries.push({
      key: source,
      title: reportTitle(cadence, period[0], period[1], locale),
      description: reportDescription(totalItems, closedItems, locale),
      href,
      cadence,
      periodStart: period[0],
      periodEnd: period[1],
      ...(totalItems !== undefined ? { totalItems } : {}),
      ...(closedItems !== undefined ? { closedItems } : {}),
      ...(averageCloseMatch ? { averageClose: averageCloseMatch[1].trim() } : {}),
      ...(newStarsMatch ? { newStars: Number(newStarsMatch[1]) } : {})
    });
  }

  return entries.sort((left, right) => right.periodEnd.localeCompare(left.periodEnd));
}

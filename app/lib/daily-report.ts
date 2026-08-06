import type { DocsPage, HeadingEntry } from "./content/types";
import type { Locale } from "./site-data";

export function isDailyReportPath(path: string): boolean {
  return /^\/(?:zh\/)?research(?:\/|$)/.test(path);
}

export function getDailyReportLabel(locale: Locale): string {
  return locale === "zh" ? "每日报告" : "Daily Report";
}

function getDailyReportTags(tags: string[] | undefined, locale: Locale): string[] {
  const label = getDailyReportLabel(locale);
  return (tags ?? []).map((tag) => (tag === "Research" ? label : tag));
}

const dailyReportTextReplacements: Record<Locale, ReadonlyArray<readonly [string, string]>> = {
  en: [
    ["scope-limitations-and-falsification", "what-would-change-this-conclusion"],
    ["Scope, limitations, and falsification", "What would change this conclusion?"],
    ["The design also has real limits:", "The design still leaves several hard problems:"]
  ],
  zh: [
    ["适用范围、局限与可证伪条件", "哪些结果会改变这个判断"],
    ["适用范围局限与可证伪条件", "哪些结果会改变这个判断"],
    ["适用范围、局限性与可证伪条件", "哪些结果会改变这个判断"],
    ["适用范围局限性与可证伪条件", "哪些结果会改变这个判断"],
    ["适用范围、局限与可证伪性", "哪些结果会改变这个判断"],
    ["适用范围局限与可证伪性", "哪些结果会改变这个判断"],
    ["适用范围、局限性和可证伪性", "哪些结果会改变这个判断"],
    ["适用范围局限性和可证伪性", "哪些结果会改变这个判断"],
    ["适用范围、限制与可证伪条件", "哪些结果会改变这个判断"],
    ["适用范围限制与可证伪条件", "哪些结果会改变这个判断"],
    ["适用范围与可证伪条件", "哪些结果会改变这个判断"],
    ["它也有明显局限：", "它仍有几个无法绕开的难点："]
  ]
};

export function rewriteDailyReportText(value: string, locale: Locale): string {
  return dailyReportTextReplacements[locale].reduce(
    (current, [source, replacement]) => current.split(source).join(replacement),
    value
  );
}

export function rewriteDailyReportHeadings(
  headings: HeadingEntry[] | undefined,
  locale: Locale
): HeadingEntry[] {
  return (headings ?? []).map((heading) => ({
    ...heading,
    id: rewriteDailyReportText(heading.id, locale),
    text: rewriteDailyReportText(heading.text, locale)
  }));
}

export function prepareDailyReportPage(page: DocsPage, locale: Locale): DocsPage {
  if (!isDailyReportPath(page.path)) {
    return page;
  }

  return {
    ...page,
    bodyHtml: rewriteDailyReportText(page.bodyHtml, locale),
    headings: rewriteDailyReportHeadings(page.headings, locale),
    tags: getDailyReportTags(page.tags, locale),
    metadata: null
  };
}

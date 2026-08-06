import type { Locale } from "../lib/site-data";

export function DailyReportFooterNote({ locale }: { locale: Locale }) {
  const text =
    locale === "zh"
      ? "本页由自动化工具生成，未经人工逐项核验；使用前请结合引用的一手资料独立验证。"
      : "Generated automatically and not individually verified by an editor; check the cited primary sources before relying on it.";

  return (
    <p role="note" className="mt-4 max-w-3xl text-[11px] leading-5 text-slate-400">
      {text}
    </p>
  );
}

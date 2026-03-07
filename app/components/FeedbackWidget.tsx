"use client";

import { startTransition, useState } from "react";

import type { Locale } from "../lib/site-data";
import { feedbackCopyByLocale } from "../lib/ui-copy";

declare global {
  interface Window {
    gtag?: (...args: unknown[]) => void;
  }
}

type FeedbackWidgetProps = {
  locale: Locale;
  path: string;
  title: string;
};

type FeedbackState = "helpful" | "improve" | null;

function recordFeedback(state: Exclude<FeedbackState, null>, path: string, title: string) {
  if (typeof window === "undefined" || typeof window.gtag !== "function") {
    return;
  }

  window.gtag("event", "page_feedback", {
    event_category: "engagement",
    event_label: path,
    page_location: window.location.href,
    page_path: path,
    page_title: title,
    value: state === "helpful" ? 1 : 0
  });
}

export function FeedbackWidget({ locale, path, title }: FeedbackWidgetProps) {
  const [selected, setSelected] = useState<FeedbackState>(null);
  const copy = feedbackCopyByLocale[locale];

  function submitFeedback(nextState: Exclude<FeedbackState, null>) {
    recordFeedback(nextState, path, title);
    startTransition(() => {
      setSelected(nextState);
    });
  }

  return (
    <div className="rounded-[1.5rem] border border-slate-200 bg-white p-5">
      <p className="text-sm font-semibold text-ink">{copy.title}</p>
      <div className="mt-3 flex flex-wrap gap-3">
        <button
          type="button"
          onClick={() => submitFeedback("helpful")}
          className={`inline-flex rounded-full border px-4 py-2 text-sm font-medium transition ${
            selected === "helpful"
              ? "border-emerald-500 bg-emerald-500 text-white"
              : "border-slate-200 text-slate-700 hover:border-emerald-500 hover:text-emerald-600"
          }`}
        >
          {copy.helpful}
        </button>
        <button
          type="button"
          onClick={() => submitFeedback("improve")}
          className={`inline-flex rounded-full border px-4 py-2 text-sm font-medium transition ${
            selected === "improve"
              ? "border-amber-500 bg-amber-500 text-white"
              : "border-slate-200 text-slate-700 hover:border-amber-500 hover:text-amber-600"
          }`}
        >
          {copy.improve}
        </button>
      </div>
      <div aria-live="polite" className="mt-3 min-h-6 text-sm text-slate-600">
        {selected === "helpful" ? <p>{copy.thanks}</p> : null}
        {selected === "improve" ? (
          <p>
            {copy.improvePrefix}{" "}
            <a
              href="https://github.com/orgs/eunomia-bpf/discussions"
              target="_blank"
              rel="noreferrer"
              className="font-medium text-azure underline-offset-4 hover:underline"
            >
              {copy.discussion}
            </a>
            .
          </p>
        ) : null}
      </div>
    </div>
  );
}

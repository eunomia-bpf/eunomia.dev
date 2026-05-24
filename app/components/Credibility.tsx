import { useEffect, useState } from "react";

import type { ReactPageLink } from "../lib/content/types";
import type { Locale } from "../lib/site-data";

const ORG_URL = "https://github.com/eunomia-bpf";
const ORG_API = "https://api.github.com/orgs/eunomia-bpf/repos";
const SHIELD = "https://img.shields.io/github/stars/eunomia-bpf";
const CONTACT_EMAIL = "yusheng@eunomia.dev";

/** Fallback shown until the live count resolves, or if the GitHub API is rate-limited.
 *  Set to the org-wide total as of 2026-05; the live fetch paginates all repos and
 *  overrides this on success. */
const FALLBACK_TOTAL_STARS = 9300;

/**
 * Live, org-wide GitHub star total. shields.io has no "sum across an org" badge,
 * so we fetch the org's public repos in the browser and sum stargazers_count.
 * Kept resilient: renders a rounded fallback immediately, upgrades to the live
 * total on success, and never blocks or errors the page if the API is throttled.
 */
export function OrgStarTotal({ locale, className = "" }: { locale: Locale; className?: string }) {
  const [total, setTotal] = useState<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        let sum = 0;
        for (let page = 1; page <= 5; page += 1) {
          const res = await fetch(`${ORG_API}?per_page=100&type=public&page=${page}`);
          if (!res.ok) break;
          const repos = (await res.json()) as Array<{ stargazers_count?: number }>;
          if (!Array.isArray(repos) || repos.length === 0) break;
          sum += repos.reduce((acc, repo) => acc + (repo.stargazers_count ?? 0), 0);
          if (repos.length < 100) break;
        }
        if (!cancelled && sum > 0) setTotal(sum);
      } catch {
        /* keep the fallback */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const value = total ?? FALLBACK_TOTAL_STARS;
  const display = `${(Math.floor(value / 100) * 100).toLocaleString("en-US")}+`;

  return (
    <a
      href={ORG_URL}
      target="_blank"
      rel="noopener"
      className={`inline-flex items-center gap-1.5 rounded-full border border-cyan-700/40 bg-cyan-50 px-3 py-1 text-xs font-semibold text-cyan-800 transition hover:border-cyan-700/70 ${className}`.trim()}
    >
      <span aria-hidden="true">★</span>
      {locale === "zh" ? `${display} GitHub 星标` : `${display} GitHub stars`}
    </a>
  );
}

export type StarRepo = { repo: string; label: string };

/**
 * Live GitHub star badges (shields.io). Rendered as plain <img> on purpose: the
 * counts must stay fresh on every page view, and static export cannot embed a
 * build-time number without it going stale.
 */
export function StarBar({ repos, locale }: { repos: StarRepo[]; locale: Locale }) {
  return (
    <div className="flex flex-wrap items-center gap-2.5">
      <span className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
        {locale === "zh" ? "开源采用" : "Open-source traction"}
      </span>
      {repos.map((entry) => (
        <a
          key={entry.repo}
          href={`${ORG_URL}/${entry.repo}`}
          target="_blank"
          rel="noopener"
          className="inline-flex items-center gap-2 rounded-md border border-slate-200 bg-white px-2.5 py-1 text-xs font-semibold text-slate-600 transition hover:border-cyan-700/40 hover:text-ink"
        >
          {entry.label}
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={`${SHIELD}/${entry.repo}?style=flat-square&label=&color=0e7490&logo=github&logoColor=white`}
            alt={`${entry.label} stars on GitHub`}
            className="h-4"
            loading="lazy"
          />
        </a>
      ))}
    </div>
  );
}

/**
 * Stable credibility facts (publication, license, research provenance). Unlike
 * star counts these do not change, so the copy is fixed.
 */
export function CredibilityStrip({
  locale,
  osdi,
  className = ""
}: {
  locale: Locale;
  osdi?: ReactPageLink;
  className?: string;
}) {
  const facts: Array<{ label: string; href?: string }> = [
    osdi
      ? { label: osdi.label, href: osdi.href }
      : {
          label: locale === "zh" ? "OSDI 2025 发表" : "Published at OSDI 2025",
          href: "https://www.usenix.org/conference/osdi25/presentation/zheng-yusheng"
        },
    { label: locale === "zh" ? "MIT 开源协议" : "MIT licensed", href: ORG_URL },
    { label: locale === "zh" ? "源自系统研究" : "Backed by systems research" }
  ];

  const badgeClass =
    "inline-flex items-center gap-1.5 rounded-full border border-cyan-700/30 bg-cyan-50/60 px-3 py-1 text-xs font-semibold text-cyan-800";

  return (
    <div className={`flex flex-wrap items-center gap-2 ${className}`.trim()}>
      {facts.map((fact) => {
        const dot = <span aria-hidden="true" className="h-1.5 w-1.5 rounded-full bg-cyan-600" />;
        if (!fact.href) {
          return (
            <span key={fact.label} className={badgeClass}>
              {dot}
              {fact.label}
            </span>
          );
        }
        const external = /^https?:\/\//.test(fact.href);
        return (
          <a
            key={fact.label}
            href={fact.href}
            target={external ? "_blank" : undefined}
            rel={external ? "noopener" : undefined}
            className={`${badgeClass} transition hover:border-cyan-700/60`}
          >
            {dot}
            {fact.label}
          </a>
        );
      })}
    </div>
  );
}

/**
 * Commercial trust block: who maintains the work and how to reach a human.
 * `contact` is sourced from page config when available (mkdocs.yaml), otherwise
 * falls back to the canonical email.
 */
export function ContactCard({
  locale,
  contact,
  className = ""
}: {
  locale: Locale;
  contact?: ReactPageLink;
  className?: string;
}) {
  const email = contact?.href.startsWith("mailto:") ? contact.href.slice("mailto:".length) : CONTACT_EMAIL;
  const copy =
    locale === "zh"
      ? {
          eyebrow: "联系",
          title: "和维护团队直接对话",
          body:
            "由 eunomia-bpf 开源团队维护，工作根植于已发表的系统研究（OSDI 2025）。企业评估、POC 和生产集成可直接邮件联系，我们通常在两个工作日内回复。",
          cta: "邮件联系",
          response: "通常 2 个工作日内回复"
        }
      : {
          eyebrow: "Contact",
          title: "Talk to the people who maintain it",
          body:
            "Maintained by the eunomia-bpf open-source team, with work grounded in published systems research (OSDI 2025). Reach out directly for enterprise evaluation, POCs, and production integration.",
          cta: "Email us",
          response: "Typically replies within 2 business days"
        };

  return (
    <div
      className={`flex flex-col gap-5 rounded-lg border border-slate-200 bg-slate-50 p-6 md:flex-row md:items-center md:justify-between ${className}`.trim()}
    >
      <div className="max-w-2xl">
        <p className="text-xs font-semibold uppercase tracking-[0.16em] text-cyan-700">{copy.eyebrow}</p>
        <h2 className="mt-2 text-xl font-semibold tracking-normal text-ink">{copy.title}</h2>
        <p className="mt-2 text-sm leading-6 text-slate-600">{copy.body}</p>
      </div>
      <div className="flex shrink-0 flex-col items-start gap-2">
        <a
          href={`mailto:${email}`}
          className="inline-flex min-h-11 items-center rounded-md bg-slate-950 px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-slate-800"
        >
          {copy.cta}
        </a>
        <span className="text-xs text-slate-500">{email}</span>
        <span className="text-xs text-slate-400">{copy.response}</span>
      </div>
    </div>
  );
}

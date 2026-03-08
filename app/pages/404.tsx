import Head from "next/head";
import { useRouter } from "next/router";

import { SiteChrome } from "../components/SiteChrome";
import type { Locale } from "../lib/site-data";

export default function NotFoundPage() {
  const { asPath } = useRouter();
  const locale: Locale = asPath.startsWith("/zh") ? "zh" : "en";
  const copy =
    locale === "zh"
      ? {
          eyebrow: "页面未找到",
          title: "这个页面不存在",
          intro: "链接可能已经变更，或者该内容还没有在这个语言版本中提供。",
          home: "返回首页",
          tutorials: "浏览教程",
          blog: "查看博客"
        }
      : {
          eyebrow: "Not found",
          title: "This page does not exist",
          intro: "The link may have moved, or this content may not be available in this locale yet.",
          home: "Back home",
          tutorials: "Browse tutorials",
          blog: "Visit the blog"
        };
  const prefix = locale === "zh" ? "/zh" : "";

  return (
    <>
      <Head>
        <title>{`${copy.title} | eunomia`}</title>
        <meta name="robots" content="noindex,follow" />
      </Head>
      <SiteChrome locale={locale} eyebrow={copy.eyebrow} title={copy.title} intro={copy.intro}>
        <section className="mx-auto max-w-4xl px-5 pb-16">
          <div className="grid gap-3 rounded-[2rem] border border-slate-200 bg-white/90 p-8 shadow-panel md:grid-cols-3">
            <a
              href={`${prefix}/`}
              className="rounded-[1.5rem] border border-slate-200 px-5 py-4 transition hover:border-azure hover:shadow-sm"
            >
              {copy.home}
            </a>
            <a
              href={`${prefix}/tutorials/`}
              className="rounded-[1.5rem] border border-slate-200 px-5 py-4 transition hover:border-azure hover:shadow-sm"
            >
              {copy.tutorials}
            </a>
            <a
              href={`${prefix}/blog/`}
              className="rounded-[1.5rem] border border-slate-200 px-5 py-4 transition hover:border-azure hover:shadow-sm"
            >
              {copy.blog}
            </a>
          </div>
        </section>
      </SiteChrome>
    </>
  );
}

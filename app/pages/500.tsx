import Head from "next/head";
import { useRouter } from "next/router";

import { SiteChrome } from "../components/SiteChrome";
import type { Locale } from "../lib/site-data";

export default function ServerErrorPage() {
  const { asPath } = useRouter();
  const locale: Locale = asPath.startsWith("/zh") ? "zh" : "en";
  const copy =
    locale === "zh"
      ? {
          eyebrow: "服务器错误",
          title: "服务器无法完成此页面的渲染。",
          intro: "请尝试重新加载页面。如果错误持续出现，请返回首页并浏览其他内容，我们会尽快修复此页面。",
          home: "返回首页",
          tutorials: "浏览教程",
          blog: "查看博客"
        }
      : {
          eyebrow: "Internal error",
          title: "The server could not finish rendering this page.",
          intro: "Try reloading the route. If the error persists, return to the homepage and keep browsing another section while this page is fixed.",
          home: "Back to home",
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

import type { GetStaticPaths, GetStaticProps, InferGetStaticPropsType } from "next";

import { ArticleLayout } from "../../../components/ArticleLayout";
import { CardGrid } from "../../../components/CardGrid";
import { MarkdownContent } from "../../../components/MarkdownContent";
import { SeoHead } from "../../../components/SeoHead";
import { SiteChrome } from "../../../components/SiteChrome";
import {
  getBlogRoutes,
  loadBlogIndex,
  loadBlogPage
} from "../../../lib/content";
import { canonicalAlternates } from "../../../lib/seo";

type BlogPageProps =
  | {
      kind: "index";
      page: Awaited<ReturnType<typeof loadBlogIndex>>;
    }
  | {
      kind: "article";
      page: NonNullable<Awaited<ReturnType<typeof loadBlogPage>>>;
    };

export const getStaticPaths: GetStaticPaths = async () => ({
  paths: getBlogRoutes().map((slug) => ({
    params: {
      slug
    }
  })),
  fallback: "blocking"
});

export const getStaticProps: GetStaticProps<BlogPageProps> = async ({ params }) => {
  const slug = Array.isArray(params?.slug) ? params.slug : [];

  if (!slug.length) {
    return {
      props: {
        kind: "index",
        page: await loadBlogIndex("zh")
      }
    };
  }

  const page = await loadBlogPage(slug, "zh");
  if (!page) {
    return {
      notFound: true
    };
  }

  return {
    props: {
      kind: "article",
      page
    }
  };
};

export default function ZhBlogPage({
  kind,
  page
}: InferGetStaticPropsType<typeof getStaticProps>) {
  return (
    <>
      <SeoHead
        title={page.title}
        description={page.description}
        path={page.path}
        alternates={canonicalAlternates(page.alternates.en, page.alternates.zh)}
      />
      <SiteChrome locale="zh" eyebrow="博客" title={page.title} intro={page.description}>
        {kind === "index" ? (
          <>
            <section className="mx-auto max-w-4xl px-5 pb-10">
              <article className="rounded-[2rem] border border-white/70 bg-white/90 p-8 shadow-panel md:p-10">
                <MarkdownContent html={page.introHtml} />
              </article>
            </section>
            <CardGrid cards={page.cards} />
          </>
        ) : (
          <ArticleLayout title={page.title} description={page.description} sourceHref={page.sourcePath}>
            <MarkdownContent html={page.html} />
          </ArticleLayout>
        )}
      </SiteChrome>
    </>
  );
}

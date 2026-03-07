import type { GetStaticPaths, GetStaticProps, InferGetStaticPropsType } from "next";

import { ArticleLayout } from "../../components/ArticleLayout";
import { CardGrid } from "../../components/CardGrid";
import { MarkdownContent } from "../../components/MarkdownContent";
import { SeoHead } from "../../components/SeoHead";
import { SiteChrome } from "../../components/SiteChrome";
import {
  getTutorialRoutes,
  loadTutorialIndex,
  loadTutorialPage
} from "../../lib/content";
import { canonicalAlternates } from "../../lib/seo";

type TutorialsPageProps =
  | {
      kind: "index";
      page: Awaited<ReturnType<typeof loadTutorialIndex>>;
    }
  | {
      kind: "article";
      page: NonNullable<Awaited<ReturnType<typeof loadTutorialPage>>>;
    };

export const getStaticPaths: GetStaticPaths = async () => ({
  paths: getTutorialRoutes("en").map((slug) => ({
    params: {
      slug
    }
  })),
  fallback: "blocking"
});

export const getStaticProps: GetStaticProps<TutorialsPageProps> = async ({ params }) => {
  const slug = Array.isArray(params?.slug) ? params.slug : [];

  if (!slug.length) {
    return {
      props: {
        kind: "index",
        page: await loadTutorialIndex("en")
      }
    };
  }

  const page = await loadTutorialPage(slug, "en");
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

export default function TutorialsPage({
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
      <SiteChrome locale="en" eyebrow="Tutorials" title={page.title} intro={page.description}>
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

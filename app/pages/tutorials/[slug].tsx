import type { GetStaticPaths, GetStaticProps, InferGetStaticPropsType } from "next";

import { ArticleLayout } from "../../components/ArticleLayout";
import { SeoHead } from "../../components/SeoHead";
import { SiteChrome } from "../../components/SiteChrome";
import { canonicalAlternates } from "../../lib/seo";
import { absoluteUrl } from "../../lib/seo";
import { getTutorial, siteConfig, tutorialArticles, tutorialPath } from "../../lib/site-data";

export const getStaticPaths: GetStaticPaths = async () => {
  return {
    paths: tutorialArticles.map((article) => ({ params: { slug: article.slug } })),
    fallback: false
  };
};

export const getStaticProps: GetStaticProps<{
  slug: string;
}> = async ({ params }) => {
  return {
    props: {
      slug: String(params?.slug)
    }
  };
};

export default function TutorialArticlePage({
  slug
}: InferGetStaticPropsType<typeof getStaticProps>) {
  const article = getTutorial(slug);

  if (!article) {
    return null;
  }

  return (
    <>
      <SeoHead
        title={article.title}
        description={article.description}
        path={tutorialPath(article.slug)}
        alternates={canonicalAlternates(tutorialPath(article.slug), "/zh/tutorials/")}
      />
      <SiteChrome
        locale="en"
        eyebrow="Tutorial Article"
        title={article.title}
        intro={article.summary}
      >
        <ArticleLayout
          title={article.title}
          description={article.description}
          sourceHref={`${siteConfig.repoUrl}/tree/main/${article.sourcePath}`}
        >
          {article.body.map((paragraph) => (
            <p key={paragraph}>{paragraph}</p>
          ))}
          <p>
            Canonical URL: <code>{absoluteUrl(tutorialPath(article.slug))}</code>
          </p>
        </ArticleLayout>
      </SiteChrome>
    </>
  );
}

import type { GetStaticPaths, GetStaticProps, InferGetStaticPropsType } from "next";

import { ArticleLayout } from "../../../components/ArticleLayout";
import { MarkdownContent } from "../../../components/MarkdownContent";
import { SeoHead } from "../../../components/SeoHead";
import { SiteChrome } from "../../../components/SiteChrome";
import {
  getGenericSectionRoutes,
  loadSectionPage
} from "../../../lib/content";
import { canonicalAlternates } from "../../../lib/seo";

export const getStaticPaths: GetStaticPaths = async () => ({
  paths: getGenericSectionRoutes().map((route) => ({
    params: {
      section: route.section,
      slug: route.slug
    }
  })),
  fallback: "blocking"
});

export const getStaticProps: GetStaticProps = async ({ params }) => {
  const section = typeof params?.section === "string" ? params.section : "";
  const slug = Array.isArray(params?.slug) ? params.slug : [];
  const page = await loadSectionPage(section, slug, "zh");

  if (!page) {
    return {
      notFound: true
    };
  }

  return {
    props: {
      page,
      section
    }
  };
};

export default function ZhGenericSectionPage({
  page,
  section
}: InferGetStaticPropsType<typeof getStaticProps>) {
  return (
    <>
      <SeoHead
        title={page.title}
        description={page.description}
        path={page.path}
        alternates={canonicalAlternates(page.alternates.en, page.alternates.zh)}
      />
      <SiteChrome locale="zh" eyebrow={section} title={page.title} intro={page.description}>
        <ArticleLayout title={page.title} description={page.description} sourceHref={page.sourcePath}>
          <MarkdownContent html={page.html} />
        </ArticleLayout>
      </SiteChrome>
    </>
  );
}

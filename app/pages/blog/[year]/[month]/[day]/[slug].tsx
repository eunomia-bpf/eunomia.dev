import type { GetStaticPaths, GetStaticProps, InferGetStaticPropsType } from "next";

import { ArticleLayout } from "../../../../../components/ArticleLayout";
import { SeoHead } from "../../../../../components/SeoHead";
import { SiteChrome } from "../../../../../components/SiteChrome";
import { canonicalAlternates } from "../../../../../lib/seo";
import { blogPath, blogPosts, getBlogPost, siteConfig } from "../../../../../lib/site-data";

export const getStaticPaths: GetStaticPaths = async () => {
  return {
    paths: blogPosts.map((post) => ({
      params: {
        year: post.year,
        month: post.month,
        day: post.day,
        slug: post.slug
      }
    })),
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

export default function BlogArticlePage({
  slug
}: InferGetStaticPropsType<typeof getStaticProps>) {
  const post = getBlogPost(slug);

  if (!post) {
    return null;
  }

  return (
    <>
      <SeoHead
        title={post.title}
        description={post.description}
        path={blogPath(post)}
        alternates={canonicalAlternates(blogPath(post), "/zh/blog/")}
      />
      <SiteChrome
        locale="en"
        eyebrow="Blog Article"
        title={post.title}
        intro={post.excerpt}
      >
        <ArticleLayout
          title={post.title}
          description={post.description}
          sourceHref={`${siteConfig.repoUrl}/tree/main/${post.sourcePath}`}
        >
          {post.body.map((paragraph) => (
            <p key={paragraph}>{paragraph}</p>
          ))}
        </ArticleLayout>
      </SiteChrome>
    </>
  );
}

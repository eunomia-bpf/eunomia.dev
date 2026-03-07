import type { GetStaticPaths, GetStaticProps } from "next";

import {
  getBlogRoutes,
  loadBlogIndex,
  loadBlogPage
} from "../../../lib/content";
import { buildSlugStaticPaths, CollectionPageView, loadCollectionStaticProps, type CollectionPageProps } from "../../../lib/page-factories";

type BlogPageProps = CollectionPageProps<
  Awaited<ReturnType<typeof loadBlogIndex>>,
  NonNullable<Awaited<ReturnType<typeof loadBlogPage>>>
>;

export const getStaticPaths: GetStaticPaths = async () => buildSlugStaticPaths(getBlogRoutes());

export const getStaticProps: GetStaticProps<BlogPageProps> = async ({ params }) =>
  loadCollectionStaticProps(params, {
    loadIndex: () => loadBlogIndex("zh"),
    loadArticle: (slug) => loadBlogPage(slug, "zh")
  });

export default function ZhBlogPage(props: BlogPageProps) {
  return <CollectionPageView {...props} locale="zh" eyebrow="博客" />;
}

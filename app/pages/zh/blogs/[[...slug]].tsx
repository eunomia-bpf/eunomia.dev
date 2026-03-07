import type { GetStaticPaths, GetStaticProps } from "next";

import {
  getLegacyBlogRoutes,
  loadLegacyBlogIndex,
  loadLegacyBlogPage
} from "../../../lib/content";
import { buildSlugStaticPaths, CollectionPageView, loadCollectionStaticProps, type CollectionPageProps } from "../../../lib/page-factories";

type LegacyBlogPageProps = CollectionPageProps<
  Awaited<ReturnType<typeof loadLegacyBlogIndex>>,
  NonNullable<Awaited<ReturnType<typeof loadLegacyBlogPage>>>
>;

export const getStaticPaths: GetStaticPaths = async () => buildSlugStaticPaths(getLegacyBlogRoutes());

export const getStaticProps: GetStaticProps<LegacyBlogPageProps> = async ({ params }) =>
  loadCollectionStaticProps(params, {
    loadIndex: () => loadLegacyBlogIndex("zh"),
    loadArticle: (slug) => loadLegacyBlogPage(slug, "zh")
  });

export default function ZhLegacyBlogPage(props: LegacyBlogPageProps) {
  return <CollectionPageView {...props} locale="zh" eyebrow="旧博客" />;
}

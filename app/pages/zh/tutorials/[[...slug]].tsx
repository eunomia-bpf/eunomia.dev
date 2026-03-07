import type { GetStaticPaths, GetStaticProps } from "next";

import {
  getTutorialRoutes,
  loadTutorialIndex,
  loadTutorialPage
} from "../../../lib/content";
import { buildSlugStaticPaths, CollectionPageView, loadCollectionStaticProps, type CollectionPageProps } from "../../../lib/page-factories";

type TutorialsPageProps = CollectionPageProps<
  Awaited<ReturnType<typeof loadTutorialIndex>>,
  NonNullable<Awaited<ReturnType<typeof loadTutorialPage>>>
>;

export const getStaticPaths: GetStaticPaths = async () => buildSlugStaticPaths(getTutorialRoutes("zh"));

export const getStaticProps: GetStaticProps<TutorialsPageProps> = async ({ params }) =>
  loadCollectionStaticProps(params, {
    loadIndex: () => loadTutorialIndex("zh"),
    loadArticle: (slug) => loadTutorialPage(slug, "zh")
  });

export default function ZhTutorialsPage(props: TutorialsPageProps) {
  return <CollectionPageView {...props} locale="zh" eyebrow="教程" />;
}

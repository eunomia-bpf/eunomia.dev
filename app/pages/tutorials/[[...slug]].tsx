import type { GetStaticPaths, GetStaticProps } from "next";

import {
  getTutorialRoutes,
  loadTutorialIndex,
  loadTutorialPage
} from "../../lib/content";
import { buildSlugStaticPaths, CollectionPageView, loadCollectionStaticProps, type CollectionPageProps } from "../../lib/page-factories";

type TutorialsPageProps = CollectionPageProps<
  Awaited<ReturnType<typeof loadTutorialIndex>>,
  NonNullable<Awaited<ReturnType<typeof loadTutorialPage>>>
>;

export const getStaticPaths: GetStaticPaths = async () => buildSlugStaticPaths(getTutorialRoutes("en"));

export const getStaticProps: GetStaticProps<TutorialsPageProps> = async ({ params }) =>
  loadCollectionStaticProps(params, {
    loadIndex: () => loadTutorialIndex("en"),
    loadArticle: (slug) => loadTutorialPage(slug, "en")
  });

export default function TutorialsPage(props: TutorialsPageProps) {
  return <CollectionPageView {...props} locale="en" eyebrow="Tutorials" />;
}

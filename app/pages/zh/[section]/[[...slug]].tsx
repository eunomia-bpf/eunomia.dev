import type { GetStaticPaths, GetStaticProps } from "next";

import {
  getGenericSectionRoutes,
  loadSectionPage
} from "../../../lib/content";
import { buildSectionStaticPaths, loadSectionStaticProps, SectionPageView } from "../../../lib/page-factories";

type SectionPageProps = {
  page: NonNullable<Awaited<ReturnType<typeof loadSectionPage>>>;
  section: string;
};

export const getStaticPaths: GetStaticPaths = async () => buildSectionStaticPaths(getGenericSectionRoutes("zh"));

export const getStaticProps: GetStaticProps<SectionPageProps> = async ({ params }) =>
  loadSectionStaticProps(params, (section, slug) => loadSectionPage(section, slug, "zh"));

export default function ZhGenericSectionPage({ page, section }: SectionPageProps) {
  return <SectionPageView page={page} section={section} locale="zh" />;
}

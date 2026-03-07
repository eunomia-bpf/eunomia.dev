import type { GetStaticProps } from "next";

import { loadHomePage } from "../lib/content";
import { HomePageView } from "../lib/page-factories";

type HomePageProps = {
  page: Awaited<ReturnType<typeof loadHomePage>>;
};

export const getStaticProps: GetStaticProps<HomePageProps> = async () => ({
  props: {
    page: await loadHomePage("en")
  }
});

export default function HomePage({ page }: HomePageProps) {
  return <HomePageView page={page} locale="en" eyebrow="Home" />;
}

import type { GetStaticProps, InferGetStaticPropsType } from "next";

import { CardGrid } from "../components/CardGrid";
import { SeoHead } from "../components/SeoHead";
import { SiteChrome } from "../components/SiteChrome";
import { loadHomePage } from "../lib/content";
import { canonicalAlternates } from "../lib/seo";

export const getStaticProps: GetStaticProps = async () => ({
  props: {
    page: await loadHomePage("en")
  }
});

export default function HomePage({ page }: InferGetStaticPropsType<typeof getStaticProps>) {
  return (
    <>
      <SeoHead
        title={page.title}
        description={page.description}
        path={page.path}
        alternates={canonicalAlternates(page.alternates.en, page.alternates.zh)}
      />
      <SiteChrome locale="en" eyebrow="Home" title={page.title} intro={page.intro}>
        <CardGrid cards={page.cards} />
      </SiteChrome>
    </>
  );
}

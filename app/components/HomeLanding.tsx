import type { HomePageData } from "../lib/page-factories";
import { MarkdownContent } from "./MarkdownContent";

type HomeLandingProps = {
  page: HomePageData;
};

export function HomeLanding({ page }: HomeLandingProps) {
  return (
    <article className="home-landing content-copy max-w-none">
      <MarkdownContent html={page.bodyHtml} />
    </article>
  );
}

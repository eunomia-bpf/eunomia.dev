import type { HomePageData } from "../lib/page-factories";
import { MarkdownContent } from "./MarkdownContent";

type HomeLandingProps = {
  page: HomePageData;
};

export function HomeLanding({ page }: HomeLandingProps) {
  return (
    <>
      {page.inlineStyles.map((css, index) => (
        <style
          // Home page styles are repository-authored content extracted from docs/index.md.
          dangerouslySetInnerHTML={{ __html: css }}
          key={`home-inline-style-${index}`}
        />
      ))}
      <article className="content-copy max-w-none">
        <MarkdownContent html={page.bodyHtml} />
      </article>
    </>
  );
}

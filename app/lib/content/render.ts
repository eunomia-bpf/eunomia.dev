import rehypeRaw from "rehype-raw";
import rehypeSanitize from "rehype-sanitize";
import rehypeSlug from "rehype-slug";
import rehypeStringify from "rehype-stringify";
import remarkGfm from "remark-gfm";
import remarkParse from "remark-parse";
import remarkRehype from "remark-rehype";
import { unified } from "unified";
import { visit } from "unist-util-visit";

import type { Locale } from "../site-data";
import { createRehypeRewriter } from "./rewrite";
import { parseMarkdown } from "./markdown";
import { markdownSanitizeSchema } from "./sanitize";
import type { HeadingEntry, RenderedMarkdown } from "./types";

type HastNode = {
  type?: string;
  value?: unknown;
  tagName?: string;
  properties?: {
    id?: unknown;
  };
  children?: HastNode[];
};

function collapseHeadingText(value: string): string {
  return value.replace(/\s+/g, " ").trim();
}

function extractNodeText(node: HastNode): string {
  if (node.type === "text" && typeof node.value === "string") {
    return node.value;
  }

  return (node.children ?? []).map(extractNodeText).join("");
}

function createHeadingCollector(headings: HeadingEntry[]) {
  return function headingCollector() {
    return function collect(tree: unknown) {
      visit(tree as Parameters<typeof visit>[0], "element", (node: HastNode) => {
        if (!node.tagName) {
          return;
        }

        const depth = /^h([2-6])$/i.test(node.tagName) ? Number.parseInt(node.tagName.slice(1), 10) : 0;
        const id = typeof node.properties?.id === "string" ? node.properties.id : undefined;
        const text = collapseHeadingText(extractNodeText(node));

        if (!depth || !id || !text) {
          return;
        }

        headings.push({
          id,
          text,
          depth
        });
      });
    };
  };
}

export async function renderMarkdownDocumentBody(
  markdown: string,
  relativePath: string,
  locale: Locale
): Promise<RenderedMarkdown> {
  const headings: HeadingEntry[] = [];
  const processed = await unified()
    .use(remarkParse)
    .use(remarkGfm)
    .use(remarkRehype, {
      allowDangerousHtml: true
    })
    .use(rehypeRaw)
    .use(rehypeSanitize, markdownSanitizeSchema)
    .use(rehypeSlug)
    .use(createHeadingCollector(headings))
    .use(createRehypeRewriter(relativePath, locale))
    .use(rehypeStringify)
    .process(markdown);

  return {
    html: String(processed),
    headings
  };
}

export async function renderMarkdownBody(markdown: string, relativePath: string, locale: Locale): Promise<string> {
  const rendered = await renderMarkdownDocumentBody(markdown, relativePath, locale);
  return rendered.html;
}

export async function renderMarkdown(relativePath: string, locale: Locale): Promise<string> {
  const parsed = parseMarkdown(relativePath);
  return renderMarkdownBody(parsed.body, relativePath, locale);
}

export async function renderMarkdownDocument(relativePath: string, locale: Locale): Promise<RenderedMarkdown> {
  const parsed = parseMarkdown(relativePath);
  return renderMarkdownDocumentBody(parsed.body, relativePath, locale);
}

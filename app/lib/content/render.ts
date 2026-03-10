import rehypePrettyCode from "rehype-pretty-code";
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
import { createCodeLanguageNormalizer, prettyCodeOptions } from "./highlight";
import { splitMaterialBlocks } from "./material-blocks";
import { createRehypeRewriter } from "./rewrite";
import { parseMarkdown } from "./markdown";
import { markdownSanitizeSchema } from "./sanitize";
import type { HeadingEntry, RenderedMarkdown } from "./types";
import { escapeXml } from "../utils";

type HastNode = {
  type?: string;
  value?: unknown;
  tagName?: string;
  properties?: Record<string, unknown>;
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

function normalizeClassNames(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value.filter((item): item is string => typeof item === "string");
  }

  if (typeof value === "string") {
    return value.split(/\s+/).filter(Boolean);
  }

  return [];
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

function createMermaidFenceTransformer() {
  return function mermaidFenceTransformer() {
    return function transform(tree: unknown) {
      visit(
        tree as Parameters<typeof visit>[0],
        "element",
        (node: HastNode, index: number | undefined, parent: HastNode | undefined) => {
          if (node.tagName !== "pre" || !parent || index === undefined) {
            return;
          }

          const codeNode = (node.children ?? []).find((child) => child.tagName === "code");
          if (!codeNode) {
            return;
          }

          const classNames = normalizeClassNames(codeNode.properties?.className);
          if (!classNames.includes("language-mermaid")) {
            return;
          }

          const source = extractNodeText(codeNode).replace(/\n+$/, "");
          if (!source.trim()) {
            return;
          }

          parent.children ??= [];
          parent.children[index] = {
            type: "element",
            tagName: "pre",
            properties: {
              className: ["mermaid-diagram"],
              "data-mermaid-diagram": ""
            },
            children: [
              {
                type: "text",
                value: source
              }
            ]
          };
        }
      );
    };
  };
}

async function renderMarkdownChunk(
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
    .use(createCodeLanguageNormalizer())
    .use(createMermaidFenceTransformer())
    .use(rehypePrettyCode, prettyCodeOptions)
    .use(createRehypeRewriter(relativePath, locale))
    .use(rehypeStringify)
    .process(markdown);

  return {
    html: String(processed),
    headings
  };
}

type RenderState = {
  tabGroupIndex: number;
};

async function renderCompositeMarkdown(
  markdown: string,
  relativePath: string,
  locale: Locale,
  state: RenderState
): Promise<RenderedMarkdown> {
  const blocks = splitMaterialBlocks(markdown);
  if (blocks.length === 1 && blocks[0]?.type === "markdown") {
    return renderMarkdownChunk(blocks[0].content, relativePath, locale);
  }

  const htmlParts: string[] = [];
  const headings: HeadingEntry[] = [];

  for (const block of blocks) {
    if (block.type === "markdown") {
      if (!block.content.trim()) {
        continue;
      }

      const rendered = await renderMarkdownChunk(block.content, relativePath, locale);
      htmlParts.push(rendered.html);
      headings.push(...rendered.headings);
      continue;
    }

    if (block.type === "admonition") {
      const rendered = await renderCompositeMarkdown(block.content, relativePath, locale, state);
      headings.push(...rendered.headings);
      const kindClass = escapeXml(block.kind.toLowerCase());
      const title = escapeXml(block.title);
      const bodyHtml = `<div class="content-admonition-body">${rendered.html}</div>`;

      if (block.collapsible) {
        htmlParts.push(
          `<details class="content-admonition content-admonition-${kindClass}"${block.open ? " open" : ""}><summary>${title}</summary>${bodyHtml}</details>`
        );
      } else {
        htmlParts.push(
          `<section class="content-admonition content-admonition-${kindClass}"><p class="content-admonition-title">${title}</p>${bodyHtml}</section>`
        );
      }
      continue;
    }

    const tabGroupId = `content-tabs-${state.tabGroupIndex++}`;
    const tabHtml = await Promise.all(
      block.items.map(async (item, index) => {
        const rendered = await renderCompositeMarkdown(item.content, relativePath, locale, state);
        headings.push(...rendered.headings);
        const inputId = `${tabGroupId}-${index}`;
        return `<div class="content-tab"><input class="content-tab-input" type="radio" name="${tabGroupId}" id="${inputId}"${
          index === 0 ? " checked" : ""
        }><label class="content-tab-label" for="${inputId}">${escapeXml(item.label)}</label><div class="content-tab-panel">${rendered.html}</div></div>`;
      })
    );

    htmlParts.push(`<div class="content-tabs">${tabHtml.join("")}</div>`);
  }

  return {
    html: htmlParts.join("\n"),
    headings
  };
}

export async function renderMarkdownDocumentBody(
  markdown: string,
  relativePath: string,
  locale: Locale
): Promise<RenderedMarkdown> {
  return renderCompositeMarkdown(markdown, relativePath, locale, {
    tabGroupIndex: 0
  });
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

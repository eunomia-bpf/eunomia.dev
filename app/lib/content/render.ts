import rehypeRaw from "rehype-raw";
import rehypeSanitize from "rehype-sanitize";
import rehypeSlug from "rehype-slug";
import rehypeStringify from "rehype-stringify";
import remarkGfm from "remark-gfm";
import remarkParse from "remark-parse";
import remarkRehype from "remark-rehype";
import { unified } from "unified";

import type { Locale } from "../site-data";
import { createRehypeRewriter } from "./rewrite";
import { parseMarkdown } from "./markdown";
import { markdownSanitizeSchema } from "./sanitize";

export async function renderMarkdownBody(markdown: string, relativePath: string, locale: Locale): Promise<string> {
  const processed = await unified()
    .use(remarkParse)
    .use(remarkGfm)
    .use(remarkRehype, {
      allowDangerousHtml: true
    })
    .use(rehypeRaw)
    .use(rehypeSanitize, markdownSanitizeSchema)
    .use(rehypeSlug)
    .use(createRehypeRewriter(relativePath, locale))
    .use(rehypeStringify)
    .process(markdown);

  return String(processed);
}

export async function renderMarkdown(relativePath: string, locale: Locale): Promise<string> {
  const parsed = parseMarkdown(relativePath);
  return renderMarkdownBody(parsed.body, relativePath, locale);
}

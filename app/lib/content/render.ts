import rehypeRaw from "rehype-raw";
import rehypeSlug from "rehype-slug";
import rehypeStringify from "rehype-stringify";
import remarkGfm from "remark-gfm";
import remarkParse from "remark-parse";
import remarkRehype from "remark-rehype";
import { unified } from "unified";

import type { Locale } from "../site-data";
import { createRehypeRewriter } from "./rewrite";
import { parseMarkdown } from "./markdown";

export async function renderMarkdown(relativePath: string, locale: Locale): Promise<string> {
  const parsed = parseMarkdown(relativePath);
  const processed = await unified()
    .use(remarkParse)
    .use(remarkGfm)
    .use(remarkRehype, {
      allowDangerousHtml: true
    })
    .use(rehypeRaw)
    .use(rehypeSlug)
    .use(createRehypeRewriter(relativePath, locale))
    .use(rehypeStringify)
    .process(parsed.body);

  return String(processed);
}

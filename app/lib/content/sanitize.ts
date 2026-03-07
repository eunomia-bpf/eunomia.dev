import type { Schema } from "hast-util-sanitize";
import { defaultSchema } from "rehype-sanitize";

function extendStringList(
  current: readonly string[] | null | undefined,
  additions: readonly string[]
): string[] {
  return Array.from(new Set([...(current ?? []), ...additions]));
}

const defaultAttributes = defaultSchema.attributes ?? {};
const defaultProtocols = defaultSchema.protocols ?? {};

export const markdownSanitizeSchema: Schema = {
  ...defaultSchema,
  tagNames: extendStringList(defaultSchema.tagNames, ["div"]),
  attributes: {
    ...defaultAttributes,
    "*": [...(defaultAttributes["*"] ?? []), "id"],
    a: [...(defaultAttributes.a ?? []), "className", "rel", "target"],
    div: [...(defaultAttributes.div ?? []), "align", "className"],
    img: [...(defaultAttributes.img ?? []), "align", "className", "height", "width"],
    p: [...(defaultAttributes.p ?? []), "align", "className"]
  },
  protocols: {
    ...defaultProtocols,
    href: extendStringList(defaultProtocols.href, ["http", "https", "mailto", "tel"]),
    src: extendStringList(defaultProtocols.src, ["http", "https"])
  }
};

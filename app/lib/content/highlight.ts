import rehypePrettyCode from "rehype-pretty-code";
import { visit } from "unist-util-visit";

const languageAliases: Record<string, string> = {
  bt: "plaintext",
  conf: "plaintext",
  console: "shellsession",
  cuda: "cpp",
  make: "makefile",
  plain: "plaintext",
  plaintext: "plaintext",
  ptx: "asm",
  shell: "bash",
  sh: "bash",
  text: "plaintext",
  txt: "plaintext",
  yml: "yaml"
};

type HastElement = {
  tagName?: string;
  properties?: {
    className?: unknown;
  };
  children?: HastElement[];
};

function normalizeClassNames(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value.filter((item): item is string => typeof item === "string");
  }

  if (typeof value === "string") {
    return value.split(/\s+/).filter(Boolean);
  }

  return [];
}

function normalizeLanguage(value: string): string {
  return languageAliases[value.toLowerCase()] ?? value.toLowerCase();
}

export function createCodeLanguageNormalizer() {
  return function codeLanguageNormalizer() {
    return function normalize(tree: unknown) {
      visit(
        tree as Parameters<typeof visit>[0],
        "element",
        (node: HastElement, index: number | undefined, parent: HastElement | undefined) => {
          if (node.tagName !== "code" || parent?.tagName !== "pre") {
            return;
          }

          const classNames = normalizeClassNames(node.properties?.className);
          const nextClassNames = classNames.map((className) => {
            if (!className.startsWith("language-")) {
              return className;
            }

            return `language-${normalizeLanguage(className.slice("language-".length))}`;
          });

          if (node.properties) {
            node.properties.className = nextClassNames;
          }
        }
      );
    };
  };
}

export const prettyCodeOptions = {
  theme: "github-dark-dimmed",
  keepBackground: false,
  bypassInlineCode: true,
  defaultLang: {
    inline: "plaintext"
  },
  onVisitLine(node: {
    children?: Array<{
      type: string;
      value?: string;
    }>;
  }) {
    if (!node.children?.length) {
      node.children = [{ type: "text", value: " " }];
    }
  }
} satisfies Parameters<typeof rehypePrettyCode>[0];

type MarkdownBlock =
  | {
      type: "markdown";
      content: string;
    }
  | {
      type: "admonition";
      kind: string;
      title: string;
      collapsible: boolean;
      open: boolean;
      content: string;
    }
  | {
      type: "tabs";
      items: Array<{
        label: string;
        content: string;
      }>;
    };

const admonitionPattern = /^( {0,3})(!!!|\?\?\?\+?)[ \t]+([a-zA-Z0-9_-]+)(?:[ \t]+"([^"]+)")?[ \t]*$/;
const tabPattern = /^( {0,3})===\s+"([^"]+)"\s*$/;

function countIndent(line: string): number {
  let width = 0;
  for (const character of line) {
    if (character === " ") {
      width += 1;
      continue;
    }

    if (character === "\t") {
      width += 4;
      continue;
    }

    break;
  }

  return width;
}

function stripIndent(line: string, indent: number): string {
  let remaining = indent;
  let index = 0;

  while (remaining > 0 && index < line.length) {
    const character = line[index];
    if (character === " ") {
      remaining -= 1;
      index += 1;
      continue;
    }

    if (character === "\t") {
      remaining -= 4;
      index += 1;
      continue;
    }

    break;
  }

  return line.slice(index);
}

function normalizeBlockContent(lines: string[]): string {
  const normalized = [...lines];
  while (normalized.length && !normalized[0]?.trim()) {
    normalized.shift();
  }
  while (normalized.length && !normalized[normalized.length - 1]?.trim()) {
    normalized.pop();
  }
  return normalized.join("\n");
}

function defaultAdmonitionTitle(kind: string): string {
  return kind
    .split(/[_-]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function flushMarkdownBlock(blocks: MarkdownBlock[], buffer: string[]) {
  const content = normalizeBlockContent(buffer);
  if (content) {
    blocks.push({
      type: "markdown",
      content
    });
  }
  buffer.length = 0;
}

function collectIndentedBlock(lines: string[], startIndex: number, parentIndent: number) {
  const requiredIndent = parentIndent + 4;
  const content: string[] = [];
  let index = startIndex;

  while (index < lines.length) {
    const line = lines[index] ?? "";
    if (!line.trim()) {
      content.push("");
      index += 1;
      continue;
    }

    const indent = countIndent(line);
    if (indent < requiredIndent) {
      break;
    }

    content.push(stripIndent(line, requiredIndent));
    index += 1;
  }

  return {
    content: normalizeBlockContent(content),
    nextIndex: index
  };
}

function matchAdmonition(line: string) {
  const match = line.match(admonitionPattern);
  if (!match) {
    return null;
  }

  const marker = match[2] ?? "!!!";
  return {
    indent: countIndent(match[1] ?? ""),
    kind: match[3] ?? "note",
    title: (match[4] ?? "").trim(),
    collapsible: marker.startsWith("???"),
    open: marker === "???+"
  };
}

function matchTab(line: string) {
  const match = line.match(tabPattern);
  if (!match) {
    return null;
  }

  return {
    indent: countIndent(match[1] ?? ""),
    label: (match[2] ?? "").trim()
  };
}

export function splitMaterialBlocks(markdown: string): MarkdownBlock[] {
  const lines = markdown.replace(/\r\n?/g, "\n").split("\n");
  const blocks: MarkdownBlock[] = [];
  const buffer: string[] = [];
  let index = 0;

  while (index < lines.length) {
    const line = lines[index] ?? "";
    const admonition = matchAdmonition(line);
    if (admonition) {
      flushMarkdownBlock(blocks, buffer);
      const { content, nextIndex } = collectIndentedBlock(lines, index + 1, admonition.indent);
      blocks.push({
        type: "admonition",
        kind: admonition.kind,
        title: admonition.title || defaultAdmonitionTitle(admonition.kind),
        collapsible: admonition.collapsible,
        open: admonition.open,
        content
      });
      index = nextIndex;
      continue;
    }

    const tab = matchTab(line);
    if (tab) {
      flushMarkdownBlock(blocks, buffer);
      const items: Array<{ label: string; content: string }> = [];
      let groupIndex = index;

      while (groupIndex < lines.length) {
        const groupLine = lines[groupIndex] ?? "";
        const currentTab = matchTab(groupLine);
        if (!currentTab || currentTab.indent !== tab.indent) {
          break;
        }

        const { content, nextIndex } = collectIndentedBlock(lines, groupIndex + 1, currentTab.indent);
        items.push({
          label: currentTab.label,
          content
        });
        groupIndex = nextIndex;

        while (groupIndex < lines.length && !lines[groupIndex]?.trim()) {
          const nextTab = matchTab(lines[groupIndex + 1] ?? "");
          if (!nextTab || nextTab.indent !== tab.indent) {
            break;
          }
          groupIndex += 1;
        }
      }

      blocks.push({
        type: "tabs",
        items
      });
      index = groupIndex;
      continue;
    }

    buffer.push(line);
    index += 1;
  }

  flushMarkdownBlock(blocks, buffer);
  return blocks;
}

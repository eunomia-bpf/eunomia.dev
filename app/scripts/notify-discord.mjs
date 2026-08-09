import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const repoRoot = path.resolve(import.meta.dirname, "..", "..");
const manifestPath = path.join(repoRoot, "app", ".generated", "content", "manifest.json");
const siteUrl = "https://eunomia.dev";

function resolveRevisionRange() {
  const head = process.env.HEAD_SHA || "HEAD";
  const requestedBase = process.env.BASE_SHA;
  const base = requestedBase && !/^0+$/.test(requestedBase) ? requestedBase : `${head}^`;
  return { base, head };
}

function changedMarkdown(base, head) {
  const output = execFileSync(
    "git",
    ["diff", "--name-only", "--diff-filter=AM", base, head, "--", "docs"],
    { cwd: repoRoot, encoding: "utf8" }
  );

  return new Set(
    output
      .split(/\r?\n/)
      .filter((file) => file.endsWith(".md"))
      .map((file) => file.replace(/^docs\//, ""))
  );
}

function readManifest() {
  const payload = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
  if (!Array.isArray(payload.manifest)) {
    throw new Error("Generated content manifest is missing its manifest array");
  }
  return payload.manifest;
}

function changedRecords(files, manifest) {
  return manifest.filter((record) => {
    const sources = new Set([
      ...Object.values(record.sourceByLocale ?? {}),
      ...(record.sourceAliases ?? [])
    ]);
    return [...sources].some((source) => files.has(source));
  });
}

function primarySource(record, files) {
  const candidates = [record.sourceByLocale?.en, record.sourceByLocale?.zh].filter(Boolean);
  return candidates.find((source) => files.has(source)) ?? candidates[0];
}

function cleanInlineMarkdown(value) {
  return value
    .replace(/<!--.*?-->/g, "")
    .replace(/!\[[^\]]*\]\([^)]*\)/g, "")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/[`*_~]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function readPage(record, files) {
  const source = primarySource(record, files);
  const markdown = fs.readFileSync(path.join(repoRoot, "docs", source), "utf8");
  const body = markdown.replace(/^---\r?\n[\s\S]*?\r?\n---\r?\n/, "");
  const title = cleanInlineMarkdown(body.match(/^#\s+(.+)$/m)?.[1] ?? record.key);
  const frontmatterDescription = markdown
    .match(/^---\r?\n[\s\S]*?^description:\s*(.+)$/m)?.[1]
    ?.replace(/^['"]|['"]$/g, "");
  const firstParagraph = body
    .split(/\r?\n\s*\r?\n/)
    .map(cleanInlineMarkdown)
    .find((paragraph) => paragraph && !paragraph.startsWith("#") && !paragraph.startsWith("- "));
  const description = cleanInlineMarkdown(frontmatterDescription ?? firstParagraph ?? "");
  const route = record.routeByLocale?.en ?? record.routeByLocale?.zh;
  return { title, description, markdown, url: new URL(route, siteUrl).toString() };
}

function destination(record) {
  if (record.kind === "blog-page") {
    return { label: "New article", webhook: process.env.DISCORD_CONTENT_WEBHOOK };
  }
  if (record.kind === "tutorial-page") {
    return { label: "New tutorial", webhook: process.env.DISCORD_CONTENT_WEBHOOK };
  }
  if (record.section === "ebpf-qa") {
    return { label: "eBPF Q&A and community digest", webhook: process.env.DISCORD_DIGEST_WEBHOOK };
  }
  if (record.section === "reports" && record.slug?.length) {
    return { label: "Community report", webhook: process.env.DISCORD_DIGEST_WEBHOOK };
  }
  return null;
}

const projectDestinations = [
  ["agentsight", "AgentSight", "DISCORD_AGENTSIGHT_WEBHOOK"],
  ["actplane", "ActPlane", "DISCORD_ACTPLANE_WEBHOOK"],
  ["akeep", "Akeep", "DISCORD_AKEEP_WEBHOOK"],
  ["bpftime", "bpftime", "DISCORD_BPFTIME_WEBHOOK"]
];

function formatMessage(label, page) {
  const summary = page.description.length > 360
    ? `${page.description.slice(0, 357).trimEnd()}...`
    : page.description;
  return [`**${label}**`, `**${page.title}**`, summary, page.url].filter(Boolean).join("\n");
}

async function post(webhook, content) {
  if (!webhook) {
    console.log("Skipping a Discord destination because its webhook secret is not configured.");
    return;
  }
  if (process.argv.includes("--dry-run")) {
    console.log(content);
    return;
  }

  const response = await fetch(webhook, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      username: "eunomia.dev",
      content,
      allowed_mentions: { parse: [] }
    })
  });
  if (!response.ok) {
    throw new Error(`Discord webhook returned HTTP ${response.status}`);
  }
}

const { base, head } = resolveRevisionRange();
const files = changedMarkdown(base, head);
const records = changedRecords(files, readManifest());

for (const record of records) {
  const primary = destination(record);
  if (!primary) continue;

  const page = readPage(record, files);
  await post(primary.webhook, formatMessage(primary.label, page));

  const searchable = `${record.key}\n${page.title}\n${page.description}`.toLowerCase();
  for (const [keyword, project, environmentName] of projectDestinations) {
    if (searchable.includes(keyword)) {
      await post(process.env[environmentName], formatMessage(`${project} update`, page));
    }
  }
}

console.log(`Discord notification scan completed for ${records.length} changed public content record(s).`);

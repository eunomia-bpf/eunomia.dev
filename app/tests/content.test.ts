import assert from "node:assert/strict";
import test from "node:test";

import { serveRawAsset } from "../lib/content/assets";
import { getBlogEntries } from "../lib/content/collections";
import { parseMarkdown } from "../lib/content/markdown";
import { getContentManifest } from "../lib/content/manifest";
import { renderMarkdown, renderMarkdownBody, renderMarkdownDocument } from "../lib/content/render";
import { docPathToRoute, getGenericSectionRoutes, listSitemapRoutes } from "../lib/content/routes";
import { searchContent } from "../lib/content/search";
import { rewriteContentUrl } from "../lib/content/rewrite";
import { resolveLocalizedSource, slugifyTitle } from "../lib/content/source";

test("resolveLocalizedSource prefers zh variant when present", () => {
  assert.equal(
    resolveLocalizedSource("tutorials/38-btf-uprobe/README.md", "zh"),
    "tutorials/38-btf-uprobe/README.zh.md"
  );
});

test("resolveLocalizedSource falls back to english variant when zh variant is missing", () => {
  assert.equal(
    resolveLocalizedSource("eunomia-bpf/setup/README.md", "zh"),
    "eunomia-bpf/setup/README.en.md"
  );
});

test("docPathToRoute maps nested tutorial docs to public tutorial routes", () => {
  assert.equal(
    docPathToRoute("tutorials/38-btf-uprobe/test-verify/README.md", "en"),
    "/tutorials/38-btf-uprobe/test-verify/"
  );
});

test("docPathToRoute normalizes .en.md docs into stable public routes", () => {
  assert.equal(
    docPathToRoute("eunomia-bpf/setup/build.en.md", "en"),
    "/eunomia-bpf/setup/build/"
  );
});

test("slugifyTitle normalizes punctuation and accents into stable slugs", () => {
  assert.equal(
    slugifyTitle("AgentCgroup: What Happens When AI Coding Agents Meet OS Resources?"),
    "agentcgroup-what-happens-when-ai-coding-agents-meet-os-resources"
  );
  assert.equal(slugifyTitle("Caf\u00e9 Noir / 2026"), "cafe-noir-2026");
});

test("blog entries derive dated slugs from parsed metadata", () => {
  const entry = getBlogEntries().find((candidate) => candidate.key === "agentcgroup-characterization");
  assert.ok(entry);
  assert.equal(entry.slug, slugifyTitle(entry.title));
  assert.deepEqual([entry.year, entry.month, entry.day], ["2026", "02", "17"]);
  assert.equal(entry.sourceByLocale.en, "blog/posts/agentcgroup-characterization.md");
  assert.equal(entry.sourceByLocale.zh, undefined);
});

test("rewriteContentUrl rewrites nested relative asset paths to the raw asset endpoint", () => {
  assert.equal(
    rewriteContentUrl("./tcpconnlat1.png", "tutorials/13-tcpconnlat/README.md", "en"),
    "/api/raw-assets/docs/tutorials/13-tcpconnlat/tcpconnlat1.png"
  );
});

test("parseMarkdown extracts stable metadata from docs content", () => {
  const page = parseMarkdown("tutorials/1-helloworld/README.md");
  assert.equal(typeof page.title, "string");
  assert.notEqual(page.title.length, 0);
  assert.equal(typeof page.description, "string");
  assert.notEqual(page.description.length, 0);
});

test("serveRawAsset resolves docs assets with the right mime type", async () => {
  const asset = await serveRawAsset("docs", ["tutorials", "13-tcpconnlat", "tcpconnlat1.png"]);
  assert.ok(asset);
  assert.equal(asset.contentType, "image/png");
});

test("content manifest keeps localized routes for english-only section pages", () => {
  const record = getContentManifest().find((candidate) => candidate.key === "section:eunomia-bpf:setup/build");
  assert.ok(record);
  assert.equal(record.sourceByLocale.en, "eunomia-bpf/setup/build.en.md");
  assert.equal(record.sourceByLocale.zh, "eunomia-bpf/setup/build.en.md");
  assert.equal(record.routeByLocale.en, "/eunomia-bpf/setup/build/");
  assert.equal(record.routeByLocale.zh, "/zh/eunomia-bpf/setup/build/");
});

test("generic section routes preserve zh-only pages without inventing english routes", () => {
  const zhRoute = getGenericSectionRoutes("zh").find(
    (candidate) =>
      candidate.section === "eunomia-bpf" && candidate.slug.join("/") === "ecli/ecli-dockerfile-usage"
  );
  const enRoute = getGenericSectionRoutes("en").find(
    (candidate) =>
      candidate.section === "eunomia-bpf" && candidate.slug.join("/") === "ecli/ecli-dockerfile-usage"
  );

  assert.deepEqual(zhRoute, {
    section: "eunomia-bpf",
    slug: ["ecli", "ecli-dockerfile-usage"]
  });
  assert.equal(enRoute, undefined);
});

test("docPathToRoute preserves the only surviving localized route for zh-only section docs", () => {
  assert.equal(
    docPathToRoute("eunomia-bpf/ecli/ecli-dockerfile-usage.zh.md", "zh"),
    "/zh/eunomia-bpf/ecli/ecli-dockerfile-usage/"
  );
  assert.equal(
    docPathToRoute("eunomia-bpf/ecli/ecli-dockerfile-usage.zh.md", "en"),
    "/zh/eunomia-bpf/ecli/ecli-dockerfile-usage/"
  );
});

test("listSitemapRoutes keeps key legacy and section routes", () => {
  const routes = new Set(listSitemapRoutes());
  assert.ok(routes.has("/zh/blogs/bpftime/"));
  assert.ok(routes.has("/eunomia-bpf/setup/build/"));
  assert.ok(routes.has("/tutorials/38-btf-uprobe/test-verify/"));
  assert.ok(routes.has("/zh/eunomia-bpf/ecli/ecli-dockerfile-usage/"));
  assert.ok(!routes.has("/eunomia-bpf/ecli/ecli-dockerfile-usage/"));
});

test("searchContent returns locale-aware tutorial results", () => {
  const enResults = searchContent("hello world", "en");
  const zhResults = searchContent("hello world", "zh");

  assert.ok(enResults.some((result) => result.href === "/tutorials/1-helloworld/"));
  assert.ok(zhResults.some((result) => result.href === "/zh/tutorials/1-helloworld/"));
});

test("renderMarkdown preserves allowed raw HTML used by current docs", async () => {
  const html = await renderMarkdown("GPTtrace/agentsight.md", "en");

  assert.match(html, /<div align="center">/);
  assert.match(html, /<img[^>]+src="https:\/\/github\.com\/eunomia-bpf\/agentsight\/raw\/master\/docs\/demo-tree\.png"/);
  assert.match(html, /<p><em>Real-time process tree visualization/);
});

test("renderMarkdownDocument extracts TOC headings from article content", async () => {
  const rendered = await renderMarkdownDocument("tutorials/38-btf-uprobe/test-verify/README.md", "en");

  assert.ok(rendered.headings.length > 0);
  assert.deepEqual(rendered.headings[0], {
    id: "usage",
    text: "Usage",
    depth: 2
  });
  assert.ok(
    rendered.headings.some(
      (heading) => heading.id === "verify-struct-data-access" && heading.text === "Verify struct data access"
    )
  );
});

test("renderMarkdownBody applies syntax highlighting to fenced code blocks", async () => {
  const html = await renderMarkdownBody("```c\nint main() { return 0; }\n```", "tutorials/fake.md", "en");

  assert.match(html, /<figure data-rehype-pretty-code-figure="">/);
  assert.match(html, /data-language="c"/);
  assert.match(html, /<span style="color:/);
});

test("renderMarkdownBody normalizes shell and plaintext-ish aliases for highlighting", async () => {
  const shellHtml = await renderMarkdownBody("```console\nsudo ./ecli run package.json\n```", "tutorials/fake.md", "en");
  const textHtml = await renderMarkdownBody("```conf\nALLOW=true\n```", "tutorials/fake.md", "en");
  const cudaHtml = await renderMarkdownBody("```cuda\n__global__ void add() {}\n```", "tutorials/fake.md", "en");

  assert.match(shellHtml, /data-language="shellsession"/);
  assert.match(textHtml, /data-language="plaintext"/);
  assert.match(cudaHtml, /data-language="cpp"/);
});

test("renderMarkdownBody rewrites local asset URLs inside allowed raw HTML", async () => {
  const html = await renderMarkdownBody(
    '<div align="center"><img src="./tcpconnlat1.png" alt="demo" width="800"></div>',
    "tutorials/13-tcpconnlat/README.md",
    "en"
  );

  assert.match(html, /\/api\/raw-assets\/docs\/tutorials\/13-tcpconnlat\/tcpconnlat1\.png/);
});

test("renderMarkdownBody strips unsafe raw HTML and dangerous URLs", async () => {
  const html = await renderMarkdownBody(
    '<script>alert(1)</script><a href="javascript:alert(1)">bad</a><img src="javascript:alert(2)" alt="x">',
    "blog/posts/fake.md",
    "en"
  );

  assert.doesNotMatch(html, /<script/i);
  assert.doesNotMatch(html, /javascript:/i);
  assert.match(html, />bad<\/a>/);
});

test("renderMarkdownBody keeps fenced HTML samples escaped", async () => {
  const html = await renderMarkdownBody("```html\n<script>alert(1)</script>\n```", "blog/posts/fake.md", "en");

  assert.match(html, /data-language="html"/);
  assert.match(html, /&#x3C;/);
  assert.match(html, /script/);
  assert.doesNotMatch(html, /<script>alert\(1\)<\/script>/);
});

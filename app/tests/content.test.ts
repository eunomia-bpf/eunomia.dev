import assert from "node:assert/strict";
import test from "node:test";

import { serveRawAsset } from "../lib/content/assets";
import { getBlogEntries } from "../lib/content/collections";
import { parseMarkdown } from "../lib/content/markdown";
import { getContentManifest } from "../lib/content/manifest";
import { docPathToRoute, getGenericSectionRoutes, listSitemapRoutes } from "../lib/content/routes";
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

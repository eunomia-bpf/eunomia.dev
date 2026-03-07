import assert from "node:assert/strict";
import test from "node:test";

import { serveRawAsset } from "../lib/content/assets";
import { parseMarkdown } from "../lib/content/markdown";
import { docPathToRoute, listSitemapRoutes } from "../lib/content/routes";
import { rewriteContentUrl } from "../lib/content/rewrite";
import { resolveLocalizedSource } from "../lib/content/source";

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

test("listSitemapRoutes keeps key legacy and section routes", () => {
  const routes = new Set(listSitemapRoutes());
  assert.ok(routes.has("/zh/blogs/bpftime/"));
  assert.ok(routes.has("/eunomia-bpf/setup/build/"));
  assert.ok(routes.has("/tutorials/38-btf-uprobe/test-verify/"));
});

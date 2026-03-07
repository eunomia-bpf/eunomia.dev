import { load } from "cheerio";

import { baseUrl, seoPages } from "./config.mjs";
import {
  extractMeta,
  fetchText,
  pageUrl,
  printFailures
} from "./lib/site.mjs";

const failures = [];

function check(condition, message) {
  if (!condition) {
    failures.push(message);
    console.error(`FAIL ${message}`);
    return;
  }
  console.log(`PASS ${message}`);
}

async function auditRobots() {
  const url = pageUrl("/robots.txt");
  const { response, text } = await fetchText(url);
  check(response.ok, "robots.txt is reachable");
  check(text.includes("Sitemap:"), "robots.txt advertises sitemap");
}

async function auditSitemap() {
  const url = pageUrl("/sitemap.xml");
  const { response, text } = await fetchText(url);
  check(response.ok, "sitemap.xml is reachable");
  const $ = load(text, { xmlMode: true });
  const urls = $("url > loc");
  check(urls.length > 0, "sitemap contains at least one URL");
  const homeLoc = urls
    .map((_, element) => $(element).text().trim())
    .get()
    .includes(pageUrl("/"));
  check(homeLoc, "sitemap includes home page");
}

async function auditSeoPage({ label, path, expectedLang }) {
  const url = pageUrl(path);
  const { response, text } = await fetchText(url);
  check(response.ok, `${label}: page is reachable`);

  const $ = load(text);
  const canonical = $("link[rel='canonical']").attr("href") ?? "";
  const alternates = $("link[rel='alternate'][hreflang]")
    .map((_, element) => ({
      href: $(element).attr("href") ?? "",
      hreflang: $(element).attr("hreflang") ?? ""
    }))
    .get();

  check(Boolean(($("title").text() ?? "").trim()), `${label}: title exists`);
  check(
    extractMeta($, "meta[name='description']").length > 20,
    `${label}: meta description exists`
  );
  check(canonical === url, `${label}: canonical matches public URL`);
  check(Boolean(extractMeta($, "meta[property='og:title']")), `${label}: og:title exists`);
  check(
    Boolean(extractMeta($, "meta[property='og:description']")),
    `${label}: og:description exists`
  );
  check(Boolean(extractMeta($, "meta[property='og:image']")), `${label}: og:image exists`);
  check(
    alternates.some((alternate) => alternate.hreflang === "en"),
    `${label}: hreflang en exists`
  );
  check(
    alternates.some((alternate) => alternate.hreflang === "zh"),
    `${label}: hreflang zh exists`
  );
  check(
    ($("html").attr("lang") ?? "").toLowerCase() === expectedLang,
    `${label}: html lang is ${expectedLang}`
  );

  const html = text.toLowerCase();
  check(
    html.includes("googletagmanager.com/gtag/js") || html.includes("google-analytics.com"),
    `${label}: analytics script is present`
  );
}

async function main() {
  console.log(`Auditing SEO and infrastructure for ${baseUrl.toString()}`);
  await auditRobots();
  await auditSitemap();
  for (const page of seoPages) {
    await auditSeoPage(page);
  }

  if (failures.length) {
    printFailures(failures);
    process.exitCode = 1;
    return;
  }

  console.log("\nHTTP audit passed.");
}

await main();

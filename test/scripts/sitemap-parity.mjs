import { baseUrl, legacySitemapPath } from "./config.mjs";
import {
  fetchSitemapPaths,
  printFailures,
  readLocalSitemapPaths
} from "./lib/site.mjs";

const failures = [];
const datedBlogRoutePattern = /^\/(?:zh\/)?blog\/\d{4}\/\d{2}\/\d{2}\/[^/]+\/$/;
const retiredLegacySitemapPaths = new Set([
  "/GPTtrace/agentsight/",
  "/zh/GPTtrace/agentsight/",
  "/tutorials/SUMMARY/",
  "/zh/tutorials/SUMMARY/"
]);
// Legacy /blogs/** (and /zh/blogs/**) pages are served with robots: noindex, so
// they are intentionally excluded from the app sitemap (see isSitemapExcludedRoute).
const retiredLegacyBlogPattern = /^\/(?:zh\/)?blogs(?:\/|$)/;
const expectedAppOnlyPaths = new Set([
  "/about/",
  "/products/",
  "/products/agent-runtime-infrastructure/",
  "/products/services/",
  "/projects/",
  "/zh/about/",
  "/zh/products/",
  "/zh/products/agent-runtime-infrastructure/",
  "/zh/products/services/",
  "/zh/projects/"
]);

function check(condition, message) {
  if (!condition) {
    failures.push(message);
    console.error(`FAIL ${message}`);
    return false;
  }
  return true;
}

async function main() {
  console.log(`Comparing sitemap parity for ${baseUrl.toString()}`);

  const legacySitemapPathnames = readLocalSitemapPaths(legacySitemapPath).map(
    (url) => new URL(url).pathname
  );
  const retiredPaths = new Set([
    ...retiredLegacySitemapPaths,
    ...legacySitemapPathnames.filter((pathname) => retiredLegacyBlogPattern.test(pathname))
  ]);
  const legacyPaths = new Set(
    legacySitemapPathnames.filter((pathname) => !retiredPaths.has(pathname))
  );
  const { response, paths } = await fetchSitemapPaths();
  check(response.ok, "target sitemap is reachable");
  check(new Set(paths).size === paths.length, "app sitemap does not contain duplicate URLs");

  const appPaths = new Set(paths.map((url) => new URL(url).pathname));
  const missing = [...legacyPaths].filter((pathname) => !appPaths.has(pathname)).sort();
  const extra = [...appPaths].filter((pathname) => !legacyPaths.has(pathname)).sort();
  const compatibleGrowth = extra.filter((pathname) => datedBlogRoutePattern.test(pathname));
  const expectedGrowth = extra.filter((pathname) => expectedAppOnlyPaths.has(pathname));
  const missingExpectedGrowth = [...expectedAppOnlyPaths].filter((pathname) => !appPaths.has(pathname)).sort();
  const retiredStillPublic = [...retiredPaths].filter((pathname) => appPaths.has(pathname)).sort();

  check(missing.length === 0, `all legacy sitemap paths exist in app sitemap (${missing.length} missing)`);
  check(
    missingExpectedGrowth.length === 0,
    `all expected app-only sitemap paths exist (${missingExpectedGrowth.length} missing)`
  );
  check(retiredStillPublic.length === 0, `retired legacy sitemap paths stay excluded (${retiredStillPublic.length} present)`);

  console.log(`Legacy sitemap paths: ${legacyPaths.size}`);
  console.log(`App sitemap paths: ${appPaths.size}`);
  console.log(`App-only sitemap paths: ${extra.length}`);
  console.log(`Compatible dated blog paths: ${compatibleGrowth.length}`);
  console.log(`Expected app-only paths: ${expectedGrowth.length}`);
  console.log(`Retired legacy sitemap paths: ${retiredPaths.size}`);

  if (missing.length) {
    for (const pathname of missing.slice(0, 50)) {
      console.error(`MISSING ${pathname}`);
    }
  }

  if (compatibleGrowth.length) {
    for (const pathname of compatibleGrowth.slice(0, 20)) {
      console.log(`DATED ${pathname}`);
    }
  }

  if (missingExpectedGrowth.length) {
    for (const pathname of missingExpectedGrowth.slice(0, 20)) {
      console.error(`MISSING_EXPECTED ${pathname}`);
    }
  }

  if (retiredStillPublic.length) {
    for (const pathname of retiredStillPublic.slice(0, 20)) {
      console.error(`RETIRED_PUBLIC ${pathname}`);
    }
  }

  if (failures.length) {
    printFailures(failures);
    process.exitCode = 1;
    return;
  }

  console.log("\nSitemap parity passed.");
}

await main();

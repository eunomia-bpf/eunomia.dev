import { baseUrl, legacySitemapPath } from "./config.mjs";
import {
  fetchSitemapPaths,
  printFailures,
  readLocalSitemapPaths
} from "./lib/site.mjs";

const failures = [];

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

  const legacyPaths = new Set(
    readLocalSitemapPaths(legacySitemapPath).map((url) => new URL(url).pathname)
  );
  const { response, paths } = await fetchSitemapPaths();
  check(response.ok, "target sitemap is reachable");

  const appPaths = new Set(paths.map((url) => new URL(url).pathname));
  const missing = [...legacyPaths].filter((pathname) => !appPaths.has(pathname)).sort();
  const extra = [...appPaths].filter((pathname) => !legacyPaths.has(pathname)).sort();

  check(missing.length === 0, `all legacy sitemap paths exist in app sitemap (${missing.length} missing)`);

  console.log(`Legacy sitemap paths: ${legacyPaths.size}`);
  console.log(`App sitemap paths: ${appPaths.size}`);
  console.log(`App-only sitemap paths: ${extra.length}`);

  if (missing.length) {
    for (const pathname of missing.slice(0, 50)) {
      console.error(`MISSING ${pathname}`);
    }
  }

  if (extra.length) {
    for (const pathname of extra.slice(0, 20)) {
      console.log(`EXTRA ${pathname}`);
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

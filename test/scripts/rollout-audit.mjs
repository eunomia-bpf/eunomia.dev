import { legacySitemapPath, rolloutAuditDistDir, rolloutAuditSampleBlogRoute } from "./config.mjs";
import { fetchSitemapPathsAt, printFailures, readLocalSitemapPaths } from "./lib/site.mjs";
import { getAvailablePort, runNextBuild, startNextServer, stopNextServer } from "./lib/next-app.mjs";

const failures = [];
const datedBlogRoutePattern = /^\/(?:zh\/)?blog\/\d{4}\/\d{2}\/\d{2}\/[^/]+\/$/;

function check(condition, message) {
  if (!condition) {
    failures.push(message);
    console.error(`FAIL ${message}`);
    return false;
  }
  console.log(`PASS ${message}`);
  return true;
}

async function fetchStagePaths(stage, siteUrl) {
  const port = Number.parseInt(new URL(siteUrl).port, 10);
  const server = await startNextServer({
    distDir: rolloutAuditDistDir,
    siteUrl,
    port,
    extraEnv: {
      EUNOMIA_SITEMAP_STAGE: stage
    }
  });

  try {
    const { response, paths } = await fetchSitemapPathsAt(siteUrl, "/sitemap.xml");
    check(response.ok, `${stage}: sitemap is reachable`);
    return new Set(paths.map((url) => new URL(url).pathname));
  } finally {
    await stopNextServer(server);
  }
}

async function main() {
  const port = await getAvailablePort();
  const siteUrl = `http://127.0.0.1:${port}`;
  const legacyPaths = new Set(readLocalSitemapPaths(legacySitemapPath).map((url) => new URL(url).pathname));

  console.log(`Auditing rollout stages with ${siteUrl}/`);

  const buildOutput = await runNextBuild({
    distDir: rolloutAuditDistDir,
    siteUrl
  });
  const built = check(buildOutput.exitCode === 0, "rollout audit build completes successfully");
  if (!built) {
    throw new Error("rollout audit build failed");
  }

  const shadowPaths = await fetchStagePaths("shadow", siteUrl);
  const shadowMissing = [...legacyPaths].filter((pathname) => !shadowPaths.has(pathname));
  const shadowExtra = [...shadowPaths].filter((pathname) => !legacyPaths.has(pathname));
  check(shadowMissing.length === 0, `shadow sitemap keeps all legacy routes (${shadowMissing.length} missing)`);
  check(shadowExtra.length === 0, `shadow sitemap adds no app-only routes (${shadowExtra.length} extra)`);
  check(!shadowPaths.has(rolloutAuditSampleBlogRoute), "shadow sitemap excludes dated blog cutover routes");

  const cutoverPaths = await fetchStagePaths("cutover", siteUrl);
  const cutoverMissing = [...legacyPaths].filter((pathname) => !cutoverPaths.has(pathname));
  const cutoverExtra = [...cutoverPaths].filter((pathname) => !legacyPaths.has(pathname));
  check(cutoverMissing.length === 0, `cutover sitemap keeps all legacy routes (${cutoverMissing.length} missing)`);
  check(
    cutoverExtra.every((pathname) => datedBlogRoutePattern.test(pathname)),
    `cutover sitemap only adds dated blog routes (${cutoverExtra.length} extras)`
  );
  check(cutoverPaths.has(rolloutAuditSampleBlogRoute), "cutover sitemap includes dated blog routes");

  const growthPaths = await fetchStagePaths("growth", siteUrl);
  const growthMissing = [...cutoverPaths].filter((pathname) => !growthPaths.has(pathname));
  check(growthMissing.length === 0, `growth sitemap is a superset of cutover (${growthMissing.length} missing)`);

  if (failures.length) {
    printFailures(failures);
    process.exitCode = 1;
    return;
  }

  console.log("\nRollout audit passed.");
}

await main();

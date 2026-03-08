import fs from "node:fs";
import path from "node:path";

import { appDir, requestTimeoutMs, runtimeAuditDistDir, runtimeAuditRoutes } from "./config.mjs";
import {
  extractLargePageWarnings,
  fetchStatus,
  getAvailablePort,
  runNextBuild,
  startNextServer,
  stopNextServer
} from "./lib/next-app.mjs";
import {
  formatStaticConstraintFailures,
  getStaticConstraintViolations
} from "./lib/static-constraints.mjs";

const failures = [];

function check(condition, message) {
  if (!condition) {
    failures.push(message);
    console.error(`FAIL ${message}`);
    return false;
  }
  console.log(`PASS ${message}`);
  return true;
}

function delay(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

async function main() {
  const port = await getAvailablePort();
  const runtimeBaseUrl = `http://127.0.0.1:${port}`;

  console.log(`Auditing static export delivery with ${runtimeBaseUrl}/`);

  const staticConstraintViolations = getStaticConstraintViolations(appDir);
  const staticConstraintFailures = formatStaticConstraintFailures(staticConstraintViolations);
  check(staticConstraintViolations.apiRouteFiles.length === 0, "no pages/api route files remain");
  check(staticConstraintViolations.getServerSidePropsFiles.length === 0, "no getServerSideProps usages remain");
  for (const failure of staticConstraintFailures) {
    console.error(`DETAIL ${failure}`);
  }

  const buildOutput = await runNextBuild({
    distDir: runtimeAuditDistDir,
    siteUrl: runtimeBaseUrl,
    echoOutput: true
  });
  const buildSucceeded = check(buildOutput.exitCode === 0, "runtime audit build completes successfully");
  if (!buildSucceeded) {
    throw new Error("runtime audit build failed");
  }

  const buildWarnings = extractLargePageWarnings(buildOutput.combined);
  check(buildWarnings.length === 0, "build output does not emit large page data warnings");
  check(Boolean(buildOutput.exportDir), "static export directory is recorded");
  check(fs.existsSync(path.join(buildOutput.exportDir, "index.html")), "static export writes the home page");
  check(fs.existsSync(path.join(buildOutput.exportDir, "sitemap.xml")), "static export writes sitemap.xml");

  const server = await startNextServer({
    distDir: runtimeAuditDistDir,
    siteUrl: runtimeBaseUrl,
    port,
    echoOutput: true
  });

  try {
    check(true, "static export server becomes ready");

    for (const pathname of runtimeAuditRoutes) {
      const response = await fetchStatus(new URL(pathname, runtimeBaseUrl));
      check(response.ok, `static export route reachable ${pathname}`);
    }

    const forbiddenApiRoutes = ["/api/search", "/api/og", "/api/raw-assets/docs/README.md"];
    for (const pathname of forbiddenApiRoutes) {
      const response = await fetchStatus(new URL(pathname, runtimeBaseUrl));
      check(!response.ok, `forbidden runtime endpoint stays unavailable ${pathname}`);
    }

    await delay(Math.min(requestTimeoutMs, 500));
  } finally {
    await stopNextServer(server);
  }

  if (failures.length) {
    console.error("\nFailures:");
    for (const failure of failures) {
      console.error(`- ${failure}`);
    }
    process.exitCode = 1;
    return;
  }

  console.log("\nRuntime audit passed.");
}

await main();

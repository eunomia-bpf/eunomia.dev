import { requestTimeoutMs, runtimeAuditDistDir, runtimeAuditRoutes } from "./config.mjs";
import {
  extractLargePageWarnings,
  fetchStatus,
  getAvailablePort,
  runNextBuild,
  startNextServer,
  stopNextServer
} from "./lib/next-app.mjs";

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

  console.log(`Auditing runtime warnings with ${runtimeBaseUrl}/`);

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

  const server = await startNextServer({
    distDir: runtimeAuditDistDir,
    siteUrl: runtimeBaseUrl,
    port,
    echoOutput: true
  });

  try {
    check(true, "runtime audit server becomes ready");

    for (const pathname of runtimeAuditRoutes) {
      const response = await fetchStatus(new URL(pathname, runtimeBaseUrl));
      check(response.ok, `runtime audit route reachable ${pathname}`);
    }

    await delay(Math.min(requestTimeoutMs, 500));
  } finally {
    const serverOutput = await stopNextServer(server);
    const runtimeWarnings = extractLargePageWarnings(serverOutput.combined);
    check(runtimeWarnings.length === 0, "runtime server output stays free of large page data warnings");
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

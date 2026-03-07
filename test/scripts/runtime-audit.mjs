import net from "node:net";
import path from "node:path";
import { spawn } from "node:child_process";

import { appDir, requestTimeoutMs, runtimeAuditDistDir, runtimeAuditRoutes } from "./config.mjs";

const failures = [];
const nextBin = path.join(appDir, "node_modules/next/dist/bin/next");

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

async function getAvailablePort() {
  const server = net.createServer();
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address();
  const port = typeof address === "object" && address ? address.port : 0;
  await new Promise((resolve, reject) => {
    server.close((error) => {
      if (error) {
        reject(error);
        return;
      }
      resolve();
    });
  });
  return port;
}

function spawnCommand(command, args, options = {}) {
  const child = spawn(command, args, {
    cwd: appDir,
    env: {
      ...process.env,
      ...(options.env ?? {})
    },
    stdio: ["ignore", "pipe", "pipe"]
  });

  let stdout = "";
  let stderr = "";

  child.stdout?.on("data", (chunk) => {
    const text = chunk.toString();
    stdout += text;
    if (options.echoOutput) {
      process.stdout.write(text);
    }
  });

  child.stderr?.on("data", (chunk) => {
    const text = chunk.toString();
    stderr += text;
    if (options.echoOutput) {
      process.stderr.write(text);
    }
  });

  return {
    child,
    getOutput() {
      return {
        stdout,
        stderr,
        combined: `${stdout}\n${stderr}`
      };
    }
  };
}

async function runBuild(baseUrl) {
  const build = spawnCommand(process.execPath, [nextBin, "build"], {
    env: {
      NEXT_DIST_DIR: runtimeAuditDistDir,
      NEXT_PUBLIC_SITE_URL: baseUrl
    },
    echoOutput: true
  });

  const exitCode = await new Promise((resolve, reject) => {
    build.child.once("error", reject);
    build.child.once("close", resolve);
  });

  const succeeded = check(exitCode === 0, "runtime audit build completes successfully");
  if (!succeeded) {
    throw new Error("runtime audit build failed");
  }

  return build.getOutput();
}

async function waitForServerReady(processHandle) {
  const deadline = Date.now() + requestTimeoutMs;
  while (Date.now() < deadline) {
    if (processHandle.child.exitCode !== null) {
      throw new Error("runtime audit server exited before becoming ready");
    }

    const { combined } = processHandle.getOutput();
    if (/Ready in/i.test(combined)) {
      return;
    }

    await delay(100);
  }

  throw new Error("runtime audit server did not become ready before timeout");
}

async function fetchStatus(url) {
  const response = await fetch(url, {
    redirect: "follow",
    signal: AbortSignal.timeout(requestTimeoutMs)
  });
  await response.arrayBuffer();
  return response;
}

async function waitForExit(child, timeoutMs) {
  if (child.exitCode !== null) {
    return true;
  }

  return await Promise.race([
    new Promise((resolve) => {
      child.once("close", () => resolve(true));
    }),
    delay(timeoutMs).then(() => false)
  ]);
}

async function stopProcess(processHandle) {
  if (processHandle.child.exitCode !== null) {
    return processHandle.getOutput();
  }

  processHandle.child.kill("SIGINT");
  const exitedAfterInterrupt = await waitForExit(processHandle.child, 5000);

  if (!exitedAfterInterrupt && processHandle.child.exitCode === null) {
    processHandle.child.kill("SIGKILL");
    await waitForExit(processHandle.child, 5000);
  }

  return processHandle.getOutput();
}

function extractLargePageWarnings(output) {
  return output
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => /large page data/i.test(line));
}

async function main() {
  const port = await getAvailablePort();
  const runtimeBaseUrl = `http://127.0.0.1:${port}`;

  console.log(`Auditing runtime warnings with ${runtimeBaseUrl}/`);

  const buildOutput = await runBuild(runtimeBaseUrl);
  const buildWarnings = extractLargePageWarnings(buildOutput.combined);
  check(buildWarnings.length === 0, "build output does not emit large page data warnings");

  const server = spawnCommand(process.execPath, [nextBin, "start", "--hostname", "127.0.0.1", "--port", String(port)], {
    env: {
      NEXT_DIST_DIR: runtimeAuditDistDir,
      NEXT_PUBLIC_SITE_URL: runtimeBaseUrl
    },
    echoOutput: true
  });

  try {
    await waitForServerReady(server);
    check(true, "runtime audit server becomes ready");

    for (const pathname of runtimeAuditRoutes) {
      const response = await fetchStatus(new URL(pathname, runtimeBaseUrl));
      check(response.ok, `runtime audit route reachable ${pathname}`);
    }

    await delay(500);
  } finally {
    const serverOutput = await stopProcess(server);
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

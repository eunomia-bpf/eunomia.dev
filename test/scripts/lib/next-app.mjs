import net from "node:net";
import path from "node:path";
import { spawn } from "node:child_process";

import { appDir, requestTimeoutMs } from "../config.mjs";

const nextBin = path.join(appDir, "node_modules/next/dist/bin/next");

function delay(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

export async function getAvailablePort() {
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

function spawnNextCommand(args, { env, echoOutput = false } = {}) {
  const child = spawn(process.execPath, [nextBin, ...args], {
    cwd: appDir,
    env: {
      ...process.env,
      ...(env ?? {})
    },
    stdio: ["ignore", "pipe", "pipe"]
  });

  let stdout = "";
  let stderr = "";

  child.stdout?.on("data", (chunk) => {
    const text = chunk.toString();
    stdout += text;
    if (echoOutput) {
      process.stdout.write(text);
    }
  });

  child.stderr?.on("data", (chunk) => {
    const text = chunk.toString();
    stderr += text;
    if (echoOutput) {
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

async function waitForProcessExit(child, timeoutMs) {
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

async function waitForServerReady(processHandle) {
  const deadline = Date.now() + requestTimeoutMs;
  while (Date.now() < deadline) {
    if (processHandle.child.exitCode !== null) {
      throw new Error("Next server exited before becoming ready");
    }

    const { combined } = processHandle.getOutput();
    if (/Ready in/i.test(combined)) {
      return;
    }

    await delay(100);
  }

  throw new Error("Next server did not become ready before timeout");
}

export async function runNextBuild({ distDir, siteUrl, extraEnv = {}, echoOutput = false }) {
  const build = spawnNextCommand(["build"], {
    env: {
      NEXT_DIST_DIR: distDir,
      NEXT_PUBLIC_SITE_URL: siteUrl,
      ...extraEnv
    },
    echoOutput
  });

  const exitCode = await new Promise((resolve, reject) => {
    build.child.once("error", reject);
    build.child.once("close", resolve);
  });

  return {
    exitCode,
    ...build.getOutput()
  };
}

export async function startNextServer({
  distDir,
  siteUrl,
  port,
  extraEnv = {},
  echoOutput = false
}) {
  const server = spawnNextCommand(["start", "--hostname", "127.0.0.1", "--port", String(port)], {
    env: {
      NEXT_DIST_DIR: distDir,
      NEXT_PUBLIC_SITE_URL: siteUrl,
      ...extraEnv
    },
    echoOutput
  });

  await waitForServerReady(server);
  return server;
}

export async function stopNextServer(server) {
  if (server.child.exitCode !== null) {
    return server.getOutput();
  }

  server.child.kill("SIGINT");
  const exitedAfterInterrupt = await waitForProcessExit(server.child, 5000);

  if (!exitedAfterInterrupt && server.child.exitCode === null) {
    server.child.kill("SIGKILL");
    await waitForProcessExit(server.child, 5000);
  }

  return server.getOutput();
}

export function extractLargePageWarnings(output) {
  return output
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => /large page data/i.test(line));
}

export async function fetchStatus(url) {
  const response = await fetch(url, {
    redirect: "follow",
    signal: AbortSignal.timeout(requestTimeoutMs)
  });
  await response.arrayBuffer();
  return response;
}

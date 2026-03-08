import fs from "node:fs";
import net from "node:net";
import path from "node:path";
import { spawn } from "node:child_process";

import { appDir, requestTimeoutMs } from "../config.mjs";
import { startStaticServer, stopStaticServer } from "./static-server.mjs";

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

function spawnAppScript(scriptName, { env, echoOutput = false } = {}) {
  const fallbackNpmCommand = process.platform === "win32" ? "npm.cmd" : "npm";

  const child = spawn(fallbackNpmCommand, ["run", scriptName], {
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

export async function runNextBuild({ distDir, siteUrl, extraEnv = {}, echoOutput = false }) {
  fs.rmSync(path.join(appDir, distDir), { recursive: true, force: true });
  fs.rmSync(path.join(appDir, "out"), { recursive: true, force: true });

  const build = spawnAppScript("build", {
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

  const output = build.getOutput();
  const exportDir = path.join(appDir, "out");

  return {
    exitCode,
    exportDir,
    ...output
  };
}

export async function startNextServer({
  distDir,
  siteUrl,
  port,
  extraEnv = {},
  echoOutput = false
}) {
  const exportDir = path.join(appDir, "out");
  if (!fs.existsSync(exportDir)) {
    const buildOutput = await runNextBuild({
      distDir,
      siteUrl,
      extraEnv,
      echoOutput
    });
    if (buildOutput.exitCode !== 0) {
      throw new Error(`Static export build failed with code ${buildOutput.exitCode}`);
    }
  }

  return await startStaticServer({
    rootDir: exportDir,
    port,
    echoOutput
  });
}

export async function stopNextServer(server) {
  if (server?.kind === "static-server") {
    return await stopStaticServer(server);
  }

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

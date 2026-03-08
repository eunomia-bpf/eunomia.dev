import fs from "node:fs";
import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { getAvailablePort, runNextBuild, startNextServer, stopNextServer } from "../../test/scripts/lib/next-app.mjs";
import { assertStaticConstraintViolations } from "../../test/scripts/lib/static-constraints.mjs";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const appDir = path.resolve(scriptDir, "..");
const testDir = path.resolve(appDir, "..", "test");

function quoteShellArg(value) {
  if (process.platform === "win32") {
    return `"${String(value).replace(/(["^%])/g, "^$1")}"`;
  }

  return `'${String(value).replace(/'/g, `'\\''`)}'`;
}

function getShellLauncher(cmd, args) {
  const commandLine = [cmd, ...args].map(quoteShellArg).join(" ");

  if (process.platform === "win32") {
    return {
      cmd: process.env.ComSpec ?? "cmd.exe",
      args: ["/d", "/s", "/c", commandLine]
    };
  }

  return {
    cmd: "/bin/sh",
    args: ["-lc", commandLine]
  };
}

function runCommand(cmd, args, options = {}) {
  const pathValue =
    options.env?.PATH ??
    options.env?.Path ??
    process.env.PATH ??
    process.env.Path ??
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin";
  const launcher = getShellLauncher(cmd, args);

  return new Promise((resolve, reject) => {
    const child = spawn(launcher.cmd, launcher.args, {
      cwd: options.cwd ?? appDir,
      env: {
        ...process.env,
        ...(options.env ?? {}),
        PATH: pathValue
      },
      stdio: "inherit"
    });

    child.once("error", reject);
    child.once("close", (code) => {
      if (code === 0) {
        resolve();
        return;
      }

      reject(new Error(`${cmd} ${args.join(" ")} exited with code ${code}`));
    });
  });
}

async function ensureNodeModules(workingDir) {
  if (fs.existsSync(path.join(workingDir, "node_modules"))) {
    return;
  }

  await runCommand("npm", ["ci"], { cwd: workingDir });
}

async function main() {
  const port = await getAvailablePort();
  const baseUrl = `http://127.0.0.1:${port}`;
  const distDir = process.env.NEXT_DIST_DIR ?? ".static-builds/export";
  const distPath = path.resolve(appDir, distDir);

  await runCommand("npm", ["run", "typecheck"]);
  await runCommand("npm", ["run", "test:content"]);
  assertStaticConstraintViolations(appDir);
  fs.rmSync(distPath, { recursive: true, force: true });
  const build = await runNextBuild({
    distDir,
    siteUrl: baseUrl,
    echoOutput: true
  });
  if (build.exitCode !== 0) {
    throw new Error(`next build exited with code ${build.exitCode}`);
  }
  assertStaticConstraintViolations(appDir);
  await runCommand("node", ["scripts/assert-content-artifacts.mjs"], { cwd: appDir });

  const server = await startNextServer({
    distDir,
    siteUrl: baseUrl,
    port,
    echoOutput: true
  });

  try {
    await ensureNodeModules(testDir);
    await runCommand("npm", ["run", "audit:http"], { cwd: testDir, env: { BASE_URL: baseUrl } });
    await runCommand("npm", ["run", "audit:parity"], { cwd: testDir, env: { BASE_URL: baseUrl } });
    await runCommand("npm", ["run", "audit:browser"], { cwd: testDir, env: { BASE_URL: baseUrl } });
    await runCommand("npm", ["run", "audit:links"], { cwd: testDir, env: { BASE_URL: baseUrl } });
  } finally {
    await stopNextServer(server);
  }

  await runCommand("npm", ["run", "audit:runtime"], {
    cwd: testDir,
    env: {
      APP_DIR: appDir,
      RUNTIME_AUDIT_SKIP_BUILD: "1"
    }
  });
}

await main();

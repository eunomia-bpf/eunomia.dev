import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";

const MIME_TYPES = new Map([
  [".css", "text/css; charset=utf-8"],
  [".gif", "image/gif"],
  [".html", "text/html; charset=utf-8"],
  [".ico", "image/x-icon"],
  [".jpeg", "image/jpeg"],
  [".jpg", "image/jpeg"],
  [".js", "application/javascript; charset=utf-8"],
  [".json", "application/json; charset=utf-8"],
  [".map", "application/json; charset=utf-8"],
  [".png", "image/png"],
  [".rss", "application/rss+xml; charset=utf-8"],
  [".svg", "image/svg+xml; charset=utf-8"],
  [".txt", "text/plain; charset=utf-8"],
  [".webp", "image/webp"],
  [".woff", "font/woff"],
  [".woff2", "font/woff2"],
  [".xml", "application/xml; charset=utf-8"]
]);

function logBuffer(echoOutput) {
  let stdout = "";
  let stderr = "";

  function write(stream, text) {
    if (stream === "stdout") {
      stdout += text;
      if (echoOutput) {
        process.stdout.write(text);
      }
      return;
    }

    stderr += text;
    if (echoOutput) {
      process.stderr.write(text);
    }
  }

  return {
    write,
    getOutput() {
      return {
        stdout,
        stderr,
        combined: `${stdout}\n${stderr}`
      };
    }
  };
}

function safeJoin(rootDir, requestPath) {
  const pathname = decodeURIComponent(requestPath);
  const relativePath = pathname.replace(/^\/+/, "");
  const resolvedPath = path.resolve(rootDir, relativePath);

  if (resolvedPath !== rootDir && !resolvedPath.startsWith(`${rootDir}${path.sep}`)) {
    return null;
  }

  return resolvedPath;
}

function resolveStaticFile(rootDir, requestPath) {
  const normalizedPath = requestPath.split("?")[0];
  const resolvedBase = safeJoin(rootDir, normalizedPath);
  if (!resolvedBase) {
    return null;
  }

  const hasExtension = path.extname(resolvedBase).length > 0;
  const candidates = normalizedPath.endsWith("/")
    ? [path.join(resolvedBase, "index.html")]
    : hasExtension
      ? [resolvedBase]
      : [resolvedBase, `${resolvedBase}.html`, path.join(resolvedBase, "index.html")];

  for (const candidate of candidates) {
    if (!fs.existsSync(candidate)) {
      continue;
    }

    const stats = fs.statSync(candidate);
    if (stats.isFile()) {
      return candidate;
    }
  }

  return null;
}

function contentTypeFor(filePath) {
  return MIME_TYPES.get(path.extname(filePath).toLowerCase()) ?? "application/octet-stream";
}

export async function startStaticServer({
  rootDir,
  port,
  host = "127.0.0.1",
  echoOutput = false
}) {
  const buffer = logBuffer(echoOutput);
  const absoluteRoot = path.resolve(rootDir);

  const server = http.createServer((request, response) => {
    const requestPath = new URL(request.url ?? "/", `http://${host}:${port}`).pathname;
    const targetFile = resolveStaticFile(absoluteRoot, requestPath);

    if (!targetFile) {
      const notFoundFile = resolveStaticFile(absoluteRoot, "/404.html");
      if (notFoundFile) {
        const body = fs.readFileSync(notFoundFile);
        response.writeHead(404, {
          "content-length": body.byteLength,
          "content-type": contentTypeFor(notFoundFile)
        });
        if (request.method !== "HEAD") {
          response.end(body);
          return;
        }
        response.end();
        return;
      }

      response.writeHead(404, { "content-type": "text/plain; charset=utf-8" });
      response.end("Not Found");
      return;
    }

    const body = fs.readFileSync(targetFile);
    response.writeHead(200, {
      "content-length": body.byteLength,
      "content-type": contentTypeFor(targetFile)
    });

    if (request.method !== "HEAD") {
      response.end(body);
      return;
    }

    response.end();
  });

  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(port, host, resolve);
  });

  buffer.write("stdout", `Static export ready on http://${host}:${port}/ serving ${absoluteRoot}\n`);

  return {
    kind: "static-server",
    rootDir: absoluteRoot,
    server,
    getOutput: buffer.getOutput
  };
}

export async function stopStaticServer(handle) {
  if (!handle?.server) {
    return handle?.getOutput?.() ?? { stdout: "", stderr: "", combined: "" };
  }

  if (!handle.server.listening) {
    return handle.getOutput();
  }

  await new Promise((resolve, reject) => {
    handle.server.close((error) => {
      if (error) {
        reject(error);
        return;
      }
      resolve();
    });
  });

  return handle.getOutput();
}

async function main() {
  const configuredRoot = process.env.STATIC_ROOT ?? "out";
  const rootDir = path.resolve(process.cwd(), configuredRoot);
  const port = Number.parseInt(process.env.PORT ?? "3000", 10);
  const host = process.env.HOST ?? "127.0.0.1";

  const handle = await startStaticServer({ rootDir, port, host, echoOutput: true });
  const shutdown = async () => {
    await stopStaticServer(handle);
    process.exit(0);
  };

  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}

const isEntrypoint = process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isEntrypoint) {
  await main();
}

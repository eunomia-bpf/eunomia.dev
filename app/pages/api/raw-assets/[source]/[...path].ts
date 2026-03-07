import fs from "node:fs";

import type { NextApiRequest, NextApiResponse } from "next";

import { serveRawAsset } from "../../../../lib/content";

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  const source = req.query.source;
  const pathSegments = req.query.path;

  if ((source !== "docs" && source !== "site") || !Array.isArray(pathSegments)) {
    res.status(404).end("Not Found");
    return;
  }

  const asset = await serveRawAsset(source, pathSegments);
  if (!asset) {
    res.status(404).end("Not Found");
    return;
  }

  res.setHeader("Content-Type", asset.contentType);
  res.setHeader("Cache-Control", "public, max-age=3600");

  try {
    res.status(200).send(fs.readFileSync(asset.filePath));
  } catch {
    res.status(404).end("Not Found");
  }
}

import fs from "node:fs";
import path from "node:path";

import { useContentCache } from "./cache";
import {
  discoverBlogEntries,
  discoverGenericSectionRouteEntries,
  discoverLegacyBlogEntries,
  discoverTutorialDocSources,
  discoverTutorialReadmeSources,
} from "./discovery";
import { generatedContentDir } from "./roots";
import type { BlogEntry, GenericSectionRouteEntry, LegacyBlogEntry } from "./types";

export type ContentModelArtifact = {
  generatedAt: string;
  tutorialReadmeSources: string[];
  tutorialDocSources: string[];
  blogEntries: BlogEntry[];
  legacyBlogEntries: LegacyBlogEntry[];
  genericSectionRoutes: GenericSectionRouteEntry[];
};

export type { GenericSectionRouteEntry } from "./types";

type SerializedContentModel = ContentModelArtifact;

const generatedContentModelPath = path.join(generatedContentDir, "content-model.json");

let contentModelCache: ContentModelArtifact | null = null;

function allowContentModelFallback(): boolean {
  return process.env.NODE_ENV === "development";
}

function buildContentModelFromSource(): ContentModelArtifact {
  const blogEntries = discoverBlogEntries() as BlogEntry[];
  const legacyBlogEntries = discoverLegacyBlogEntries() as LegacyBlogEntry[];

  return {
    generatedAt: new Date().toISOString(),
    tutorialReadmeSources: discoverTutorialReadmeSources(),
    tutorialDocSources: discoverTutorialDocSources(),
    blogEntries,
    legacyBlogEntries,
    genericSectionRoutes: discoverGenericSectionRouteEntries()
  };
}

function hydrateContentModel(model: ContentModelArtifact) {
  contentModelCache = model;
}

function readPrebuiltContentModel(filePath: string = generatedContentModelPath): ContentModelArtifact | null {
  if (!fs.existsSync(filePath)) {
    return null;
  }

  try {
    const payload = JSON.parse(fs.readFileSync(filePath, "utf8")) as SerializedContentModel;
    if (
      !Array.isArray(payload.tutorialReadmeSources) ||
      !Array.isArray(payload.tutorialDocSources) ||
      !Array.isArray(payload.blogEntries) ||
      !Array.isArray(payload.legacyBlogEntries) ||
      !Array.isArray(payload.genericSectionRoutes)
    ) {
      return null;
    }

    return payload;
  } catch (error) {
    if (!allowContentModelFallback() && filePath === generatedContentModelPath) {
      throw new Error(`Failed to read prebuilt content model: ${String(error)}`);
    }

    console.warn("Failed to read prebuilt content model. Falling back to source scan.", error);
    return null;
  }
}

export function getContentModel(
  options: {
    allowFallback?: boolean;
    outputPath?: string;
  } = {}
): ContentModelArtifact {
  const outputPath = options.outputPath ?? generatedContentModelPath;
  const fallbackAllowed = options.allowFallback ?? allowContentModelFallback();

  if (useContentCache && outputPath === generatedContentModelPath && contentModelCache) {
    return contentModelCache;
  }

  const prebuilt = readPrebuiltContentModel(outputPath);
  if (prebuilt) {
    if (useContentCache && outputPath === generatedContentModelPath) {
      hydrateContentModel(prebuilt);
    }
    return prebuilt;
  }

  if (!fallbackAllowed) {
    throw new Error(`Missing prebuilt content model at ${outputPath}. Run generate:content-index first.`);
  }

  const rebuilt = buildContentModelFromSource();
  if (useContentCache && outputPath === generatedContentModelPath) {
    hydrateContentModel(rebuilt);
  }
  return rebuilt;
}

export function writeContentModel(outputPath: string = generatedContentModelPath) {
  const model = buildContentModelFromSource();

  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  const tempPath = `${outputPath}.tmp`;
  fs.writeFileSync(tempPath, `${JSON.stringify(model)}\n`, "utf8");
  fs.renameSync(tempPath, outputPath);

  if (useContentCache && outputPath === generatedContentModelPath) {
    hydrateContentModel(model);
  }

  return {
    count:
      model.tutorialDocSources.length +
      model.blogEntries.length +
      model.legacyBlogEntries.length +
      model.genericSectionRoutes.length,
    filePath: outputPath
  };
}

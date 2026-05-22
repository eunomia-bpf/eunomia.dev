import { writeGeneratedSiteConfig } from "../lib/content/mkdocs-config";

const result = writeGeneratedSiteConfig();

console.log(`Generated site config from mkdocs metadata at ${result.filePath}`);

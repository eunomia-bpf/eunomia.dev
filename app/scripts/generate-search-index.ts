import { writeSearchIndexes } from "../lib/content/search";

const results = writeSearchIndexes();

for (const result of results) {
  console.log(`Generated ${result.locale} search index with ${result.count} documents at ${result.filePath}`);
}

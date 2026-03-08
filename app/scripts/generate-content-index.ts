import { writeDocumentIndex } from "../lib/content/documents";

const result = writeDocumentIndex();

console.log(`Generated document index with ${result.count} documents at ${result.filePath}`);

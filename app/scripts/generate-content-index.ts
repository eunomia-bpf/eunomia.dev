import { writeDocumentIndex } from "../lib/content/documents";
import { writeSiteSections } from "../lib/site-ia-source";

const result = writeDocumentIndex();
const sectionResult = writeSiteSections();

console.log(`Generated document index with ${result.count} documents at ${result.filePath}`);
console.log(`Generated site section index with ${sectionResult.count} sections at ${sectionResult.filePath}`);

import { writeContentManifest } from "../lib/content/manifest";
import { writeContentModel } from "../lib/content/model";
import { writeDocumentIndex } from "../lib/content/documents";
import { writeSiteSections } from "../lib/site-ia-source";

const result = writeDocumentIndex();
const contentModelResult = writeContentModel();
const manifestResult = writeContentManifest();
const sectionResult = writeSiteSections();

console.log(`Generated document index with ${result.count} documents at ${result.filePath}`);
console.log(`Generated content model with ${contentModelResult.count} records at ${contentModelResult.filePath}`);
console.log(`Generated content manifest with ${manifestResult.count} routes at ${manifestResult.filePath}`);
console.log(`Generated site section index with ${sectionResult.count} sections at ${sectionResult.filePath}`);

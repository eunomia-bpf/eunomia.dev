import { writeStaticAssets } from "../lib/content/assets";

const result = writeStaticAssets();

console.log(`Generated ${result.count} static content assets at ${result.outputRoot}`);
console.log(`Wrote static asset index at ${result.indexPath}`);

import { createLintConfig } from "./eslint.shared.mjs";

export default createLintConfig({
  ignores: [
    "node_modules/**",
    ".next/**",
    ".next-*/**",
    ".static-builds/**",
    ".generated/**",
    "out/**",
    "public/_content-assets/**",
    "public/search-index/**",
    "public/og/**",
    "next-env.d.ts"
  ]
});

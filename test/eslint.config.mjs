import { createLintConfig } from "../app/eslint.shared.mjs";

export default createLintConfig({
  ignores: [
    "node_modules/**",
    ".tmp*/**"
  ],
  files: ["scripts/**/*.{js,mjs,ts}", "*.mjs"],
  includeReactHooks: false,
  includeNext: false
});

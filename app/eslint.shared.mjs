import js from "@eslint/js";
import nextPlugin from "@next/eslint-plugin-next";
import globals from "globals";
import tseslint from "typescript-eslint";
import reactHooks from "eslint-plugin-react-hooks";

export function createLintConfig({
  ignores = [],
  files = ["**/*.{js,mjs,ts,tsx}"],
  includeReactHooks = true,
  includeNext = true
} = {}) {
  return tseslint.config(
    {
      ignores
    },
    js.configs.recommended,
    ...tseslint.configs.recommended,
    {
      files,
      languageOptions: {
        ecmaVersion: "latest",
        sourceType: "module",
        parserOptions: {
          ecmaFeatures: {
            jsx: true
          }
        },
        globals: {
          ...globals.node,
          ...globals.browser
        }
      },
      plugins: {
        ...(includeNext ? { "@next/next": nextPlugin } : {}),
        ...(includeReactHooks ? { "react-hooks": reactHooks } : {})
      },
      rules: {
        ...(includeNext ? nextPlugin.configs.recommended.rules : {}),
        ...(includeReactHooks ? reactHooks.configs.recommended.rules : {}),
        "no-console": "off",
        "@typescript-eslint/no-explicit-any": "off"
      }
    }
  );
}

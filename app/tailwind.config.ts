import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}"
  ],
  theme: {
    extend: {
      colors: {
        ink: "#091627",
        mist: "#e5efff",
        accent: "#ff9f1c",
        signal: "#00b894",
        azure: "#2f66ff"
      },
      boxShadow: {
        panel: "0 18px 60px rgba(9, 22, 39, 0.12)"
      }
    }
  },
  plugins: []
};

export default config;

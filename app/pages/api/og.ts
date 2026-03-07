import type { NextApiRequest, NextApiResponse } from "next";

function escapeHtml(value: string) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function wrapText(value: string, limit: number) {
  const words = value.trim().split(/\s+/);
  const lines: string[] = [];
  let current = "";

  for (const word of words) {
    const candidate = current ? `${current} ${word}` : word;
    if (candidate.length <= limit) {
      current = candidate;
      continue;
    }

    if (current) {
      lines.push(current);
    }
    current = word;
  }

  if (current) {
    lines.push(current);
  }

  return lines.slice(0, 3);
}

export default function handler(req: NextApiRequest, res: NextApiResponse<string>) {
  const title = String(req.query.title ?? "eunomia");
  const eyebrow = String(req.query.eyebrow ?? "eunomia.dev");
  const lines = wrapText(title, 28);

  const svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg width="1200" height="630" viewBox="0 0 1200 630" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="bg" x1="98" y1="72" x2="1086" y2="574" gradientUnits="userSpaceOnUse">
      <stop stop-color="#081A33"/>
      <stop offset="0.52" stop-color="#0F305D"/>
      <stop offset="1" stop-color="#18507F"/>
    </linearGradient>
    <radialGradient id="glowA" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(980 120) rotate(120) scale(380 420)">
      <stop stop-color="#F59E0B" stop-opacity="0.45"/>
      <stop offset="1" stop-color="#F59E0B" stop-opacity="0"/>
    </radialGradient>
    <radialGradient id="glowB" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(220 520) rotate(12) scale(360 260)">
      <stop stop-color="#38BDF8" stop-opacity="0.34"/>
      <stop offset="1" stop-color="#38BDF8" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <rect width="1200" height="630" rx="40" fill="url(#bg)"/>
  <rect x="32" y="32" width="1136" height="566" rx="30" stroke="rgba(255,255,255,0.16)" fill="rgba(255,255,255,0.03)"/>
  <ellipse cx="980" cy="120" rx="320" ry="260" fill="url(#glowA)"/>
  <ellipse cx="220" cy="520" rx="280" ry="180" fill="url(#glowB)"/>
  <rect x="88" y="86" width="180" height="44" rx="22" fill="rgba(255,255,255,0.12)"/>
  <text x="118" y="114" fill="#E0F2FE" font-size="22" font-family="ui-sans-serif, system-ui, sans-serif" letter-spacing="3">${escapeHtml(
    eyebrow.toUpperCase()
  )}</text>
  <text x="88" y="196" fill="#FFFFFF" font-size="64" font-weight="700" font-family="ui-serif, Georgia, serif">
    ${lines
      .map(
        (line, index) =>
          `<tspan x="88" y="${196 + index * 76}">${escapeHtml(line)}</tspan>`
      )
      .join("")}
  </text>
  <text x="88" y="520" fill="#C7D2FE" font-size="28" font-family="ui-sans-serif, system-ui, sans-serif">eunomia.dev</text>
  <text x="88" y="560" fill="#E2E8F0" font-size="24" font-family="ui-sans-serif, system-ui, sans-serif">Open-source eBPF tools, tutorials, and systems research</text>
</svg>`;

  res.setHeader("Content-Type", "image/svg+xml");
  res.setHeader("Cache-Control", "public, max-age=0, s-maxage=86400, stale-while-revalidate=604800");
  res.status(200).send(svg);
}

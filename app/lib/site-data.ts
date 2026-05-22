import { generatedSiteConfig } from "./site-config.generated";

export type Locale = "en" | "zh";

const siteUrl = process.env.NEXT_PUBLIC_SITE_URL ?? generatedSiteConfig.siteUrl;

export const siteConfig = {
  name: generatedSiteConfig.name,
  description: generatedSiteConfig.description,
  siteUrl,
  analyticsId: "G-1YVMXGL0MY",
  repoUrl: generatedSiteConfig.repoUrl,
  copyright: generatedSiteConfig.copyright,
  remoteBranch: generatedSiteConfig.remoteBranch,
  editUri: generatedSiteConfig.editUri
};

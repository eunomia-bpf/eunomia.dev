export type RolloutStage = "shadow" | "cutover" | "growth";
export type RouteClass =
  | "home"
  | "tutorial"
  | "blog-index"
  | "blog"
  | "legacy-blog"
  | "section";

type RouteRolloutPolicy = {
  routeClass: RouteClass;
  sitemapStage: RolloutStage;
};

const stageRank: Record<RolloutStage, number> = {
  shadow: 0,
  cutover: 1,
  growth: 2
};

export const routeRolloutPolicies: Record<RouteClass, RouteRolloutPolicy> = {
  home: {
    routeClass: "home",
    sitemapStage: "shadow"
  },
  tutorial: {
    routeClass: "tutorial",
    sitemapStage: "shadow"
  },
  "blog-index": {
    routeClass: "blog-index",
    sitemapStage: "shadow"
  },
  blog: {
    routeClass: "blog",
    sitemapStage: "cutover"
  },
  "legacy-blog": {
    routeClass: "legacy-blog",
    sitemapStage: "shadow"
  },
  section: {
    routeClass: "section",
    sitemapStage: "shadow"
  }
};

export function normalizeRolloutStage(value: string | undefined): RolloutStage {
  if (value === "shadow" || value === "cutover" || value === "growth") {
    return value;
  }

  return "cutover";
}

export function getActiveRolloutStage(): RolloutStage {
  return normalizeRolloutStage(process.env.EUNOMIA_SITEMAP_STAGE ?? process.env.EUNOMIA_ROLLOUT_STAGE);
}

export function stageAllowsRoute(routeStage: RolloutStage, activeStage: RolloutStage): boolean {
  return stageRank[routeStage] <= stageRank[activeStage];
}

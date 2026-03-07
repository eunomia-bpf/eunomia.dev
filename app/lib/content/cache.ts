export const useContentCache =
  process.env.NODE_ENV === "production" && process.env.EUNOMIA_DISABLE_CONTENT_CACHE !== "1";

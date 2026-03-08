import { createContentPage } from "../lib/page-builders";
import { createContentPageRoute } from "../lib/route-builders";

const contentPageRoute = createContentPageRoute("en");
const contentPage = createContentPage("en");

export const getStaticPaths = contentPageRoute.getStaticPaths;
export const getStaticProps = contentPageRoute.getStaticProps;

export default contentPage;

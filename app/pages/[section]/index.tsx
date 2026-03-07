import { createSectionPage } from "../../lib/page-builders";
import { createSectionPageRoute } from "../../lib/route-builders";

const sectionPageRoute = createSectionPageRoute("en");
const sectionPage = createSectionPage("en");

export const getStaticPaths = sectionPageRoute.getStaticPaths;
export const getStaticProps = sectionPageRoute.getStaticProps;

export default sectionPage;

import { createSectionPage } from "../../lib/page-builders";
import { createSectionArticleRoute } from "../../lib/route-builders";

const sectionPageRoute = createSectionArticleRoute("en");
const sectionPage = createSectionPage("en");

export const getServerSideProps = sectionPageRoute.getServerSideProps;

export default sectionPage;

import { createSearchPage } from "../lib/page-builders";
import { createSearchPageRoute } from "../lib/route-builders";

const searchPageRoute = createSearchPageRoute("en");
const searchPage = createSearchPage("en");

export const getStaticProps = searchPageRoute.getStaticProps;

export default searchPage;

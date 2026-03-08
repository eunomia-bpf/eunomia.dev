import { createSearchPage } from "../../lib/page-builders";
import { createSearchPageRoute } from "../../lib/route-builders";

const searchPageRoute = createSearchPageRoute("zh");
const searchPage = createSearchPage("zh");

export const getStaticProps = searchPageRoute.getStaticProps;

export default searchPage;

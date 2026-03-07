import { createHomePage } from "../../lib/page-builders";
import { createHomePageRoute } from "../../lib/route-builders";

const homePageRoute = createHomePageRoute("zh");
const homePage = createHomePage("zh", "主页");

export const getStaticProps = homePageRoute.getStaticProps;

export default homePage;

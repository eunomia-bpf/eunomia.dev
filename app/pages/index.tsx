import { createHomePage } from "../lib/page-builders";
import { createHomePageRoute } from "../lib/route-builders";

const homePageRoute = createHomePageRoute("en");
const homePage = createHomePage("en", "Home");

export const getStaticProps = homePageRoute.getStaticProps;

export default homePage;

import { createCollectionPage } from "../../../lib/page-builders";
import { createTutorialPageRoute } from "../../../lib/route-builders";
import { loadTutorialIndex, loadTutorialPage } from "../../../lib/content";

const tutorialsPageRoute = createTutorialPageRoute("zh");
const tutorialsPage = createCollectionPage<
  Awaited<ReturnType<typeof loadTutorialIndex>>,
  NonNullable<Awaited<ReturnType<typeof loadTutorialPage>>>
>("zh", "教程");

export const getStaticProps = tutorialsPageRoute.getStaticProps;

export default tutorialsPage;

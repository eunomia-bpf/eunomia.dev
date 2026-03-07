import Document, {
  Head,
  Html,
  Main,
  NextScript,
  type DocumentContext,
  type DocumentInitialProps
} from "next/document";

import { siteConfig } from "../lib/site-data";

type Props = DocumentInitialProps & {
  localeLang: "en" | "zh";
};

export default class MyDocument extends Document<Props> {
  static async getInitialProps(ctx: DocumentContext): Promise<Props> {
    const initialProps = await Document.getInitialProps(ctx);
    const requestPath = ctx.req?.url ?? ctx.pathname ?? "/";
    const localeLang = requestPath.startsWith("/zh") ? "zh" : "en";
    return {
      ...initialProps,
      localeLang
    };
  }

  render() {
    const analyticsId = JSON.stringify(siteConfig.analyticsId);

    return (
      <Html lang={this.props.localeLang}>
        <Head>
          <script
            async
            src={`https://www.googletagmanager.com/gtag/js?id=${encodeURIComponent(siteConfig.analyticsId)}`}
          />
          <script
            dangerouslySetInnerHTML={{
              __html: `window.dataLayer = window.dataLayer || [];
function gtag(){dataLayer.push(arguments);}
gtag('js', new Date());
gtag('config', ${analyticsId});`
            }}
          />
        </Head>
        <body>
          <Main />
          <NextScript />
        </body>
      </Html>
    );
  }
}

import type { ReactNode } from "react";
import { COURSE, SERIES } from "./brand";
import { Monogram } from "./Monogram";

/**
 * The top of every course in the series: the accent rule (drawn by brand.css
 * on body), the series row, the course title, its tagline, and whatever
 * navigation the course passes in.
 *
 * The title is split across two lines on purpose. The wordmark carries the
 * series half of the name and the heading carries the subject, so
 * "Grokking / Nets" reads down the page and a sibling course reads
 * "Grokking / Ciphers" with nothing else moved. A screen reader is given the
 * whole name in the heading, because a heading that announces "Nets" alone
 * would leave a listener without the series it belongs to.
 */
export function Masthead({ nav }: { nav?: ReactNode }) {
  const wordmark = (
    <>
      <Monogram />
      <span className="brand-wordmark">{SERIES.name}</span>
    </>
  );
  return (
    <header className="masthead">
      <p className="brand-row">
        {SERIES.homeUrl ? (
          <a className="brand-mark" href={SERIES.homeUrl}>
            {wordmark}
          </a>
        ) : (
          <span className="brand-mark">{wordmark}</span>
        )}
        <span className="brand-series-note">{SERIES.note}</span>
      </p>
      <h1 className="masthead-title">
        <span className="sr-only">{SERIES.name} </span>
        {COURSE.subject}
      </h1>
      <p className="masthead-tagline">{COURSE.tagline}</p>
      {nav}
    </header>
  );
}

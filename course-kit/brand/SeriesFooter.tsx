import type { ReactNode } from "react";
import { COURSE, SERIES } from "./brand";
import { Monogram } from "./Monogram";

/**
 * The bottom of every course in the series: what the series is, links to its
 * other courses, and then the course's own legal text.
 *
 * The legal text arrives as children rather than from brand.ts, because every
 * course carries different obligations and a shared component that tried to
 * hold them would end up either wrong or empty. This one is only responsible
 * for putting them below a rule, where a reader looking for the licence knows
 * to find them.
 */
export function SeriesFooter({ children }: { children: ReactNode }) {
  const siblings = SERIES.courses.filter((c) => c.id !== COURSE.id && c.url);
  return (
    <footer className="series-footer">
      <div className="series-band">
        <Monogram />
        <span className="series-what">
          {SERIES.homeUrl ? <a href={SERIES.homeUrl}>{SERIES.name}</a> : <b>{SERIES.name}</b>}
          {". "}
          {SERIES.what}
        </span>
        {siblings.length > 0 && (
          <span className="series-siblings">
            Also in this series:{" "}
            {siblings.map((c, i) => (
              <span key={c.id}>
                {i > 0 && ", "}
                <a href={c.url!}>
                  {SERIES.name} {c.subject}
                </a>
              </span>
            ))}
          </span>
        )}
      </div>
      <div className="series-legal">{children}</div>
    </footer>
  );
}

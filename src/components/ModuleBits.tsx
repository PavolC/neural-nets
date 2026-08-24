import type { ReactNode } from "react";

// Shared building blocks for module pages (conventions in CLAUDE.md):
// opener with "What you'll be able to do after this", figures with
// captions, and a closing recap with a "Go deeper" link to Nielsen.

export function AfterThis({ items }: { items: string[] }) {
  return (
    <div className="after-this">
      <h3>What you'll be able to do after this</h3>
      <ul>
        {items.map((item, i) => (
          <li key={i}>{item}</li>
        ))}
      </ul>
    </div>
  );
}

/** A quick departure from the lesson (an analogy, a scope note, a why-
 * digression), visually set off so the reader knows the main thread pauses
 * here and resumes after the box. */
export function Aside({ children }: { children: ReactNode }) {
  return (
    <aside className="module-aside">
      <p className="module-aside-label">An aside</p>
      {children}
    </aside>
  );
}

export function Figure({ children, caption }: { children: ReactNode; caption: string }) {
  return (
    <figure className="module-figure">
      {children}
      <figcaption>{caption}</figcaption>
    </figure>
  );
}

export function Recap({
  items,
  chapter,
  href,
}: {
  items: string[];
  chapter: string;
  href: string;
}) {
  return (
    <div className="recap">
      <h3>Recap</h3>
      <ul>
        {items.map((item, i) => (
          <li key={i}>{item}</li>
        ))}
      </ul>
      <p className="go-deeper">
        Go deeper: this module follows <a href={href}>{chapter}</a> of Nielsen's{" "}
        <em>Neural Networks and Deep Learning</em>, which covers the same ground with
        more depth and history.
      </p>
    </div>
  );
}

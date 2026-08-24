import { useEffect, useRef, useState } from "react";
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

/** An anchored section heading inside a module; the floating table of
 * contents discovers these and tracks which one the reader is in. Ids must
 * be unique across modules (prefix with the module: "m4-blame"). */
export function SectionHeader({ id, title }: { id: string; title: string }) {
  return (
    <h3 id={id} className="module-section-title">
      {title}
    </h3>
  );
}

/** Floating on-this-page navigation. Render once per module, anywhere inside
 * the module's article: it discovers that article's SectionHeaders from the
 * DOM, fixes itself to the right gutter on wide screens (hidden where there
 * is no gutter), and highlights the section currently in view. Modules stay
 * mounted but hidden on tab switches, which also hides their toc. */
export function ModuleToc() {
  const ref = useRef<HTMLElement>(null);
  const [sections, setSections] = useState<{ id: string; title: string }[]>([]);
  const [active, setActive] = useState<string | null>(null);

  useEffect(() => {
    const article = ref.current?.closest("article");
    if (!article) return;
    const headers = Array.from(
      article.querySelectorAll<HTMLHeadingElement>(".module-section-title"),
    );
    setSections(headers.map((h) => ({ id: h.id, title: h.textContent ?? "" })));
    const onScroll = () => {
      let current: string | null = null;
      for (const h of headers) {
        if (h.offsetParent === null) continue; // module hidden by tab switch
        if (h.getBoundingClientRect().top <= 140) current = h.id;
      }
      setActive(current);
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("hashchange", onScroll);
    onScroll();
    return () => {
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("hashchange", onScroll);
    };
  }, []);

  return (
    <nav ref={ref} className="module-toc" aria-label="On this page">
      <p className="module-toc-label">On this page</p>
      <ul>
        {sections.map((s) => (
          <li key={s.id}>
            <button
              className={`module-toc-item ${active === s.id ? "module-toc-active" : ""}`}
              onClick={() =>
                document
                  .getElementById(s.id)
                  ?.scrollIntoView({ behavior: "smooth", block: "start" })
              }
            >
              {s.title}
            </button>
          </li>
        ))}
      </ul>
    </nav>
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

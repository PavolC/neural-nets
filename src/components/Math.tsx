import katex from "katex";
import "katex/dist/katex.min.css";

// KaTeX wrappers. Per the content rules (CLAUDE.md), every displayed
// equation gets a one-sentence plain-language gloss immediately after it.

function render(tex: string, displayMode: boolean): { __html: string } {
  return { __html: katex.renderToString(tex, { displayMode, throwOnError: false }) };
}

export function Eq({ tex, gloss }: { tex: string; gloss: string }) {
  return (
    <div className="eq">
      <div dangerouslySetInnerHTML={render(tex, true)} />
      <p className="eq-gloss">{gloss}</p>
    </div>
  );
}

export function M({ tex }: { tex: string }) {
  return <span dangerouslySetInnerHTML={render(tex, false)} />;
}

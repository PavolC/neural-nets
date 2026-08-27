import { COURSE } from "./brand";

/**
 * The course's mark: its glyph in white on a rounded tile of its accent hue.
 * Decorative in every place it is used, because the wordmark beside it always
 * carries the name in text, so it is hidden from screen readers rather than
 * given an alt text that would be read out twice.
 *
 * Sized by --monogram-size, which the footer lowers, so one component serves
 * both placements.
 */
export function Monogram() {
  const { viewBox, d, strokeWidth } = COURSE.glyph;
  return (
    <svg className="brand-monogram" viewBox={viewBox} aria-hidden="true" focusable="false">
      <rect width="32" height="32" rx="7" fill="var(--accent)" />
      <path
        d={d}
        fill="none"
        stroke="var(--on-accent)"
        strokeWidth={strokeWidth}
        strokeLinecap="round"
      />
    </svg>
  );
}

/**
 * Counting readers, without tracking them.
 *
 * The design doc's non-goals said "no analytics", beside "no accounts" and
 * "no backend", and for a course built for one named learner that was right:
 * there was nothing to measure that asking him would not answer better.
 *
 * It stopped being right when the goal became reach. A course nobody can
 * measure is a course whose author cannot tell the difference between "two
 * hundred people read Module 1 and stopped" and "nobody arrived", and those
 * two failures have opposite fixes. The method this course was built with runs
 * on knowing where a reader stops; that instrument existed for one reader and
 * for nobody else.
 *
 * What this is NOT allowed to become: the page must stay usable with the
 * request blocked, must set no cookie, must send nothing a learner typed, and
 * must never send the contents of the editor. Progress stays in localStorage
 * and stays the learner's. GoatCounter is the choice because it is free for
 * non-commercial use, stores no personal data and sets no cookies, so the
 * promise the README makes is still true with it turned on.
 *
 * It is off unless VITE_GOATCOUNTER names a site. With the variable unset this
 * module compiles to a function that returns immediately, no script is
 * requested, and the build is byte-for-byte the private one. Turn it on with
 *
 *     VITE_GOATCOUNTER=https://yourcode.goatcounter.com/count npm run build
 *
 * or by putting that line in .env.local, which .gitignore already covers.
 */

const ENDPOINT = import.meta.env.VITE_GOATCOUNTER as string | undefined;

export function startAnalytics(): void {
  if (!ENDPOINT) return;

  // Never count the author reloading his own dev server.
  if (import.meta.env.DEV) return;

  // Respect a stated preference not to be measured. This is the one signal a
  // browser gives that means "do not count me", and honouring it costs a line.
  if (navigator.doNotTrack === "1" || (window as { doNotTrack?: string }).doNotTrack === "1") {
    return;
  }

  const script = document.createElement("script");
  script.async = true;
  script.dataset.goatcounter = ENDPOINT;
  script.src = "//gc.zgo.at/count.js";
  // A blocked or failed request is the expected case for a good fraction of
  // this audience, and it must not reach the console as an error the learner
  // has to wonder about.
  script.onerror = () => script.remove();
  document.head.appendChild(script);
}

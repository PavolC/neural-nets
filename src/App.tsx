import { Suspense, useEffect, useRef, useState } from "react";
import { Masthead } from "./brand/Masthead";
import { SeriesFooter } from "./brand/SeriesFooter";
import { DockHandle, DockShell } from "./components/DockShell";
import { WorkbenchProvider, useWorkbench } from "./components/WorkbenchProvider";
import { StartPage } from "./start/StartPage";
import { MODULES } from "./modules/NN";
import { ensureLibrary, refreshMarkers } from "./state/progress";

const START_TAB = "start";

// The learner's file is built out of whatever this browser already holds
// before anything renders, so no component ever sees a half-migrated state.
ensureLibrary();
// A file saved before the chapters rename still carries "Module" markers.
refreshMarkers();

export default function App() {
	return (
		<WorkbenchProvider>
			<AppShell />
		</WorkbenchProvider>
	);
}

// The active tab lives in the URL hash (#start, #c1) so a reload or a shared
// link lands on the same page. Every chapter is always reachable; a bare link
// and an unknown hash both land on the start page, which is where a reader who
// has not seen the course before needs to arrive.
//
// This course shipped its chapters as modules, addressed #m1 to #m10, and those
// addresses are in bookmarks and in links already sent. An old id still resolves
// to the chapter it always named, and the effect below rewrites the address bar
// to the new one, so a stale link works once and corrects itself. A straight
// swap would have broken every one of them (METHOD.md).
const RENAMED_TAB = /^m(\d+)$/;

function tabFromHash(): string {
	const id = window.location.hash.slice(1);
	if (MODULES.some((m) => m.id === id)) return id;
	const renamed = id.replace(RENAMED_TAB, "c$1");
	return MODULES.some((m) => m.id === renamed) ? renamed : START_TAB;
}

function AppShell() {
	const workbench = useWorkbench();
	const [tab, setTab] = useState<string>(tabFromHash);
	// Which chapters have been opened. Chapters load on demand, but once one is
	// rendered it stays rendered, so its editor and visualization state survives
	// every later tab switch.
	const [opened, setOpened] = useState<Set<string>>(() => new Set([tabFromHash()]));
	const selectTab = (id: string) => {
		window.location.hash = id;
		setTab(id);
	};
	useEffect(() => {
		setOpened((prev) => (prev.has(tab) ? prev : new Set(prev).add(tab)));
	}, [tab]);

	// Warm the remaining chapter chunks once the page is quiet, so deferring them
	// costs a download on first paint and nothing on navigation.
	useEffect(() => {
		const warm = () => MODULES.forEach((m) => m.preload?.());
		const idle = window.requestIdleCallback;
		if (idle) {
			const id = idle(warm);
			return () => window.cancelIdleCallback(id);
		}
		const id = window.setTimeout(warm, 1500);
		return () => window.clearTimeout(id);
	}, []);

	useEffect(() => {
		const onHashChange = () => setTab(tabFromHash());
		window.addEventListener("hashchange", onHashChange);
		return () => window.removeEventListener("hashchange", onHashChange);
	}, []);

	// An unknown hash resolves to chapter 1; rewrite it so the URL never claims
	// to be somewhere the reader is not.
	useEffect(() => {
		if (window.location.hash.slice(1) !== tab) window.location.hash = tab;
	}, [tab]);

	// Chapters are stacked in one document and hidden rather than unmounted, so
	// without this a tab switch keeps the previous chapter's scroll offset and the
	// browser clamps it into the new one: leaving Chapter 4 halfway through used to
	// land the reader on Chapter 2's closing recap. Start every chapter at its top,
	// which is what the Continue button already did.
	// The tab strip is one panning row below 720px, so the active tab can sit
	// off-screen: a link straight to Chapter 8 would show the row scrolled to
	// Start. Nothing else moves, hence inline and nearest rather than a scroll
	// into view that would also drag the page.
	const tabStrip = useRef<HTMLElement>(null);
	useEffect(() => {
		const active = tabStrip.current?.querySelector<HTMLElement>(".tab-active");
		// Centred rather than nearest: nearest parks the active tab against the
		// edge fade with no neighbours beside it, which loses the one thing the
		// strip is for, namely where you are in the sequence. The first and last
		// tabs clamp to their ends on their own.
		active?.scrollIntoView({ inline: "center", block: "nearest" });
	}, [tab]);

	// Both navigations read the same list: the strip that wraps to two rows on a
	// wide screen, and the picker that replaces it below 880px.
	const navItems = [{ id: START_TAB, label: "Start" }, ...MODULES.map((m) => ({ id: m.id, label: m.navLabel }))];
	const current = navItems.find((n) => n.id === tab) ?? navItems[0];
	const [navOpen, setNavOpen] = useState(false);
	// A chapter switch from the picker closes it; one from anywhere else (the
	// Continue button, a deep link) leaves it as it was, which is closed.
	useEffect(() => setNavOpen(false), [tab]);

	const panels = useRef<Record<string, HTMLDivElement | null>>({});
	// The tab this effect last acted on. A boolean "have I mounted yet" flag does
	// not survive StrictMode, which runs the effect, cleans up, and runs it again:
	// the first pass sets the flag and the second pass sails through and scrolls
	// and steals focus on arrival. Comparing tab values is idempotent.
	const actedOn = useRef(tab);

	useEffect(() => {
		// First render, or a re-run for a tab we already handled: leave the page
		// alone. A deep link should not fight the browser's own restoration.
		if (actedOn.current === tab) return;
		actedOn.current = tab;
		window.scrollTo(0, 0);
		// Rescue focus only when the switch is what took it away. The tab strip
		// lives in the header, outside the panels, so a tab click hides nothing the
		// reader was focused on; the one control that does is "Continue to Chapter
		// N", which sits inside the panel being hidden. Hence the test: is the
		// focused element inside a subtree we just hid? Asking whether
		// document.activeElement is <body> answers a different question, and
		// answers it wrongly twice over. Too early, because React has hidden the
		// panel but the browser has not blurred the button in it yet; too eager,
		// because nothing focused is also the resting state of a freshly loaded
		// page. Both readings put a focus ring around the chapter title.
		const active = document.activeElement as HTMLElement | null;
		const lostFocus = !!active && active !== document.body && (!active.isConnected || active.closest("[hidden]") !== null);
		if (!lostFocus) return;
		const heading = panels.current[tab]?.querySelector<HTMLElement>("h2");
		if (heading) {
			heading.tabIndex = -1;
			heading.focus({ preventScroll: true });
		}
	}, [tab]);

	return (
		<>
			<DockShell dockState={workbench.dockState}>
				<Masthead
					compact={tab !== START_TAB}
					nav={
						<>
							<nav className="tabs" ref={tabStrip} aria-label="Course modules">
								{navItems.map((n) => (
									<button key={n.id} className={`tab ${tab === n.id ? "tab-active" : ""}`} aria-current={tab === n.id ? "page" : undefined} onClick={() => selectTab(n.id)}>
										{n.label}
									</button>
								))}
							</nav>
							{/* Below 880px the strip cannot show eleven tabs without panning,
                and dragging a row sideways to find Chapter 9 is the worst way to
                offer a list of eleven things. The same list, folded: one line
                that names where you are and opens the rest vertically. Only one
                of the two is ever displayed, and the other is display: none, so
                a screen reader is not read the course twice. */}
							<nav className="module-picker" aria-label="Course modules">
								<button className="module-picker-toggle" aria-expanded={navOpen} onClick={() => setNavOpen((o) => !o)}>
									<span className="module-picker-where">Chapter</span>
									<span className="module-picker-title">{current.label}</span>
									<span className="module-picker-caret" aria-hidden="true">
										{navOpen ? "\u25b2" : "\u25bc"}
									</span>
								</button>
								{navOpen && (
									<ul className="module-picker-list">
										{navItems.map((n) => (
											<li key={n.id}>
												<button
													className={`module-picker-item ${tab === n.id ? "module-picker-current" : ""}`}
													aria-current={tab === n.id ? "page" : undefined}
													onClick={() => selectTab(n.id)}
												>
													{n.label}
												</button>
											</li>
										))}
									</ul>
								)}
							</nav>
						</>
					}
				/>
				<main>
					<div
						hidden={tab !== START_TAB}
						ref={(el) => {
							panels.current[START_TAB] = el;
						}}
					>
						<StartPage onGoTo={selectTab} />
					</div>
					{/* A chapter renders on first visit and stays rendered after that, so
            tab switches never lose editor or visualization state. */}
					{MODULES.map((m, i) => {
						const next = MODULES[i + 1];
						return (
							<div
								key={m.id}
								hidden={tab !== m.id}
								ref={(el) => {
									panels.current[m.id] = el;
								}}
							>
								<Suspense fallback={<p className="module-loading">Loading {m.navLabel}...</p>}>{opened.has(m.id) && <m.Component />}</Suspense>
								{next && (
									<div className="next-module">
										<button onClick={() => selectTab(next.id)}>Continue to {next.navLabel} →</button>
									</div>
								)}
							</div>
						);
					})}
				</main>
				<SeriesFooter>
					<p>
						Built by <a href="https://github.com/PavolC">Pavol Chvala</a>. The course is open source at <a href="https://github.com/PavolC/neural-nets">PavolC/neural-nets</a>.
					</p>
					<p>
						Adapted from Michael A. Nielsen,{" "}
						<a href="http://neuralnetworksanddeeplearning.com/">
							<em>Neural Networks and Deep Learning</em>
						</a>
						, Determination Press, 2015, licensed <a href="https://creativecommons.org/licenses/by-nc/3.0/">CC BY-NC 3.0</a>. This course's content follows its source and is
						non-commercial. Reference implementations are adapted from Nielsen's MIT-licensed code, and the software around them is MIT too. Full terms in{" "}
						<a href="./LICENSE.txt">LICENSE</a>.
					</p>
				</SeriesFooter>
			</DockShell>
			<DockHandle onOpen={() => workbench.open()} />
		</>
	);
}

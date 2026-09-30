"""FORM/AI design system: tokens, typography and Streamlit overrides.

Fonts and base colours are set in .streamlit/config.toml so native widgets
follow them; this sheet styles the custom HTML and removes Streamlit chrome.
"""

CSS = """
<style>
:root {
  --bg: #0C0C0D; --raise: #131315; --raise-2: #19191C;
  --line: #242427; --line-2: #34343A;
  --ink: #EDECE8; --ink-2: #B4B2AC; --muted: #85837D;
  --accent: #F4513C; --accent-soft: rgba(244, 81, 60, .12);
  --good: #55C08A; --warning: #E7B24A; --bad: #F4513C; --neutral: #85837D;
  --display: "Archivo", "IBM Plex Sans", sans-serif;
  --sans: "IBM Plex Sans", system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, monospace;
  --r: 4px;
  --ease: cubic-bezier(.16, 1, .3, 1);
}

/* ---- Streamlit chrome ------------------------------------------------ */
[data-testid="stHeader"], [data-testid="stDecoration"], [data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"], footer { display: none !important; }
.stApp { background: var(--bg); }
.block-container, [data-testid="stMainBlockContainer"] {
  max-width: 1320px; padding: 0 40px 48px !important;
}
.stMarkdown p { color: var(--ink-2); }
::selection { background: var(--accent); color: var(--bg); }

/* ---- Typography ------------------------------------------------------ */
h1, h2, h3, .display { font-family: var(--display) !important; color: var(--ink); }
.mono, .label { font-family: var(--mono); }
.label {
  font-size: .7rem; letter-spacing: .14em; text-transform: uppercase; color: var(--muted);
  font-weight: 500;
}
.page-title {
  font: 800 2.4rem/1 var(--display); font-stretch: 108%; letter-spacing: -.02em;
  text-transform: uppercase; margin: 0; padding: 0; color: var(--ink);
}
.section-title {
  font: 750 1.9rem/1.05 var(--display); font-stretch: 104%; letter-spacing: -.015em;
  margin: 0 0 .5rem; color: var(--ink);
}
.stMarkdown p.lede, .lede { font-size: 1.05rem; line-height: 1.6; color: var(--ink-2); max-width: 60ch; margin: 0; }
.stMarkdown p.fine, .fine { font-size: .8rem; color: var(--muted); line-height: 1.5; }

/* ---- Top navigation ---------------------------------------------------- */
[data-testid="stLayoutWrapper"]:has(> .st-key-topnav) {
  position: sticky; top: 0; z-index: 50; background: rgba(12, 12, 13, .9);
  backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
}
.st-key-topnav { border-bottom: 1px solid var(--line); padding: 12px 0; min-height: 64px; }
.brand { display: flex; align-items: center; gap: 12px; white-space: nowrap; }
.brand svg { width: 26px; height: 26px; flex: none; }
.brand-name { font: 800 1.2rem/1 var(--display); font-stretch: 112%; letter-spacing: .01em; color: var(--ink); }
.brand-name i { font-style: normal; color: var(--accent); }
.brand-sub { font: 500 .6rem/1 var(--mono); letter-spacing: .2em; color: var(--muted); padding-left: 12px;
  border-left: 1px solid var(--line-2); }
.st-key-navlinks { gap: 4px !important; }
.st-key-navlinks button {
  font: 500 .9rem var(--sans) !important; color: var(--ink-2) !important; padding: 6px 12px !important;
  position: relative; border-radius: 0 !important;
}
.st-key-navlinks button:hover { color: var(--ink) !important; }
.st-key-navlinks button::after {
  content: ""; position: absolute; left: 12px; right: 12px; bottom: -2px; height: 2px;
  background: var(--accent); transform: scaleX(0); transition: transform .35s var(--ease);
}

/* ---- Buttons ----------------------------------------------------------- */
button[data-testid="stBaseButton-primary"], button[data-testid="stBaseButton-secondary"] {
  font: 600 .85rem var(--sans) !important; letter-spacing: .02em; min-height: 44px;
  padding: 0 22px !important; transition: transform .2s var(--ease), background .2s, border-color .2s, color .2s;
}
button[data-testid="stBaseButton-primary"] { color: #120605 !important; }
button[data-testid="stBaseButton-primary"]:hover { background: #FF6650 !important; border-color: #FF6650 !important; }
button[data-testid="stBaseButton-secondary"] { background: transparent !important; border-color: var(--line-2) !important; }
button[data-testid="stBaseButton-secondary"]:hover { border-color: var(--ink-2) !important; color: var(--ink) !important; }
button[data-testid^="stBaseButton"]:active { transform: translateY(1px); }
button[data-testid="stBaseButton-tertiary"] { color: var(--ink-2) !important; }
button[data-testid="stBaseButton-tertiary"]:hover { color: var(--accent) !important; }
.st-key-topnav button[data-testid="stBaseButton-primary"] { min-height: 38px; }

/* segmented controls (source switcher, exercise selector) */
[data-testid="stButtonGroup"] button { font: 500 .85rem var(--sans) !important; min-height: 38px; }
[data-testid="stWidgetLabel"] p { font: 500 .7rem var(--mono) !important; letter-spacing: .14em;
  text-transform: uppercase; color: var(--muted) !important; }

/* ---- Hero ---------------------------------------------------------------- */
.st-key-hero { position: relative; min-height: min(640px, calc(100vh - 90px)); padding-bottom: 24px; }
.hero-copy { animation: rise .9s var(--ease) both; margin-bottom: 18px; }
.hero-kicker { display: flex; align-items: center; gap: 10px; margin-bottom: 28px; }
.hero-kicker::before { content: ""; width: 28px; height: 1px; background: var(--accent); }
.hero-title {
  font: 850 clamp(2.9rem, 4.9vw, 4.7rem)/.94 var(--display); font-stretch: 110%;
  letter-spacing: -.03em; text-transform: uppercase; color: var(--ink); margin: 0 0 28px;
}
.hero-title em { font-style: normal; color: var(--accent); }
.hero-figure { animation: fade 1.4s var(--ease) .15s both; }
.fx-hero, .fx-card, .fx-pair, .fx-guide { display: block; width: 100%; height: auto; overflow: visible; }
.fx-still { display: none; }
@media (prefers-reduced-motion: reduce) {
  .fx-motion { display: none; } .fx-still { display: inline; }
  .hero-copy, .hero-figure, .rise { animation: none !important; }
}
@keyframes rise { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
@keyframes fade { from { opacity: 0; } to { opacity: 1; } }
.rise { animation: rise .7s var(--ease) both; }

/* ---- Stat band ------------------------------------------------------------ */
.stats { display: grid; grid-template-columns: repeat(4, 1fr); border-top: 1px solid var(--line);
  border-bottom: 1px solid var(--line); margin: 8px 0 72px; }
.stats > div { padding: 26px 24px 24px 0; }
.stats > div + div { padding-left: 24px; border-left: 1px solid var(--line); }
.stat-n { font: 800 2.6rem/1 var(--display); font-stretch: 108%; color: var(--ink); letter-spacing: -.02em; }
.stat-n small { font-size: 1rem; color: var(--muted); font-weight: 600; margin-left: 4px; }
.stat-t { margin-top: 10px; font-size: .88rem; color: var(--ink-2); line-height: 1.45; max-width: 26ch; }

/* ---- Exercise cards --------------------------------------------------------- */
.ex-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-top: 28px; }
.ex-card { background: var(--raise); border-top: 2px solid var(--line-2); padding: 18px 18px 20px;
  transition: border-color .3s var(--ease), background .3s; }
.ex-card:hover { border-top-color: var(--accent); background: var(--raise-2); }
.ex-card .fig { height: 230px; display: flex; align-items: flex-end; }
.ex-card .fig svg { max-height: 100%; }
.ex-name { font: 750 1.15rem/1.1 var(--display); font-stretch: 106%; text-transform: uppercase;
  color: var(--ink); margin: 18px 0 8px; letter-spacing: .01em; }
.ex-meta { font: .72rem/1.7 var(--mono); color: var(--muted); letter-spacing: .04em; }
.ex-meta b { color: var(--ink-2); font-weight: 500; }

/* ---- Analysis screen ----------------------------------------------------------- */
.stage { position: relative; background: var(--raise); border: 1px solid var(--line);
  aspect-ratio: 4 / 3; display: flex; flex-direction: column; align-items: center;
  justify-content: center; overflow: hidden; }
.stage .fx-guide { height: 72%; width: auto; }
.stage-note { position: absolute; left: 20px; bottom: 16px; right: 20px; display: flex;
  justify-content: space-between; gap: 16px; }
.stage-note span { font: .72rem var(--mono); color: var(--muted); letter-spacing: .06em; }
.st-key-stagecol [data-testid="stImage"] img, .st-key-stagecol [data-testid="stVideo"],
.st-key-stagecol video { border: 1px solid var(--line); border-radius: 0; }

.panel { border-top: 2px solid var(--line-2); padding-top: 14px; }
.panel.good { border-top-color: var(--good); } .panel.warning { border-top-color: var(--warning); }
.panel.bad { border-top-color: var(--bad); }
.panel-head { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; }
.panel-ex { font: 800 1.5rem/1 var(--display); font-stretch: 108%; text-transform: uppercase; color: var(--ink); }
.readouts { display: grid; grid-template-columns: .8fr 1.4fr 1fr; margin: 18px 0 6px;
  border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); }
.readouts > div { padding: 12px 0; }
.readouts > div + div { padding-left: 14px; border-left: 1px solid var(--line); }
.ro-v { font: 800 2.2rem/1.05 var(--display); color: var(--ink); margin-top: 6px; letter-spacing: -.02em;
  font-variant-numeric: tabular-nums; }
.ro-v.phase { font-size: 1rem; font-family: var(--mono); font-weight: 500; letter-spacing: .06em;
  padding-top: 12px; line-height: 1.3; color: var(--accent); }
.ro-v small { font-size: .8rem; font-weight: 600; color: var(--muted); margin-left: 4px; }
.block { margin-top: 18px; }
.block .label { display: block; margin-bottom: 8px; }
.checks { list-style: none; margin: 0; padding: 0; }
.checks li { display: grid; grid-template-columns: 56px 1fr auto; gap: 10px; align-items: baseline;
  padding: 6px 0; font-size: .9rem; color: var(--ink); }
.checks li + li { border-top: 1px solid var(--line); }
.checks li b { font: 500 .85rem var(--mono); color: var(--ink-2); font-variant-numeric: tabular-nums; }
.tag { font: 500 .62rem/1 var(--mono); letter-spacing: .12em; padding: 4px 0; text-align: center;
  border: 1px solid currentColor; border-radius: 2px; }
.tag.good { color: var(--good); } .tag.warning { color: var(--warning); }
.tag.bad { color: var(--bad); } .tag.neutral { color: var(--neutral); }
.feedback { margin: 0; padding: 0; list-style: none; }
.feedback li { font-size: .98rem; line-height: 1.45; color: var(--ink); padding: 3px 0 3px 14px;
  border-left: 2px solid var(--accent); margin-bottom: 6px; }
.feedback.quiet li { color: var(--ink-2); border-left-color: var(--line-2); }
.telemetry { display: flex; flex-wrap: wrap; gap: 6px 18px; margin-top: 18px; padding-top: 12px;
  border-top: 1px solid var(--line); font: .72rem var(--mono); color: var(--muted); letter-spacing: .04em; }
.telemetry b { color: var(--ink-2); font-weight: 500; }
.warnings li { border-left-color: var(--warning); color: var(--ink-2); }

/* ---- Report ----------------------------------------------------------------- */
.kpis { display: grid; grid-template-columns: repeat(4, 1fr); border-top: 1px solid var(--line-2);
  border-bottom: 1px solid var(--line); margin: 6px 0 4px; }
.kpis > div { padding: 18px 20px 20px 0; }
.kpis > div + div { padding-left: 20px; border-left: 1px solid var(--line); }
.kpi-v { font: 800 3rem/1 var(--display); font-stretch: 106%; color: var(--ink); margin-top: 10px;
  letter-spacing: -.02em; font-variant-numeric: tabular-nums; }
.kpi-v small { font-size: 1rem; color: var(--muted); font-weight: 600; margin-left: 6px; letter-spacing: 0; }
.kpi-s { margin-top: 8px; font: .75rem var(--mono); color: var(--muted); }
.kpi-s.good { color: var(--good); } .kpi-s.warning { color: var(--warning); } .kpi-s.bad { color: var(--bad); }
.meta { font: .75rem/1.6 var(--mono); color: var(--muted); letter-spacing: .03em; }
.meta b { color: var(--ink-2); font-weight: 500; }
.bars { margin: 0; padding: 0; list-style: none; }
.bars li { padding: 10px 0 12px; }
.bars li + li { border-top: 1px solid var(--line); }
.bar-row { display: flex; justify-content: space-between; align-items: baseline; font-size: .92rem; color: var(--ink); }
.bar-row b { font: 600 1rem var(--mono); font-variant-numeric: tabular-nums; }
.bar { height: 4px; margin-top: 8px; border-radius: 2px; transform-origin: left;
  animation: grow .9s var(--ease) both; }
.bar.good { background: var(--good); } .bar.warning { background: var(--warning); } .bar.bad { background: var(--bad); }
.bar.neutral { background: var(--line-2); }
@keyframes grow { from { transform: scaleX(0); } to { transform: scaleX(1); } }

/* ---- Exercises page ------------------------------------------------------------ */
.ex-figure { background: var(--raise); padding: 28px 28px 20px; }
.ex-figure .fx-card { max-height: 330px; margin: 0 auto; }
.ex-figure .fx-pair { max-height: 280px; margin: 0 auto; }
.facts { display: grid; grid-template-columns: 1fr 1fr; border-top: 1px solid var(--line); margin-top: 24px; }
.facts > div { padding: 14px 16px 14px 0; border-bottom: 1px solid var(--line); }
.facts > div:nth-child(even) { padding-left: 16px; border-left: 1px solid var(--line); }
.facts .v { margin-top: 6px; color: var(--ink); font-size: .95rem; line-height: 1.45; }
.phases { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.phases span { font: 500 .7rem var(--mono); letter-spacing: .08em; color: var(--ink-2);
  border: 1px solid var(--line-2); padding: 5px 8px; border-radius: 2px; }
.phases span.turn { color: var(--accent); border-color: var(--accent); }
table.spec { width: 100%; border-collapse: collapse; margin-top: 10px; }
table.spec th { font: 500 .66rem var(--mono); letter-spacing: .12em; text-transform: uppercase;
  color: var(--muted); text-align: left; padding: 8px 10px 8px 0; border-bottom: 1px solid var(--line-2); }
table.spec td { padding: 11px 10px 11px 0; font-size: .9rem; color: var(--ink); border-bottom: 1px solid var(--line); }
table.spec td.n { font-family: var(--mono); font-size: .82rem; color: var(--ink-2); font-variant-numeric: tabular-nums; }
.scale { display: flex; height: 36px; margin-top: 14px; }
.scale > div { position: relative; border-left: 1px solid var(--bg); padding: 0 8px;
  display: flex; align-items: center; font: .72rem var(--mono); color: var(--bg); font-weight: 500; }
.scale-axis { display: flex; justify-content: space-between; font: .68rem var(--mono); color: var(--muted); margin-top: 6px; }

/* ---- About --------------------------------------------------------------------- */
.stMarkdown p.statement, .statement { font: 750 clamp(2rem, 3.4vw, 3rem)/1.02 var(--display); font-stretch: 106%;
  letter-spacing: -.02em; color: var(--ink); margin: 0; }
.dip { display: grid; grid-template-columns: 1fr 1fr; gap: 0 40px; }
.dip > div { padding: 16px 0; border-top: 1px solid var(--line); }
.dip h4 { font: 600 .98rem var(--sans) !important; color: var(--ink); margin: 0 0 4px; padding: 0; }
.stMarkdown .dip p { margin: 0; font-size: .9rem; color: var(--ink-2); line-height: 1.55; }
.flow { display: flex; flex-wrap: wrap; gap: 8px 0; align-items: center; margin-top: 10px; }
.flow span { font: 500 .75rem var(--mono); color: var(--ink); padding: 8px 12px; background: var(--raise-2);
  border-radius: 2px; letter-spacing: .03em; }
.flow i { width: 18px; height: 1px; background: var(--line-2); display: inline-block; }
.cols3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 40px; }
.stMarkdown .cols3 p, .cols3 li { font-size: .9rem; color: var(--ink-2); line-height: 1.6; }
.cols3 ul { padding-left: 1rem; margin: 0; }

.stMarkdown ul.bars, .stMarkdown ul.checks, .stMarkdown ul.feedback { padding-left: 0; margin: 0; }
.stMarkdown ul.bars li, .stMarkdown ul.checks li, .stMarkdown ul.feedback li { margin-left: 0; }

/* ---- Native widgets ------------------------------------------------------------ */
[data-testid="stExpander"] details { border-radius: var(--r); border-color: var(--line) !important; }
[data-testid="stExpander"] summary p { font-weight: 500; color: var(--ink) !important; }
[data-testid="stTabs"] button[role="tab"] p { font: 500 .9rem var(--sans); }
[data-testid="stAlert"] { border-radius: var(--r); }
[data-testid="stFileUploaderDropzone"] { background: var(--raise) !important; border: 1px dashed var(--line-2) !important;
  border-radius: var(--r); padding: 28px !important; }
[data-testid="stProgress"] > div > div > div > div { background: var(--accent) !important; }
[data-testid="stCaptionContainer"], .stCaption { color: var(--muted) !important; }
hr { border-color: var(--line) !important; }

.site-foot { margin-top: 80px; padding-top: 20px; border-top: 1px solid var(--line);
  display: flex; justify-content: space-between; gap: 24px; flex-wrap: wrap; }
.spacer-s { height: 8px; } .spacer-m { height: 28px; } .spacer-l { height: 64px; }

/* ---- Narrow windows (a laptop split screen, a tablet) ---------------------------- */
@media (max-width: 1100px) {
  .ex-grid { grid-template-columns: repeat(2, 1fr); }
  .brand-sub { display: none; }
}
@media (max-width: 760px) {
  .block-container, [data-testid="stMainBlockContainer"] { padding: 0 16px 40px !important; }
  .stats, .kpis { grid-template-columns: 1fr 1fr; }
  .stats > div:nth-child(3), .kpis > div:nth-child(3) { padding-left: 0; border-left: 0; }
  .ex-grid, .dip, .cols3, .facts { grid-template-columns: 1fr; }
  .facts > div:nth-child(even) { padding-left: 0; border-left: 0; }
}
</style>
"""

# Active nav link: the underline grows under the current page's button.
NAV_ACTIVE = """<style>.st-key-nav_{slug} button {{ color: var(--ink) !important; }}
.st-key-nav_{slug} button::after {{ transform: scaleX(1) !important; }}</style>"""

HERO_BG = """<style>.st-key-hero {{ background: url({uri}) left bottom / 100% auto no-repeat; }}</style>"""

BRAND_MARK = (
    # A joint node with its two limbs and angle arc: the analyser's core measurement.
    '<svg viewBox="0 0 26 26" aria-hidden="true"><path d="M4 22 L12 12 L22 17" fill="none" '
    'stroke="#EDECE8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
    '<path d="M8.25 16.7 A6 6 0 0 0 17.4 14.7" fill="none" stroke="#F4513C" stroke-width="1.6"/>'
    '<circle cx="12" cy="12" r="3.2" fill="#0C0C0D" stroke="#F4513C" stroke-width="2"/></svg>'
)

"""Streamlit entry point:  streamlit run app.py"""
from __future__ import annotations

import streamlit as st

from ui import pages
from ui.components import css, read_settings
from ui.styles import CSS

st.set_page_config(page_title="FORM/AI - Gym form analysis", page_icon=":material/accessibility_new:",
                   layout="wide")
css(CSS)

# ?page=... deep-links on first load; afterwards the top navigation drives it.
requested = st.query_params.get("page", "Home")
st.session_state.setdefault("page", requested if requested in pages.PAGES else "Home")
settings = read_settings()
pages.top_nav()

PAGE_FUNCS = {
    "Home": pages.home, "Live Analysis": pages.live, "Video Upload": pages.upload,
    "Demo Mode": pages.demo, "Exercise Analysis": pages.exercise_guide,
    "Results": pages.results, "About": pages.about,
}
try:
    PAGE_FUNCS[st.session_state["page"]](settings)
except RuntimeError as exc:          # e.g. pose model missing and download failed
    st.error(str(exc))
pages.footer()

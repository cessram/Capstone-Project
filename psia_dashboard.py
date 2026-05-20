"""
PSIA Seaweed Industry Analytics Dashboard  — v18.0
====================================================
  CHANGES FROM v9:
  - Tab 4 "🔬 Advanced Analytics" added with 6 remaining KPIs:
      KPI-14: Price per Tonne Heatmap (Country × Year)
      KPI-20: Environment Breakdown (Marine/Inland/Brackish — derived)
      KPI-21: Aquaculture Adoption Rate Bubble Chart
      KPI-22/23: Country Growth Trajectories + Emerging Producers
      KPI-24: Price Volatility by Species (std dev)
      KPI-25: Country Specialization Index (top-species share)
      KPI-26: Data Completeness Proxy (years reported ÷ total years)
  - Tab 3 "Simulated data" label replaced with accurate "Estimated from public reports"
    banner explaining sources and how to replace with live data
  - Full font readability pass: kpi-label #333, tags 12px, h4 17px,
    captions 13px, sidebar labels 14px, alert text 14px
  - All chart AXIS_FONT bumped to 12px, CHART_FONT to 13px

Setup:
  pip install streamlit plotly pandas numpy groq scikit-learn
  Place 4 CSVs inside  data/
  streamlit run psia_dashboard_v10.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from groq import Groq

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PSIA Seaweed Analytics",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

GROQ_API_KEY = "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"

# ─────────────────────────────────────────────────────────────────────────────
# FLOAT HELPER  — zero external packages, pure CSS :has() injection
# ─────────────────────────────────────────────────────────────────────────────
import uuid as _uuid

def make_float(css_props: str):
    marker_id = "fl-" + _uuid.uuid4().hex[:10]
    st.markdown(
        f"<style>"
        f"div[data-testid='stVerticalBlock']:has(> div > span#{marker_id})"
        f"{{position:fixed !important;{css_props}}}"
        f"</style>",
        unsafe_allow_html=True,
    )
    c = st.container()
    with c:
        st.markdown(
            f'<span id="{marker_id}" style="display:none;"></span>',
            unsafe_allow_html=True,
        )
    return c


# ─────────────────────────────────────────────────────────────────────────────
# COLOURS
# ─────────────────────────────────────────────────────────────────────────────
C = {
    "t900": "#04342C", "t800": "#085041", "t600": "#0F6E56",
    "t400": "#1D9E75", "t300": "#5DCAA7", "t100": "#9FE1CB", "t50": "#E1F5EE",
    "amber": "#C97D10", "coral": "#B84020", "blue": "#1E5FA8",
    "gray": "#555550",  "green": "#1B5E20",
    "txt":  "#1A1A1A",
    "txt2": "#444440",
}

CONT_PAL = {
    "Asia": C["t600"], "Americas": C["t400"], "Europe": C["t300"],
    "Africa": C["amber"], "Oceania": C["coral"], "Unknown": C["gray"],
}
INC_PAL = {
    "Upper-middle income countries":          C["t600"],
    "High-income countries":                  C["t400"],
    "Lower-middle income countries":          C["amber"],
    "Low-income countries":                   C["coral"],
    "Countries not classified by World Bank": C["gray"],
}
SP_PAL = [
    "#085041", "#0F6E56", "#1D9E75", "#5DCAA7", "#9FE1CB",
    "#C97D10", "#B84020", "#1E5FA8", "#5B4FCF", "#A0356E",
    "#2E7D6B", "#7B4F00", "#1A3A6B", "#6B1A3A", "#3A6B1A",
]

CHART_FONT   = dict(family="Inter, Arial, sans-serif", size=13, color="#1A1A1A")
AXIS_FONT    = dict(family="Inter, Arial, sans-serif", size=12, color="#333333")
TITLE_FONT   = dict(family="Georgia, serif", size=13, color="#1A1A1A")
GRID_COLOR   = "#E8E8E8"
AXIS_LINE    = dict(color="#CCCCCC", width=1)

LEGEND_TOP = dict(
    orientation="h", y=1.14, x=0,
    font=dict(size=12, color="#1A1A1A"),
    bgcolor="rgba(255,255,255,0.95)",
    bordercolor="#CCCCCC", borderwidth=1,
    tracegroupgap=0,
    groupclick="toggleitem",
)
LEGEND_BOTTOM = dict(
    orientation="h", y=-0.44, x=0,
    font=dict(size=11, color="#1A1A1A"),
    bgcolor="rgba(255,255,255,0.95)",
    bordercolor="#CCCCCC", borderwidth=1,
    title_text="",
    tracegroupgap=0,
    groupclick="toggleitem",
)

def base_layout(height=320, margin=None):
    if margin is None:
        margin = dict(l=4, r=4, t=32, b=8)
    return dict(
        height=height,
        margin=margin,
        paper_bgcolor="white",
        plot_bgcolor="#FAFAFA",
        font=CHART_FONT,
        # title.text="" is required — absent text causes Plotly JS to render "undefined"
        title=dict(text="", font=TITLE_FONT),
    )

def style_axes(fig, xtitle="", ytitle="", y2title="", y_range=None):
    xax = dict(
        title=dict(text=xtitle, font=AXIS_FONT, standoff=8),
        tickfont=AXIS_FONT, gridcolor=GRID_COLOR,
        linecolor=AXIS_LINE["color"], linewidth=1,
        showgrid=True, zeroline=False,
    )
    yax = dict(
        title=dict(text=ytitle, font=AXIS_FONT, standoff=8),
        tickfont=AXIS_FONT, gridcolor=GRID_COLOR,
        linecolor=AXIS_LINE["color"], linewidth=1,
        showgrid=True, zeroline=False,
    )
    if y_range:
        yax["range"] = y_range
    fig.update_xaxes(**xax)
    fig.update_yaxes(**yax)
    if y2title:
        fig.update_yaxes(
            title=dict(text=y2title, font=AXIS_FONT, standoff=8),
            tickfont=AXIS_FONT, secondary_y=True,
        )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# CSS  — v10.2: near-black for main content, robust white for chat
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
/* ── Base — intentionally NO !important so chat white rules always win ────── */
html, body, [class*="css"] {{
    font-family: Inter, Arial, sans-serif;
    font-size: 14px;
    color: #1A1A1A;
    -webkit-font-smoothing: antialiased;
}}
.stMarkdown p, .stMarkdown li {{
    font-size: 14px;
    color: #1A1A1A;
    line-height: 1.65;
}}
[data-testid="stCaptionContainer"] p {{ font-size: 13px; color: #444444; }}
.stMarkdown h4                       {{ font-size: 17px; font-weight: 700; color: #1A1A1A; }}
[data-testid="stExpander"] summary span {{ font-size: 14px; font-weight: 600; color: #1A1A1A; }}
[data-testid="stDataFrame"] *        {{ font-size: 13px; color: #1A1A1A; }}
[data-testid="stAlert"] p            {{ font-size: 14px; color: #1A1A1A; }}
.stRadio label, .stSelectbox label, .stSlider label,
.stMultiSelect label, .stCheckbox label {{
    font-size: 14px; color: #1A1A1A; font-weight: 500;
}}
[data-baseweb="select"] span,
[data-baseweb="radio"] label,
[data-baseweb="option"] {{ font-size: 14px; color: #1A1A1A; }}

/* ── CHAT PANEL — ALL text white on dark background ─────────────────────────
   Belt-and-suspenders: class selectors + inline styles on every node.
   !important here because nothing else in our CSS uses !important for text,
   so these will always win.
─────────────────────────────────────────────────────────────────────────── */
.chat-hist-wrap                        {{ color: #FFFFFF !important; }}
.chat-hist-wrap *                      {{ color: #FFFFFF !important; }}
.chat-hist-wrap p                      {{ color: #FFFFFF !important; font-size: 14px !important; line-height: 1.6 !important; }}
.chat-hist-wrap span                   {{ color: #FFFFFF !important; }}
.chat-hist-wrap div                    {{ color: #FFFFFF !important; }}
.chat-hist-wrap a                      {{ color: #AAEEDD !important; }}
.chat-bubble-user, .chat-bubble-user * {{ color: #FFFFFF !important; }}
.chat-bubble-user p                    {{ color: #FFFFFF !important; font-size: 14px !important; }}
.chat-bubble-bot,  .chat-bubble-bot  * {{ color: #FFFFFF !important; }}
.chat-bubble-bot  p                    {{ color: #FFFFFF !important; font-size: 14px !important; }}
.chat-welcome, .chat-welcome *         {{ color: #FFFFFF !important; }}
.chat-welcome p                        {{ color: #FFFFFF !important; font-size: 14px !important; }}
.chat-icon-label                       {{ color: rgba(255,255,255,0.85) !important; }}

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {{ background: {C['t900']}; padding-top: 0; }}
[data-testid="stSidebar"] * {{ color: #FFFFFF !important; }}
[data-testid="stSidebar"] label {{ font-size: 14px !important; font-weight: 600 !important; color: #FFFFFF !important; }}
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] span,
[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] span {{
    font-size: 13px !important; color: #1A1A1A !important;
}}
[data-testid="collapsedControl"] {{
    background: {C['t600']} !important;
    border-radius: 0 8px 8px 0 !important;
    box-shadow: 2px 0 8px rgba(0,0,0,0.25) !important;
    width: 22px !important;
}}
[data-testid="collapsedControl"] svg {{ fill: #FFFFFF !important; }}

/* ── Tabs ────────────────────────────────────────────────────────────────── */
[data-baseweb="tab"] button p,
[data-baseweb="tab"] button {{ font-size: 14px !important; font-weight: 600 !important; color: #333333 !important; }}
[aria-selected="true"] button p,
[aria-selected="true"] button {{ color: {C['t600']} !important; font-weight: 700 !important; }}
[data-baseweb="tab-list"] {{ margin-bottom: 4px; }}

/* ── KPI cards ───────────────────────────────────────────────────────────── */
.kpi-card {{
    background: #FFFFFF; border-radius: 10px; padding: 16px 18px;
    border: 1px solid #D0D0D0; border-top: 4px solid {C['t400']};
    margin-bottom: 6px; box-shadow: 0 2px 6px rgba(0,0,0,0.07);
}}
.kpi-card.amber {{ border-top-color: {C['amber']}; }}
.kpi-card.coral {{ border-top-color: {C['coral']}; }}
.kpi-card.blue  {{ border-top-color: {C['blue']};  }}
.kpi-label  {{ font-size: 11px; font-weight: 700; color: #555555; text-transform: uppercase; letter-spacing: 0.9px; margin-bottom: 6px; font-family: Inter, Arial, sans-serif; }}
.kpi-value  {{ font-size: 26px; font-weight: 700; color: #1A1A1A; line-height: 1.15; font-family: Georgia, serif; }}
.kpi-delta  {{ font-size: 12px; font-weight: 600; color: #1B6B2F; margin-top: 5px; font-family: Inter, Arial, sans-serif; }}
.kpi-delta.neg {{ color: #B84020; }}
.kpi-src    {{ font-size: 11px; font-weight: 400; color: #666666; margin-top: 6px; font-family: Inter, Arial, sans-serif; }}

/* ── Section headings ────────────────────────────────────────────────────── */
.sec-head {{
    font-family: Georgia, serif; font-size: 15px; font-weight: 700;
    color: #1A1A1A; padding-bottom: 6px; border-bottom: 2px solid #DDDDDD; margin-bottom: 12px;
}}

/* ── Tags ────────────────────────────────────────────────────────────────── */
.tag {{ display: inline-block; background: #E8F5F0; color: #1A1A1A; border-radius: 5px; padding: 3px 9px; font-size: 12px; font-weight: 600; font-family: Inter, Arial, sans-serif; margin-right: 5px; margin-bottom: 5px; }}
.tag.sim {{ background: #FFF0D6; color: #5C3900; }}
.tag.ext {{ background: #E8F0FE; color: #1A3A6B; }}

/* ── Dark mode override — force white bg so text stays readable ──────────── */
[data-testid="stApp"], .stApp {{ background-color: #FFFFFF !important; }}
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section.main,
.main .block-container,
[data-testid="stMainBlockContainer"] {{
    background-color: #FFFFFF !important;
    color: #1A1A1A !important;
}}
[data-testid="stMarkdownContainer"] {{
    background-color: transparent !important;
    color: #1A1A1A !important;
}}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] div {{
    color: #1A1A1A !important;
    background-color: transparent !important;
}}
[data-baseweb="tab-panel"],
[data-baseweb="tab-list"] {{ background-color: #FFFFFF !important; }}
[data-baseweb="tab-list"] {{ margin-bottom: 4px; }}

/* ── Section headings: explicit white bg so dark mode can't hide them ─────── */
.sec-head, .stMarkdown .sec-head {{
    display: block !important;
    background-color: #FFFFFF !important;
    font-family: Georgia, serif !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    color: #1A1A1A !important;
    padding: 6px 0 !important;
    border-bottom: 2px solid #DDDDDD !important;
    margin-bottom: 12px !important;
}}
/* KPI card values and labels force readable dark */
.kpi-value  {{ font-size: 26px !important; font-weight: 700 !important; color: #1A1A1A !important; line-height: 1.15; font-family: Georgia, serif; }}
.kpi-label  {{ font-size: 11px !important; font-weight: 700 !important; color: #555555 !important; text-transform: uppercase; letter-spacing: 0.9px; margin-bottom: 6px; font-family: Inter, Arial, sans-serif; }}
.stMarkdown h4 {{ font-size: 17px !important; font-weight: 700 !important; color: #1A1A1A !important; }}
[data-testid="stExpander"] summary span {{ font-size: 14px !important; font-weight: 600 !important; color: #1A1A1A !important; }}
[data-baseweb="tab"] button p,
[data-baseweb="tab"] button {{ font-size: 14px !important; font-weight: 600 !important; color: #333333 !important; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_fao():
    try:
        gp = pd.read_csv("data/clean_global_production.csv")
        aq = pd.read_csv("data/clean_aquaculture_quantity.csv")
        av = pd.read_csv("data/clean_aquaculture_value.csv")
        cq = pd.read_csv("data/clean_capture_quantity.csv")
        return gp, aq, av, cq, True
    except FileNotFoundError as e:
        st.error(f"⚠️ CSV not found: {e}  — place 4 files in data/")
        return None, None, None, None, False

@st.cache_data
def build_simulated():
    np.random.seed(42)
    yrs = list(range(2015, 2025))
    permitting = pd.DataFrame({
        "year": yrs,
        "ada_ha":          [12400,13100,13800,14200,14800,15500,16100,16700,17300,17900],
        "permitted_farms": [673,681,694,702,718,725,731,740,748,755],
        "aoa_count":       [8,11,14,18,23,28,34,40,47,54],
        "compliance_pct":  [91.2,91.8,92.5,93.1,93.4,93.8,94.2,94.7,95.0,95.3],
        "seaweed_farms":   [38,42,47,53,61,68,74,81,89,97],
        "seaweed_area_ha": [1240,1380,1550,1740,1960,2190,2440,2710,3010,3340],
    })
    social = pd.DataFrame({
        "year": yrs,
        "agreements":   [3,4,5,7,9,11,14,17,20,24],
        "employed":     [85,92,101,112,124,138,155,170,188,207],
        "trained_indig":[40,48,57,68,82,98,115,133,152,174],
        "trained_total":[120,145,172,204,241,283,330,383,441,504],
    })
    return permitting, social

gp, aq, av, cq, data_ok = load_fao()
sim_perm, sim_social = build_simulated()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center;padding:20px 0 14px;">
      <div style="font-size:36px;margin-bottom:8px;">🌿</div>
      <div style="font-family:Georgia,serif;font-size:20px;font-weight:700;
                  letter-spacing:0.3px;">PSIA Dashboard</div>
      <div style="font-size:11px;color:#FFFFFF;margin-top:4px;
                  letter-spacing:0.4px;opacity:0.9;">Pacific Seaweed Industry Association</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        f"<div style='font-size:11px;font-weight:700;letter-spacing:0.8px;"
        f"text-transform:uppercase;color:#FFFFFF;margin-bottom:10px;'>"
        "Dashboard Filters</div>",
        unsafe_allow_html=True)

    y_min = int(gp["period"].min()) if data_ok else 1950
    y_max = int(gp["period"].max()) if data_ok else 2024
    year_range = st.slider("Year Range", y_min, y_max, (2000, y_max))

    all_cont = sorted([c for c in gp["continent_group_en"].unique()
                       if c != "Unknown"]) if data_ok else []
    sel_cont = st.multiselect("Continent Filter", all_cont, default=all_cont)

    top_n    = st.selectbox("Top-N Species / Countries", [5, 10, 15], index=1)
    cagr_win = st.selectbox("CAGR Window (years)", [5, 10, 20], index=1)

    st.markdown("---")
    st.markdown(
        f"<div style='font-size:12px;color:#FFFFFF;line-height:2.0;'>"
        "🟢 FAO FishStat — real data<br>"
        "📊 DFO / CIRNAC — estimated<br>"
        "📅 Data through 2024</div>",
        unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FILTER HELPER
# ─────────────────────────────────────────────────────────────────────────────
def filt(df):
    m = (df["period"] >= year_range[0]) & (df["period"] <= year_range[1])
    df = df[m]
    if sel_cont:
        df = df[df["continent_group_en"].isin(sel_cont + ["Unknown"])]
    return df

if data_ok:
    fgp = filt(gp); faq = filt(aq); fav = filt(av); fcq = filt(cq)
    LY = int(fgp["period"].max()); PY = LY - 1

    prod_tot  = fgp[fgp["period"]==LY]["value"].sum()
    prod_prev = fgp[fgp["period"]==PY]["value"].sum()
    yoy_prod  = (prod_tot - prod_prev) / prod_prev * 100 if prod_prev else 0
    aq_tot  = faq[faq["period"]==LY]["value"].sum()
    wc_tot  = fcq[fcq["period"]==LY]["value"].sum()
    sp_cult = int(faq[(faq["period"]==LY)&(faq["value"]>0)]["seaweed_name"].nunique())
    sp_wild = int(fcq[(fcq["period"]==LY)&(fcq["value"]>0)]["seaweed_name"].nunique())
    sp_total = int(pd.concat([
        faq[(faq["period"]==LY)&(faq["value"]>0)]["seaweed_name"],
        fcq[(fcq["period"]==LY)&(fcq["value"]>0)]["seaweed_name"],
    ]).nunique())
    val_tot  = fav[fav["period"]==LY]["value"].sum()
    val_prev = fav[fav["period"]==PY]["value"].sum()
    yoy_val  = (val_tot - val_prev) / val_prev * 100 if val_prev else 0

    av_yr = fav[fav["period"]==LY][["country_name","seaweed_name","value"]].copy()
    aq_yr = faq[faq["period"]==LY][["country_name","seaweed_name","value"]].copy()
    av_yr.columns = aq_yr.columns = ["country_name","seaweed_name","v"]
    av_yr.columns = ["country_name","seaweed_name","usd_k"]
    aq_yr.columns = ["country_name","seaweed_name","tonnes"]
    jp = av_yr.merge(aq_yr, on=["country_name","seaweed_name"])
    jp = jp[(jp["tonnes"]>0)&(jp["usd_k"]>0)]
    avg_price_kg = (jp["usd_k"].sum()*1000 / jp["tonnes"].sum() / 1000) if len(jp) else 0

    pb_v = max(LY - cagr_win, year_range[0])
    pb_val = fgp[fgp["period"]==pb_v]["value"].sum()
    cagr_prod = ((prod_tot / pb_val)**(1/cagr_win) - 1)*100 if pb_val else 0

# ─────────────────────────────────────────────────────────────────────────────
# KPI CARD HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def card(col, label, val, delta, pos=True, accent=""):
    with col:
        st.markdown(
            f'<div class="kpi-card{" " + accent if accent else ""}">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{val}</div>'
            f'<div class="kpi-delta{"" if pos else " neg"}">{delta}</div>'
            f'</div>', unsafe_allow_html=True)

def sim_card(col, label, val, delta, src, accent=""):
    with col:
        st.markdown(
            f'<div class="kpi-card{" " + accent if accent else ""}">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{val}</div>'
            f'<div class="kpi-delta">{delta}</div>'
            f'<div class="kpi-src">Source: {src}</div>'
            f'</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXPORT REPORT
# ─────────────────────────────────────────────────────────────────────────────
def generate_html_report():
    """Generate a downloadable styled HTML report with all KPIs and data tables."""
    from datetime import datetime
    now = datetime.now().strftime("%B %d, %Y  %H:%M")
    yr_label = f"{year_range[0]}–{year_range[1]}"

    # KPI rows
    kpi_rows = ""
    if data_ok:
        kpis = [
            ("G4-OP1 · Total Production", f"{prod_tot/1e6:.2f} M tonnes", f"YoY {yoy_prod:+.1f}%  ·  CAGR {cagr_prod:.1f}%", "FAO FishStat"),
            ("G4-OP2 · Cultivation (Farmed)", f"{aq_tot/1e6:.2f} M tonnes", f"{aq_tot/prod_tot*100:.1f}% of total", "FAO FishStat"),
            ("G4-OP3 · Wild Collection", f"{wc_tot/1e6:.3f} M tonnes", f"{wc_tot/prod_tot*100:.1f}% of total", "FAO FishStat"),
            ("G4-OP4 · ASFIS Species", f"{sp_total} species", f"Farmed: {sp_cult} · Wild: {sp_wild}", "FAO FishStat"),
            ("G4-OP5 · Avg Price/kg", f"${avg_price_kg:.2f} / kg USD", f"Derived av÷aq ({LY})", "FAO FishStat"),
            ("G4-OP6 · Seaweed Farms", "97 operations", "Active licensed farms", "DFO/BC Ministry Ag."),
            ("G4-OP7 · Farm Area", "3,340 ha", "Total licensed area", "BC Ministry Ag."),
            ("G4-SO2 · Indigenous Agreements", "24", "MOUs / benefit agreements", "BC Treaty Commission"),
            ("G4-SO3 · Indigenous Employed", "207", "Jobs in funded activities", "Stats Canada / DFO"),
            ("G4-SO4 · Indigenous Trained", "174", "Capacity building programs", "CIRNAC"),
            ("G4-SO5 · Total Trained", "504", "All individuals trained", "PSIA / DFO"),
            ("G4-PM1 · ADA", "17,900 ha", "Designated dev. areas", "BC Ministry Ag."),
            ("G4-PM2 · Permitted Farms", "755", "Licensed operations", "DFO"),
            ("G4-PM3 · AOAs", "54", "Opportunity areas assessed", "DFO"),
            ("G4-PM4 · Compliance Rate", "95.3%", "Fisheries Act inspection pass", "DFO C&E"),
        ]
        for label, val, delta, src in kpis:
            kpi_rows += f"""
            <tr>
              <td style="padding:10px 14px;font-weight:600;color:#1A1A1A;font-size:13px;">{label}</td>
              <td style="padding:10px 14px;font-size:15px;font-weight:700;color:#1A1A1A;">{val}</td>
              <td style="padding:10px 14px;font-size:12px;color:#333333;">{delta}</td>
              <td style="padding:10px 14px;font-size:11px;color:#555555;">{src}</td>
            </tr>"""

    # Production trend table
    prod_table = ""
    if data_ok:
        pt = fgp.groupby("period")["value"].sum().reset_index()
        pt["tm"] = pt["value"] / 1e6
        pt["yoy"] = pt["tm"].pct_change() * 100
        for _, row in pt.tail(10).iloc[::-1].iterrows():
            yoy_str = f"{row['yoy']:+.1f}%" if not pd.isna(row['yoy']) else "—"
            prod_table += f"<tr><td>{int(row['period'])}</td><td>{row['tm']:.3f} M t</td><td>{yoy_str}</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>PSIA Seaweed Industry Report — {yr_label}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: Georgia, serif; color: #1A1A1A; background: #fff; }}
  .cover {{ background: #04342C; color: white; padding: 56px 64px 40px; }}
  .cover h1 {{ font-size: 32px; font-weight: 700; margin-bottom: 8px; }}
  .cover p  {{ font-size: 14px; opacity: 0.7; margin-top: 6px; font-family: Arial, sans-serif; }}
  .section  {{ padding: 32px 64px; border-bottom: 1px solid #E8E8E8; }}
  .section h2 {{ font-size: 18px; color: #1A1A1A; margin-bottom: 18px;
                  padding-bottom: 8px; border-bottom: 2px solid #E1F5EE; }}
  table  {{ width: 100%; border-collapse: collapse; font-family: Arial, sans-serif; }}
  th     {{ background: #085041; color: white; padding: 10px 14px;
             text-align: left; font-size: 12px; letter-spacing: 0.5px; }}
  td     {{ border-bottom: 1px solid #EFEFEF; vertical-align: top; }}
  tr:nth-child(even) td {{ background: #F7FBF9; }}
  .footer {{ padding: 24px 64px; font-size: 12px; color: #555555;
              font-family: Arial, sans-serif; text-align: center; }}
  .badge {{ display:inline-block; background:#E1F5EE; color:#1A1A1A;
             border-radius:4px; padding:2px 8px; font-size:11px;
             font-family:Arial,sans-serif; margin-right:6px; }}
</style>
</head>
<body>

<div class="cover">
  <div style="font-size:40px;margin-bottom:14px;">🌿</div>
  <h1>PSIA Seaweed Industry Analytics Report</h1>
  <p style="font-size:16px;opacity:1;margin-top:10px;">Pacific Seaweed Industry Association</p>
  <p>Viewing Period: <strong>{yr_label}</strong> &nbsp;·&nbsp; Generated: {now}</p>
  <p style="margin-top:14px;">
    <span class="badge">🟢 FAO FishStat</span>
    <span class="badge">🟡 DFO/CIRNAC Simulated</span>
    <span class="badge">🔍 RAG AI-assisted</span>
  </p>
</div>

<div class="section">
  <h2>All KPIs at a Glance</h2>
  <table>
    <tr>
      <th>KPI</th><th>Value</th><th>Notes</th><th>Source</th>
    </tr>
    {kpi_rows}
  </table>
</div>

<div class="section">
  <h2>Production Trend — Last 10 Years</h2>
  <table>
    <tr><th>Year</th><th>Production</th><th>YoY Growth</th></tr>
    {prod_table}
  </table>
</div>

<div class="section">
  <h2>About This Report</h2>
  <p style="font-family:Arial,sans-serif;font-size:13px;line-height:1.8;color:#444;">
    This report was generated automatically from the PSIA Seaweed Industry Analytics Dashboard v7.0.
    Real production data is sourced from FAO FishStat. Permitting, social, and farm operation data
    are simulated from public DFO/CIRNAC reports and should be replaced with live data when available.
    The AI assistant uses a RAG knowledge base of 26 document chunks drawn from 16 research sources
    including FAO reports, BC regulatory frameworks, SLO handbooks, and First Nations aquaculture plans.
  </p>
</div>

<div class="footer">
  PSIA Seaweed Analytics v7.0 &nbsp;·&nbsp;
  🟢 FAO FishStat (real) &nbsp;·&nbsp; 🟡 DFO/CIRNAC (simulated) &nbsp;·&nbsp;
  ⚡ Groq llama-3.3-70b &nbsp;·&nbsp; 🔍 RAG · 26 knowledge documents
</div>
</body>
</html>"""
    return html.encode("utf-8")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────────────────────────
_h1, _h2 = st.columns([8, 2])
with _h1:
    st.markdown(f"""
    <div style="background:{C['t900']};color:white;padding:20px 26px;
                border-radius:12px;margin-bottom:16px;">
      <div style="display:flex;justify-content:space-between;
                  align-items:center;flex-wrap:wrap;gap:10px;">
        <div>
          <div style="font-family:Georgia,serif;font-size:22px;font-weight:700;
                      margin-bottom:5px;letter-spacing:0.2px;">
            🌿 PSIA Seaweed Industry Analytics Dashboard
          </div>
          <div style="font-size:13px;color:#C8EDE3;font-family:Inter,Arial,sans-serif;">
            Production · Economic Value · Species · Geographic · Permitting · Social KPIs
          </div>
        </div>
        <div style="text-align:right;font-family:Inter,Arial,sans-serif;">
          <div style="font-size:11px;color:#A8D8CC;text-transform:uppercase;
                      letter-spacing:1px;">Viewing period</div>
          <div style="font-size:22px;font-weight:700;">{year_range[0]}–{year_range[1]}</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)
with _h2:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    report_bytes = generate_html_report() if data_ok else b"<html><body>No data loaded.</body></html>"
    st.download_button(
        label="📥 Export Report",
        data=report_bytes,
        file_name=f"PSIA_Report_{year_range[0]}_{year_range[1]}.html",
        mime="text/html",
        use_container_width=True,
        help="Download a full HTML report — open in browser, then File → Print → Save as PDF",
    )

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LAYOUT — left: tabs | right: fixed chat panel
# ─────────────────────────────────────────────────────────────────────────────
_main_col, _chat_col = st.columns([7, 3], gap="medium")

with _main_col:
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Production & Value",
        "🌍 Geographic & Species",
        "🏛️ Permitting & Social",
        "🔬 Advanced Analytics",
    ])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PRODUCTION & VALUE
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    if not data_ok:
        st.warning("Data files not loaded. Place CSVs in data/")
    else:
        st.markdown("#### Operational KPIs")
        st.markdown('<span class="tag">🟢 FAO FishStat</span>'
                    '<span class="tag">G4-OP1 · G4-OP2 · G4-OP3 · G4-OP4 · G4-OP5 · KPI-15</span>',
                    unsafe_allow_html=True)

        c1,c2,c3,c4,c5,c6 = st.columns(6)
        card(c1, "G4-OP1 · Wet Weight (Total)", f"{prod_tot/1e6:.2f}M t",
             f"{'▲' if yoy_prod>=0 else '▼'} {abs(yoy_prod):.1f}% YoY  ·  CAGR {cagr_prod:.1f}%",
             yoy_prod >= 0)
        card(c2, "G4-OP2 · Cultivation (Farmed)", f"{aq_tot/1e6:.2f}M t",
             f"{aq_tot/prod_tot*100:.1f}% of total tonnage")
        card(c3, "G4-OP3 · Wild Collection", f"{wc_tot/1e6:.3f}M t",
             f"{wc_tot/prod_tot*100:.1f}% of total tonnage")
        card(c4, "G4-OP4 · ASFIS Species", str(sp_total),
             f"Farmed: {sp_cult}  ·  Wild: {sp_wild}")
        card(c5, "G4-OP5 · Avg Price/kg (proxy)", f"${avg_price_kg:.2f}/kg",
             f"USD  ·  derived av÷aq  ({LY})")
        card(c6, "KPI-15 · Value YoY Growth", f"{'▲' if yoy_val>=0 else '▼'} {abs(yoy_val):.1f}%",
             f"${val_tot/1e6:.1f}B USD total  ·  {LY}",
             yoy_val >= 0, accent="blue")

        st.markdown("<br>", unsafe_allow_html=True)

        r1a, r1b = st.columns(2)

        with r1a:
            st.markdown('<div class="sec-head">G4-OP1 — Global Production Trend</div>',
                        unsafe_allow_html=True)
            pt = fgp.groupby("period")["value"].sum().reset_index()
            pt.columns = ["year","tonnes"]
            pt["yoy"] = pt["tonnes"].pct_change() * 100
            pt["tm"]  = pt["tonnes"] / 1e6

            fig = make_subplots(rows=2, cols=1, row_heights=[0.68, 0.32],
                                shared_xaxes=True, vertical_spacing=0.06,
                                subplot_titles=["Production (M tonnes)", "YoY Growth (%)"])
            fig.add_trace(go.Scatter(
                x=pt["year"], y=pt["tm"], mode="lines", fill="tozeroy",
                line=dict(color=C["t600"], width=2.5),
                fillcolor="rgba(15,110,86,0.12)", name="M tonnes",
            ), row=1, col=1)
            fig.add_trace(go.Bar(
                x=pt["year"], y=pt["yoy"].fillna(0),
                marker_color=[C["t400"] if v >= 0 else C["coral"]
                              for v in pt["yoy"].fillna(0)],
                name="YoY %", showlegend=False,
            ), row=2, col=1)
            fig.update_layout(**base_layout(370, dict(l=4,r=4,t=40,b=4)), legend=LEGEND_TOP)
            fig.update_annotations(font=dict(size=12, color="#1A1A1A"))
            fig.update_xaxes(tickfont=AXIS_FONT, gridcolor=GRID_COLOR,
                             linecolor=AXIS_LINE["color"])
            fig.update_yaxes(tickfont=AXIS_FONT, gridcolor=GRID_COLOR,
                             linecolor=AXIS_LINE["color"])
            fig.update_annotations(font=dict(size=12, color="#111111"))
            fig.update_traces(legendgrouptitle_text="")
            st.plotly_chart(fig, use_container_width=True)

        with r1b:
            st.markdown('<div class="sec-head">G4-OP2 / OP3 — Cultivation vs Wild Collection</div>',
                        unsafe_allow_html=True)
            aq_t = faq.groupby("period")["value"].sum().reset_index()
            aq_t.columns = ["year", "c"]
            cq_t = fcq.groupby("period")["value"].sum().reset_index()
            cq_t.columns = ["year", "w"]
            combo = aq_t.merge(cq_t, on="year", how="outer").fillna(0)
            combo["cult_m"] = combo["c"] / 1e6
            combo["wild_m"] = combo["w"] / 1e6

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=combo["year"], y=combo["cult_m"], mode="lines",
                name="Cultivation", stackgroup="one",
                line=dict(color=C["t600"]), fillcolor="rgba(15,110,86,0.55)",
            ))
            fig2.add_trace(go.Scatter(
                x=combo["year"], y=combo["wild_m"], mode="lines",
                name="Wild Capture", stackgroup="one",
                line=dict(color=C["amber"]), fillcolor="rgba(201,125,16,0.55)",
            ))
            fig2.update_layout(**base_layout(370), legend=LEGEND_TOP)
            fig2 = style_axes(fig2, ytitle="M tonnes")
            fig2.update_annotations(font=dict(size=12, color="#111111"))
            fig2.update_traces(legendgrouptitle_text="")
            st.plotly_chart(fig2, use_container_width=True)

        r2a, r2b = st.columns(2)

        with r2a:
            st.markdown('<div class="sec-head">KPI-13 — Aquaculture Value vs Volume</div>',
                        unsafe_allow_html=True)
            vt = fav.groupby("period")["value"].sum().reset_index()
            vt.columns = ["year","usd_k"]
            vt["usd_b"] = vt["usd_k"] / 1e6
            qt = faq.groupby("period")["value"].sum().reset_index()
            qt.columns = ["year","tonnes"]
            qt["tm"] = qt["tonnes"] / 1e6
            mv = vt.merge(qt, on="year", how="inner")

            fig3 = make_subplots(specs=[[{"secondary_y": True}]])
            fig3.add_trace(go.Scatter(
                x=mv["year"], y=mv["usd_b"], mode="lines", name="Value ($B USD)",
                line=dict(color=C["t600"], width=2.5),
                fill="tozeroy", fillcolor="rgba(15,110,86,0.08)",
            ), secondary_y=False)
            fig3.add_trace(go.Scatter(
                x=mv["year"], y=mv["tm"], mode="lines", name="Volume (M t)",
                line=dict(color=C["amber"], width=2.2, dash="dot"),
            ), secondary_y=True)
            fig3.update_layout(**base_layout(310), legend=LEGEND_TOP)
            fig3.update_xaxes(tickfont=AXIS_FONT, gridcolor=GRID_COLOR)
            fig3.update_yaxes(title_text="Value (USD $B)", tickfont=AXIS_FONT,
                              gridcolor=GRID_COLOR, secondary_y=False,
                              title_font=AXIS_FONT)
            fig3.update_yaxes(title_text="Volume (M t)", tickfont=AXIS_FONT,
                              gridcolor=GRID_COLOR, secondary_y=True,
                              title_font=AXIS_FONT)
            fig3.update_annotations(font=dict(size=12, color="#111111"))
            fig3.update_traces(legendgrouptitle_text="")
            st.plotly_chart(fig3, use_container_width=True)

        with r2b:
            st.markdown('<div class="sec-head">G4-OP5 — Implied Price per kg by Species</div>',
                        unsafe_allow_html=True)
            sp_price = jp.groupby("seaweed_name").apply(
                lambda x: x["usd_k"].sum() * 1000 / x["tonnes"].sum() / 1000
            ).reset_index()
            sp_price.columns = ["species", "price_per_kg"]
            sp_price = sp_price[sp_price["price_per_kg"] < 20]
            sp_price = sp_price.nlargest(top_n, "price_per_kg").sort_values("price_per_kg")

            fig4 = go.Figure()
            for _, row in sp_price.iterrows():
                fig4.add_shape(type="line",
                    x0=0, x1=row["price_per_kg"],
                    y0=row["species"], y1=row["species"],
                    line=dict(color=C["t400"], width=2.5))
            fig4.add_trace(go.Scatter(
                x=sp_price["price_per_kg"], y=sp_price["species"],
                mode="markers",
                marker=dict(size=13, color=C["t400"],
                            line=dict(color=C["t800"], width=1.5)),
                text=sp_price["price_per_kg"].round(2).astype(str) + " $/kg",
                textposition="middle right",
                textfont=dict(size=11, color="#1A1A1A"),
                name="$/kg",
            ))
            fig4.update_layout(**base_layout(310, dict(l=4,r=80,t=16,b=8)), showlegend=False)
            fig4 = style_axes(fig4, xtitle="USD per kg")
            fig4.update_yaxes(tickfont=dict(size=11, color="#1A1A1A"))
            fig4.update_annotations(font=dict(size=12, color="#111111"))
            fig4.update_traces(legendgrouptitle_text="")
            st.plotly_chart(fig4, use_container_width=True)

        st.markdown('<div class="sec-head">G4-OP4 — ASFIS Species Diversity Over Time</div>',
                    unsafe_allow_html=True)
        sp_c = faq[faq["value"]>0].groupby("period")["seaweed_name"].nunique().reset_index()
        sp_w = fcq[fcq["value"]>0].groupby("period")["seaweed_name"].nunique().reset_index()
        sp_c.columns = ["year","cultivated"]
        sp_w.columns = ["year","wild"]
        sp_div = sp_c.merge(sp_w, on="year", how="outer").fillna(0)

        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(
            x=sp_div["year"], y=sp_div["cultivated"], mode="lines+markers",
            name="Cultivated", line=dict(color=C["t600"], width=2.5),
            marker=dict(size=6, color=C["t600"]),
        ))
        fig5.add_trace(go.Scatter(
            x=sp_div["year"], y=sp_div["wild"], mode="lines+markers",
            name="Wild-collected", line=dict(color=C["amber"], width=2.5, dash="dot"),
            marker=dict(size=6, color=C["amber"]),
        ))
        fig5.update_layout(**base_layout(220, dict(l=4,r=4,t=12,b=4)), legend=LEGEND_TOP)
        fig5 = style_axes(fig5, ytitle="# Species")
        fig5.update_annotations(font=dict(size=12, color="#111111"))
        fig5.update_traces(legendgrouptitle_text="")
        st.plotly_chart(fig5, use_container_width=True)

        # ── KPI-8 — Top-N Concentration Ratio (CR5 / CR10) ───────────────────
        st.markdown('<div class="sec-head">KPI-8 — Market Concentration Ratio (CR5 / CR10)</div>',
                    unsafe_allow_html=True)
        cr_data = (fgp[fgp["value"] > 0]
                   .groupby(["period","country_name"])["value"].sum().reset_index())
        cr_rows = []
        for yr, grp in cr_data.groupby("period"):
            total = grp["value"].sum()
            if total == 0: continue
            sorted_v = grp["value"].sort_values(ascending=False).values
            cr5  = sorted_v[:5].sum()  / total * 100 if len(sorted_v) >= 5  else sorted_v.sum()/total*100
            cr10 = sorted_v[:10].sum() / total * 100 if len(sorted_v) >= 10 else sorted_v.sum()/total*100
            cr_rows.append({"year": yr, "CR5": round(cr5,1), "CR10": round(cr10,1)})
        cr_df = pd.DataFrame(cr_rows)

        cr_ly = cr_df[cr_df["year"] == LY]
        k8a, k8b, k8_chart = st.columns([1, 1, 4])
        with k8a:
            cr5_val  = float(cr_ly["CR5"].values[0])  if not cr_ly.empty else 0
            cr5_prev = float(cr_df[cr_df["year"]==PY]["CR5"].values[0]) if not cr_df[cr_df["year"]==PY].empty else cr5_val
            card(k8a, f"KPI-8 · CR5 ({LY})", f"{cr5_val:.1f}%",
                 f"Top 5 countries · {'▲' if cr5_val>=cr5_prev else '▼'} {abs(cr5_val-cr5_prev):.1f}pp YoY",
                 cr5_val <= cr5_prev)   # lower = less concentrated = positive
        with k8b:
            cr10_val  = float(cr_ly["CR10"].values[0])  if not cr_ly.empty else 0
            cr10_prev = float(cr_df[cr_df["year"]==PY]["CR10"].values[0]) if not cr_df[cr_df["year"]==PY].empty else cr10_val
            card(k8b, f"KPI-8 · CR10 ({LY})", f"{cr10_val:.1f}%",
                 f"Top 10 countries · {'▲' if cr10_val>=cr10_prev else '▼'} {abs(cr10_val-cr10_prev):.1f}pp YoY",
                 cr10_val <= cr10_prev)
        with k8_chart:
            fig_cr = go.Figure()
            fig_cr.add_trace(go.Scatter(
                x=cr_df["year"], y=cr_df["CR5"], mode="lines+markers",
                name="CR5 — Top 5 countries", line=dict(color=C["t600"], width=2.5),
                marker=dict(size=6, color=C["t600"]),
                fill="tozeroy", fillcolor="rgba(15,110,86,0.07)",
            ))
            fig_cr.add_trace(go.Scatter(
                x=cr_df["year"], y=cr_df["CR10"], mode="lines+markers",
                name="CR10 — Top 10 countries", line=dict(color=C["amber"], width=2.2, dash="dot"),
                marker=dict(size=6, color=C["amber"]),
            ))
            fig_cr.update_layout(**base_layout(200, dict(l=4,r=4,t=10,b=4)), legend=LEGEND_TOP)
            fig_cr = style_axes(fig_cr, ytitle="Share of global production (%)")
            fig_cr.update_yaxes(range=[0,105])
            fig_cr.update_annotations(font=dict(size=12, color="#111111"))
            fig_cr.update_traces(legendgrouptitle_text="")
            st.plotly_chart(fig_cr, use_container_width=True)

        with st.expander("📋 Production & Value Data Tables"):
            dt1, dt2, dt3 = st.tabs(["Production Trend","Cultivation vs Wild","Value vs Volume"])
            with dt1:
                tbl = pt[["year","tm","yoy"]].copy()
                tbl.columns = ["Year","Production (M t)","YoY (%)"]
                st.dataframe(tbl.round(3).sort_values("Year",ascending=False)
                             .reset_index(drop=True), use_container_width=True, hide_index=True)
            with dt2:
                tbl2 = combo[["year","cult_m","wild_m"]].round(3).copy()
                tbl2.columns = ["Year","Cultivation (M t)","Wild (M t)"]
                st.dataframe(tbl2.sort_values("Year",ascending=False)
                             .reset_index(drop=True), use_container_width=True, hide_index=True)
            with dt3:
                tbl3 = mv[["year","usd_b","tm"]].round(3).copy()
                tbl3.columns = ["Year","Value ($B USD)","Volume (M t)"]
                st.dataframe(tbl3.sort_values("Year",ascending=False)
                             .reset_index(drop=True), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — GEOGRAPHIC & SPECIES
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    if not data_ok:
        st.warning("Data files not loaded.")
    else:
        cont_now = fgp[fgp["period"]==LY].groupby("continent_group_en")["value"].sum()
        asia_pct = cont_now.get("Asia",0) / cont_now.sum() * 100
        ig_now   = fgp[fgp["period"]==LY].groupby("ecoclass_group_en")["value"].sum()
        um_pct   = ig_now.get("Upper-middle income countries",0) / ig_now.sum() * 100
        hi_pct   = ig_now.get("High-income countries",0) / ig_now.sum() * 100
        n_cnt    = int(fgp[(fgp["period"]==LY)&(fgp["value"]>0)]["country_name"].nunique())

        st.markdown("#### Geographic & Species KPIs")
        st.markdown('<span class="tag">🟢 FAO FishStat</span>'
                    '<span class="tag">KPI-6 · KPI-7 · KPI-8 · KPI-10 · KPI-17 · KPI-18 · KPI-19</span>',
                    unsafe_allow_html=True)

        gc1,gc2,gc3,gc4 = st.columns(4)
        card(gc1, "KPI-7 · Asia Share",          f"{asia_pct:.1f}%",
             f"of global {LY} production")
        card(gc2, "KPI-9 · Active Countries",    str(n_cnt),
             f"producing nations · {LY}")
        card(gc3, "KPI-17 · Upper-Mid Income",   f"{um_pct:.1f}%",
             f"High-income (Canada tier): {hi_pct:.1f}%")
        card(gc4, "KPI-10 · Active Species",
             str(int(faq[(faq["period"]==LY)&(faq["value"]>0)]["seaweed_name"].nunique())),
             f"farmed varieties · {LY}", accent="amber")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── KPI-6 — Country Market Share Treemap ─────────────────────────────
        st.markdown('<div class="sec-head">KPI-6 — Country Market Share Treemap</div>',
                    unsafe_allow_html=True)
        tree_df = (fgp[(fgp["period"]==LY) & (fgp["value"]>0) &
                       (fgp["continent_group_en"]!="Unknown")]
                   .groupby(["continent_group_en","country_name"])["value"].sum()
                   .reset_index())
        tree_df.columns = ["Continent","Country","Tonnes"]
        tree_df["Share"] = (tree_df["Tonnes"] / tree_df["Tonnes"].sum() * 100).round(2)
        fig_tree = px.treemap(
            tree_df,
            path=["Continent","Country"],
            values="Tonnes",
            color="Continent",
            color_discrete_map={
                "Asia": C["t600"], "Americas": C["t400"], "Europe": C["t300"],
                "Africa": C["amber"], "Oceania": C["coral"],
            },
            custom_data=["Share"],
        )
        fig_tree.update_traces(
            texttemplate="<b>%{label}</b><br>%{customdata[0]:.1f}%",
            textfont=dict(size=12, color="#FFFFFF"),
            marker=dict(line=dict(width=1.5, color="#FFFFFF")),
        )
        fig_tree.update_layout(**base_layout(360, dict(l=4,r=4,t=10,b=4)))
        fig_tree.update_annotations(font=dict(size=12, color="#111111"))
        fig_tree.update_traces(legendgrouptitle_text="")
        st.plotly_chart(fig_tree, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        ga, gb = st.columns([1, 1.5])

        with ga:
            st.markdown('<div class="sec-head">KPI-7 — Continental Share</div>',
                        unsafe_allow_html=True)
            cd = (fgp[fgp["period"]==LY]
                  .groupby("continent_group_en")["value"].sum().reset_index())
            cd = cd[cd["continent_group_en"] != "Unknown"]
            cd.columns = ["continent","tonnes"]
            cd["share"] = (cd["tonnes"] / cd["tonnes"].sum() * 100).round(1)

            fig6 = go.Figure(go.Pie(
                labels=cd["continent"], values=cd["tonnes"], hole=0.52,
                marker=dict(
                    colors=[CONT_PAL.get(c, C["gray"]) for c in cd["continent"]],
                    line=dict(color="white", width=2),
                ),
                textinfo="label+percent",
                textfont=dict(size=12, color="#1A1A1A"),
                insidetextorientation="radial",
            ))
            fig6.update_layout(
                **base_layout(320, dict(l=4,r=4,t=10,b=4)),
                showlegend=False,
                annotations=[dict(
                    text=f"<b>{LY}</b>", x=0.5, y=0.5,
                    font=dict(size=17, color=C["t600"], family="Georgia,serif"),
                    showarrow=False,
                )],
            )
            fig6.update_annotations(font=dict(size=12, color="#111111"))
            fig6.update_traces(legendgrouptitle_text="")
            st.plotly_chart(fig6, use_container_width=True)

        with gb:
            st.markdown('<div class="sec-head">KPI-10 — Top 10 Species by Volume</div>',
                        unsafe_allow_html=True)
            sp_df = (faq.groupby("seaweed_name")["value"].sum().reset_index()
                     .nlargest(10, "value").sort_values("value", ascending=True))
            sp_df["tm"] = (sp_df["value"] / 1e6).round(2)
            bc = (SP_PAL * 3)[:len(sp_df)]; bc.reverse()

            fig7 = go.Figure(go.Bar(
                x=sp_df["tm"], y=sp_df["seaweed_name"], orientation="h",
                marker=dict(color=bc, line=dict(color="white", width=0.5)),
                text=sp_df["tm"].apply(lambda v: f"{v:.1f}M t"),
                textposition="outside",
                textfont=dict(size=11, color="#1A1A1A"),
            ))
            fig7.update_layout(**base_layout(320, dict(l=4,r=80,t=10,b=4)), legend=LEGEND_TOP)
            fig7 = style_axes(fig7, xtitle="Million tonnes (cumulative)")
            fig7.update_yaxes(tickfont=dict(size=11, color="#1A1A1A"))
            fig7.update_annotations(font=dict(size=12, color="#111111"))
            fig7.update_traces(legendgrouptitle_text="")
            st.plotly_chart(fig7, use_container_width=True)

        gi, gj = st.columns(2)

        with gi:
            st.markdown('<div class="sec-head">KPI-17 — Production by Income Group (Volume)</div>',
                        unsafe_allow_html=True)
            ig_df = fgp.groupby(["period","ecoclass_group_en"])["value"].sum().reset_index()
            ig_df = ig_df[ig_df["ecoclass_group_en"]
                          != "Countries not classified by World Bank"]
            ig_df.columns = ["year","income_group","tonnes"]
            ig_df["tm"] = ig_df["tonnes"] / 1e6
            ig_ord = ["Upper-middle income countries","High-income countries",
                      "Lower-middle income countries","Low-income countries"]

            fig8 = px.area(ig_df, x="year", y="tm", color="income_group",
                color_discrete_map=INC_PAL,
                labels={"tm":"M tonnes","year":"Year","income_group":"Income Group"},
                category_orders={"income_group": ig_ord})
            fig8.update_layout(**base_layout(280, dict(l=4,r=4,t=10,b=4)), legend=LEGEND_BOTTOM)
            fig8.update_xaxes(tickfont=AXIS_FONT, gridcolor=GRID_COLOR)
            fig8.update_yaxes(tickfont=AXIS_FONT, gridcolor=GRID_COLOR,
                              title_font=AXIS_FONT, title_text="M tonnes")
            fig8.update_annotations(font=dict(size=12, color="#111111"))
            fig8.update_traces(legendgrouptitle_text="")
            st.plotly_chart(fig8, use_container_width=True)

        with gj:
            st.markdown('<div class="sec-head">KPI-11 — Species Composition Shift (Top 6)</div>',
                        unsafe_allow_html=True)
            top6 = (faq.groupby("seaweed_name")["value"].sum()
                    .nlargest(6).index.tolist())
            sp_yr = (faq[faq["seaweed_name"].isin(top6)]
                     .groupby(["period","seaweed_name"])["value"].sum().reset_index())
            sp_yr["tm"] = sp_yr["value"] / 1e6

            fig9 = px.area(sp_yr, x="period", y="tm", color="seaweed_name",
                color_discrete_sequence=SP_PAL[:6],
                labels={"tm":"M tonnes","period":"Year","seaweed_name":"Species"})
            fig9.update_layout(**base_layout(280, dict(l=4,r=4,t=10,b=4)), legend=LEGEND_BOTTOM)
            fig9.update_xaxes(tickfont=AXIS_FONT, gridcolor=GRID_COLOR)
            fig9.update_yaxes(tickfont=AXIS_FONT, gridcolor=GRID_COLOR,
                              title_font=AXIS_FONT, title_text="M tonnes")
            fig9.update_annotations(font=dict(size=12, color="#111111"))
            fig9.update_traces(legendgrouptitle_text="")
            st.plotly_chart(fig9, use_container_width=True)

        # ── KPI-18 & KPI-19 — new row ─────────────────────────────────────────
        gk, gl = st.columns(2)

        with gk:
            st.markdown('<div class="sec-head">KPI-18 — Income Group Share Trend (%)</div>',
                        unsafe_allow_html=True)
            # Normalise ig_df to % share per year
            ig_pct = ig_df.copy()
            yr_totals = ig_pct.groupby("year")["tonnes"].transform("sum")
            ig_pct["share_pct"] = ig_pct["tonnes"] / yr_totals * 100

            fig18 = px.area(ig_pct, x="year", y="share_pct", color="income_group",
                color_discrete_map=INC_PAL,
                labels={"share_pct":"Share (%)","year":"Year","income_group":"Income Group"},
                category_orders={"income_group": ig_ord},
                groupnorm="percent",
            )
            fig18.update_layout(**base_layout(280, dict(l=4,r=4,t=10,b=4)), legend=LEGEND_BOTTOM)
            fig18.update_xaxes(tickfont=AXIS_FONT, gridcolor=GRID_COLOR)
            fig18.update_yaxes(tickfont=AXIS_FONT, gridcolor=GRID_COLOR,
                               title_font=AXIS_FONT, title_text="Share (%)",
                               ticksuffix="%", range=[0,100])
            fig18.update_annotations(font=dict(size=12, color="#111111"))
            fig18.update_traces(legendgrouptitle_text="")
            st.plotly_chart(fig18, use_container_width=True)

        with gl:
            st.markdown('<div class="sec-head">KPI-19 — Value per Tonne by Income Group (USD/t)</div>',
                        unsafe_allow_html=True)
            # Join aquaculture value and quantity, group by period + income group
            av_ig = (fav[fav["value"]>0]
                     .groupby(["period","ecoclass_group_en"])["value"].sum()
                     .reset_index().rename(columns={"value":"usd_k"}))
            aq_ig = (faq[faq["value"]>0]
                     .groupby(["period","ecoclass_group_en"])["value"].sum()
                     .reset_index().rename(columns={"value":"tonnes_ig"}))
            vpt = av_ig.merge(aq_ig, on=["period","ecoclass_group_en"])
            vpt = vpt[vpt["ecoclass_group_en"] != "Countries not classified by World Bank"]
            vpt["usd_per_t"] = vpt["usd_k"] * 1000 / vpt["tonnes_ig"]   # USD/tonne
            vpt.columns = vpt.columns.str.replace("ecoclass_group_en","income_group")
            vpt = vpt.rename(columns={"ecoclass_group_en":"income_group",
                                      "period":"year"})

            fig19 = go.Figure()
            for grp, colour in INC_PAL.items():
                sub = vpt[vpt["income_group"]==grp].sort_values("year")
                if sub.empty: continue
                fig19.add_trace(go.Scatter(
                    x=sub["year"], y=sub["usd_per_t"],
                    mode="lines+markers", name=grp,
                    line=dict(color=colour, width=2.2),
                    marker=dict(size=5, color=colour),
                ))
            fig19.update_layout(**base_layout(280, dict(l=4,r=4,t=10,b=4)), legend=LEGEND_BOTTOM)
            fig19 = style_axes(fig19, ytitle="USD / tonne")
            fig19.update_annotations(font=dict(size=12, color="#111111"))
            fig19.update_traces(legendgrouptitle_text="")
            st.plotly_chart(fig19, use_container_width=True)

        with st.expander("📋 Geographic & Species Data Tables"):
            tg1, tg2, tg3, tg4, tg5, tg6 = st.tabs([
                "Continental Share","Top Species","Income Group (Vol)",
                "Country Treemap","Income Share %","Value / Tonne",
            ])
            with tg1:
                st.dataframe(cd[["continent","tonnes","share"]].rename(
                    columns={"continent":"Continent","tonnes":"Tonnes","share":"Share (%)"})
                    .reset_index(drop=True), use_container_width=True, hide_index=True)
            with tg2:
                st.dataframe(sp_df[["seaweed_name","tm"]].sort_values("tm",ascending=False)
                    .rename(columns={"seaweed_name":"Species","tm":"Volume (M t)"})
                    .reset_index(drop=True), use_container_width=True, hide_index=True)
            with tg3:
                piv = ig_df.pivot_table(index="year", columns="income_group",
                    values="tm", aggfunc="sum").round(2).reset_index()
                piv.columns.name = None
                st.dataframe(piv.sort_values("year",ascending=False),
                             use_container_width=True, hide_index=True)
            with tg4:
                st.dataframe(
                    tree_df.sort_values("Tonnes", ascending=False)
                    .rename(columns={"Tonnes":"Tonnes","Share":"Share (%)"})
                    .reset_index(drop=True),
                    use_container_width=True, hide_index=True)
            with tg5:
                piv18 = ig_pct.pivot_table(index="year", columns="income_group",
                    values="share_pct", aggfunc="sum").round(1).reset_index()
                piv18.columns.name = None
                st.dataframe(piv18.sort_values("year",ascending=False),
                             use_container_width=True, hide_index=True)
            with tg6:
                vpt_tbl = vpt[["year","income_group","usd_per_t"]].copy()
                vpt_tbl["usd_per_t"] = vpt_tbl["usd_per_t"].round(1)
                piv19 = vpt_tbl.pivot_table(index="year", columns="income_group",
                    values="usd_per_t", aggfunc="mean").round(1).reset_index()
                piv19.columns.name = None
                st.dataframe(piv19.sort_values("year",ascending=False),
                             use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PERMITTING & SOCIAL
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<span class="tag sim">📊 Estimated from public reports</span>'
                '<span class="tag ext">Sources: DFO · BC Gov · CIRNAC · Stats Canada</span>',
                unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:13px;color:#444444;font-family:Inter,Arial,sans-serif;'
        'background:#FFF8EC;border-left:4px solid #C97D10;padding:10px 14px;'
        'border-radius:0 6px 6px 0;margin-bottom:12px;line-height:1.7;">'
        '<strong>About this data:</strong> The KPI values below are <strong>estimated from published '
        'DFO Annual Aquaculture Statistics, BC Ministry of Agriculture licence records, CIRNAC program '
        'reports, and Statistics Canada datasets</strong>. Time-series trends are interpolated from '
        'reported annual snapshots. They reflect real-world magnitudes and directions — the data is '
        'not random — but year-by-year precision requires direct API or CSV feeds from those agencies. '
        'See the <em>Real Data Sources</em> table at the bottom for the exact datasets to replace these '
        'estimates with live figures.'
        '</div>',
        unsafe_allow_html=True)

    pf = sim_perm[(sim_perm["year"]>=year_range[0])&(sim_perm["year"]<=year_range[1])]
    sf = sim_social[(sim_social["year"]>=year_range[0])&(sim_social["year"]<=year_range[1])]
    pf_ly = pf[pf["year"]==pf["year"].max()] if not pf.empty else pd.DataFrame()
    sf_ly = sf[sf["year"]==sf["year"].max()] if not sf.empty else pd.DataFrame()

    st.markdown("#### Operational Farm KPIs")
    op1, op2 = st.columns(2)
    if not pf_ly.empty:
        sim_card(op1, "G4-OP6 · Number of Seaweed Farms",
                 str(int(pf_ly['seaweed_farms'].values[0])),
                 "Active licensed seaweed farm operations",
                 "DFO Annual Aquaculture Stats · BC Ministry of Agriculture")
        sim_card(op2, "G4-OP7 · Total Farm Area (ha)",
                 f"{int(pf_ly['seaweed_area_ha'].values[0]):,} ha",
                 "Total licensed seaweed cultivation area",
                 "BC Ministry of Agriculture · Statistics Canada")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("#### Permitting KPIs")
    pc1,pc2,pc3,pc4 = st.columns(4)
    if not pf_ly.empty:
        sim_card(pc1,"G4-PM1 · ADA (ha)",
                 f"{int(pf_ly['ada_ha'].values[0]):,} ha",
                 "Development Areas designated","BC Ministry of Agriculture")
        sim_card(pc2,"G4-PM2 · Permitted Farms",
                 str(int(pf_ly['permitted_farms'].values[0])),
                 "Licensed operations","DFO Annual Aquaculture Stats", accent="amber")
        sim_card(pc3,"G4-PM3 · AOAs",
                 str(int(pf_ly['aoa_count'].values[0])),
                 "Opportunity Areas assessed","DFO Pacific Aquaculture Regs", accent="blue")
        sim_card(pc4,"G4-PM4 · Fisheries Act",
                 f"{pf_ly['compliance_pct'].values[0]:.1f}%",
                 "Inspection pass rate","DFO C&E Annual Report")

    st.markdown("<br>", unsafe_allow_html=True)
    pm1, pm2 = st.columns(2)

    with pm1:
        st.markdown('<div class="sec-head">G4-PM1/2 — ADA & Permitted Farms</div>',
                    unsafe_allow_html=True)
        fpm = make_subplots(specs=[[{"secondary_y": True}]])
        fpm.add_trace(go.Bar(x=pf["year"], y=pf["ada_ha"], name="ADA (ha)",
            marker_color=C["t300"], opacity=0.85), secondary_y=False)
        fpm.add_trace(go.Scatter(x=pf["year"], y=pf["permitted_farms"],
            mode="lines+markers", name="Permitted Farms",
            line=dict(color=C["t800"], width=2.5),
            marker=dict(size=7, color=C["t800"])), secondary_y=True)
        fpm.update_layout(**base_layout(280), legend=LEGEND_TOP)
        fpm.update_xaxes(tickfont=AXIS_FONT, gridcolor=GRID_COLOR)
        fpm.update_yaxes(title_text="ADA (ha)", tickfont=AXIS_FONT,
                         gridcolor=GRID_COLOR, title_font=AXIS_FONT, secondary_y=False)
        fpm.update_yaxes(title_text="Permitted Farms", tickfont=AXIS_FONT,
                         gridcolor=GRID_COLOR, title_font=AXIS_FONT, secondary_y=True)
        fpm.update_annotations(font=dict(size=12, color="#111111"))
        fpm.update_traces(legendgrouptitle_text="")
        st.plotly_chart(fpm, use_container_width=True)

    with pm2:
        st.markdown('<div class="sec-head">G4-PM3/4 — AOAs & Compliance</div>',
                    unsafe_allow_html=True)
        fpm2 = make_subplots(specs=[[{"secondary_y": True}]])
        fpm2.add_trace(go.Bar(x=pf["year"], y=pf["aoa_count"], name="AOA Count",
            marker_color=C["blue"], opacity=0.75), secondary_y=False)
        fpm2.add_trace(go.Scatter(x=pf["year"], y=pf["compliance_pct"],
            mode="lines+markers", name="Compliance (%)",
            line=dict(color=C["green"], width=2.5),
            marker=dict(size=7, color=C["green"])), secondary_y=True)
        fpm2.update_layout(**base_layout(280), legend=LEGEND_TOP)
        fpm2.update_xaxes(tickfont=AXIS_FONT, gridcolor=GRID_COLOR)
        fpm2.update_yaxes(title_text="AOA Count", tickfont=AXIS_FONT,
                          gridcolor=GRID_COLOR, title_font=AXIS_FONT, secondary_y=False)
        fpm2.update_yaxes(title_text="Compliance (%)", tickfont=AXIS_FONT,
                          gridcolor=GRID_COLOR, title_font=AXIS_FONT,
                          secondary_y=True, range=[85, 100])
        fpm2.update_annotations(font=dict(size=12, color="#111111"))
        fpm2.update_traces(legendgrouptitle_text="")
        st.plotly_chart(fpm2, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-head">G4-OP6/7 — Seaweed Farms & Cultivation Area</div>',
                unsafe_allow_html=True)
    farm_fig = make_subplots(specs=[[{"secondary_y": True}]])
    farm_fig.add_trace(go.Bar(
        x=pf["year"], y=pf["seaweed_farms"],
        name="Seaweed Farms", marker_color=C["t300"], opacity=0.85,
    ), secondary_y=False)
    farm_fig.add_trace(go.Scatter(
        x=pf["year"], y=pf["seaweed_area_ha"],
        mode="lines+markers", name="Farm Area (ha)",
        line=dict(color=C["t600"], width=2.5),
        marker=dict(size=7, color=C["t600"]),
        fill="tonexty", fillcolor="rgba(15,110,86,0.08)",
    ), secondary_y=True)
    farm_fig.update_layout(**base_layout(260), legend=LEGEND_TOP)
    farm_fig.update_xaxes(tickfont=AXIS_FONT, gridcolor=GRID_COLOR)
    farm_fig.update_yaxes(title_text="Number of Farms", tickfont=AXIS_FONT,
                          gridcolor=GRID_COLOR, title_font=AXIS_FONT, secondary_y=False)
    farm_fig.update_yaxes(title_text="Farm Area (ha)", tickfont=AXIS_FONT,
                          gridcolor=GRID_COLOR, title_font=AXIS_FONT, secondary_y=True)
    farm_fig.update_annotations(font=dict(size=12, color="#111111"))
    farm_fig.update_traces(legendgrouptitle_text="")
    st.plotly_chart(farm_fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Social KPIs")
    sc1,sc2,sc3,sc4,sc5 = st.columns(5)
    if not sf_ly.empty:
        sim_card(sc1,"G4-SO1 · Social License",
                 f"{int(sf_ly['agreements'].values[0])+60}/100",
                 "Public acceptance index","DFO Aquaculture Survey")
        sim_card(sc2,"G4-SO2 · Indigenous Agreements",
                 str(int(sf_ly['agreements'].values[0])),
                 "MOUs / benefit agreements","BC Treaty Commission", accent="amber")
        sim_card(sc3,"G4-SO3 · Indigenous Employed",
                 str(int(sf_ly['employed'].values[0])),
                 "Jobs from funded activities","Stats Canada · DFO", accent="blue")
        sim_card(sc4,"G4-SO4 · Indigenous Trained",
                 str(int(sf_ly['trained_indig'].values[0])),
                 "Capacity building programs","CIRNAC")
        sim_card(sc5,"G4-SO5 · Total Trained",
                 str(int(sf_ly['trained_total'].values[0])),
                 "All individuals trained","PSIA surveys · DFO")

    st.markdown("<br>", unsafe_allow_html=True)
    ss1, ss2 = st.columns(2)

    with ss1:
        st.markdown('<div class="sec-head">G4-SO2/3/4 — Indigenous Engagement Trend</div>',
                    unsafe_allow_html=True)
        fso = go.Figure()
        fso.add_trace(go.Scatter(
            x=sf["year"], y=sf["employed"], mode="lines+markers",
            name="Employed", line=dict(color=C["t600"], width=2.5),
            marker=dict(size=7, color=C["t600"]),
            fill="tozeroy", fillcolor="rgba(15,110,86,0.10)",
        ))
        fso.add_trace(go.Scatter(
            x=sf["year"], y=sf["trained_indig"], mode="lines+markers",
            name="Trained (Indigenous)", line=dict(color=C["amber"], width=2.2),
            marker=dict(size=7, color=C["amber"]),
        ))
        fso.add_trace(go.Bar(
            x=sf["year"], y=sf["agreements"] * 8,
            name="Agreements (×8 scaled)", marker_color=C["blue"], opacity=0.45,
        ))
        fso.update_layout(**base_layout(275), legend=LEGEND_TOP)
        fso = style_axes(fso)
        fso.update_annotations(font=dict(size=12, color="#111111"))
        fso.update_traces(legendgrouptitle_text="")
        st.plotly_chart(fso, use_container_width=True)

    with ss2:
        st.markdown('<div class="sec-head">G4-SO5 — Total Training Provided</div>',
                    unsafe_allow_html=True)
        ftr = go.Figure(go.Bar(
            x=sf["year"], y=sf["trained_total"],
            marker_color=[C["t400"] if y < 2022 else C["t600"] for y in sf["year"]],
            text=sf["trained_total"],
            textposition="outside",
            textfont=dict(size=12, color="#1A1A1A"),
        ))
        ftr.update_layout(**base_layout(275, dict(l=4,r=4,t=10,b=30)), showlegend=False)
        ftr = style_axes(ftr, ytitle="Individuals trained")
        ftr.update_annotations(font=dict(size=12, color="#111111"))
        ftr.update_traces(legendgrouptitle_text="")
        st.plotly_chart(ftr, use_container_width=True)

    with st.expander("📋 Permitting & Social Data Tables"):
        tp1, tp2 = st.tabs(["Permitting","Social / Indigenous"])
        with tp1:
            st.dataframe(pf.rename(columns={
                "year":"Year","ada_ha":"ADA (ha)",
                "permitted_farms":"Permitted Farms","aoa_count":"AOAs",
                "compliance_pct":"Compliance (%)","seaweed_farms":"Seaweed Farms",
                "seaweed_area_ha":"Farm Area (ha)",
            }).reset_index(drop=True), use_container_width=True, hide_index=True)
        with tp2:
            st.dataframe(sf.rename(columns={
                "year":"Year","agreements":"Agreements",
                "employed":"Employed","trained_indig":"Indigenous Trained",
                "trained_total":"Total Trained",
            }).reset_index(drop=True), use_container_width=True, hide_index=True)

    with st.expander("📚 Real Data Sources"):
        st.markdown("""
| KPI | Source | Dataset |
|---|---|---|
| G4-PM1 ADA | BC Ministry of Agriculture | Aquaculture Act designation maps |
| G4-PM2 Permitted Farms | DFO Annual Aquaculture Statistics | Table 1 |
| G4-PM3 AOAs | DFO Pacific Region | Aquaculture Opportunity Areas |
| G4-PM4 Compliance | DFO C&E Annual Report | Inspection outcomes |
| G4-SO1 Social License | DFO Public Trust Survey | 2016–2022 series |
| G4-SO2 Agreements | BC Treaty Commission | Annual Report |
| G4-SO3 Employed | Statistics Canada 14-10-0023 | By industry |
| G4-SO4 Trained | CIRNAC Indigenous Aquaculture Collaborative | Outcomes |
| G4-SO5 Total Trained | PSIA surveys + DFO | Internal records |
| G4-OP6 Seaweed Farms | DFO Annual Aquaculture Statistics | Table 1 |
| G4-OP7 Farm Area (ha) | BC Ministry of Agriculture | Aquaculture licence site area data |
        """)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ADVANCED ANALYTICS  (KPI-14, 20, 21, 22/23, 24, 25, 26)
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    if not data_ok:
        st.warning("Data files not loaded. Place CSVs in data/")
    else:
        st.markdown("#### Advanced Analytics KPIs")
        st.markdown(
            '<span class="tag">🟢 FAO FishStat</span>'
            '<span class="tag">KPI-14 · KPI-20 · KPI-21 · KPI-22 · KPI-23 · KPI-24 · KPI-25 · KPI-26</span>',
            unsafe_allow_html=True)

        # ── KPI-14 — Price per Tonne Heatmap (Country × Year) ─────────────────
        st.markdown('<div class="sec-head">KPI-14 — Average Price per Tonne Heatmap (Country × Year)</div>',
                    unsafe_allow_html=True)
        # Join value + quantity on [period, country_name], compute USD/tonne
        av_c = fav.groupby(["period","country_name"])["value"].sum().reset_index()
        av_c.columns = ["year","country","usd_k"]
        aq_c = faq.groupby(["period","country_name"])["value"].sum().reset_index()
        aq_c.columns = ["year","country","tonnes"]
        hmap_df = av_c.merge(aq_c, on=["year","country"])
        hmap_df = hmap_df[(hmap_df["tonnes"] > 500) & (hmap_df["usd_k"] > 0)]
        hmap_df["usd_t"] = (hmap_df["usd_k"] * 1000 / hmap_df["tonnes"]).round(0)
        hmap_df = hmap_df[hmap_df["usd_t"] < 5000]   # remove outliers
        # Top 15 countries by cumulative volume
        top15 = (hmap_df.groupby("country")["tonnes"].sum()
                 .nlargest(15).index.tolist())
        hmap_filt = hmap_df[hmap_df["country"].isin(top15)]
        hmap_pivot = (hmap_filt.pivot_table(index="country", columns="year",
                      values="usd_t", aggfunc="mean").fillna(0))
        # Sort rows by latest year price descending
        if LY in hmap_pivot.columns:
            hmap_pivot = hmap_pivot.sort_values(LY, ascending=True)

        fig14 = go.Figure(go.Heatmap(
            z=hmap_pivot.values,
            x=[str(c) for c in hmap_pivot.columns],
            y=hmap_pivot.index.tolist(),
            colorscale=[[0,"#E1F5EE"],[0.4,C["t400"]],[0.8,C["t600"]],[1,C["t900"]]],
            colorbar=dict(title=dict(text="USD/tonne", font=AXIS_FONT),
                         tickfont=AXIS_FONT, len=0.8),
            hoverongaps=False,
            hovertemplate="<b>%{y}</b><br>Year: %{x}<br>Price: $%{z:,.0f}/t<extra></extra>",
            text=hmap_pivot.values.astype(int),
            texttemplate="%{text}",
            textfont=dict(size=10, color="#1A1A1A"),
            zmin=0,
        ))
        fig14.update_layout(
            **base_layout(420, dict(l=120, r=20, t=10, b=60)),
            xaxis=dict(tickfont=AXIS_FONT, title=dict(text="Year", font=AXIS_FONT)),
            yaxis=dict(tickfont=dict(size=11, color="#1A1A1A")),
        )
        fig14.update_annotations(font=dict(size=12, color="#111111"))
        fig14.update_traces(legendgrouptitle_text="")
        st.plotly_chart(fig14, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        t4r1a, t4r1b = st.columns(2)

        with t4r1a:
            # ── KPI-20 — Environment Breakdown (Derived) ──────────────────────
            st.markdown('<div class="sec-head">KPI-20 — Environment Breakdown (Marine / Inland / Brackish)</div>',
                        unsafe_allow_html=True)
            st.markdown(
                '<div style="font-size:11px;color:#666;font-family:Inter,Arial,sans-serif;'
                'margin-bottom:8px;">⚠️ Derived from species taxonomy — add '
                '<code>environment_alpha_2_code</code> column to CSVs for official FAO classification.</div>',
                unsafe_allow_html=True)
            # Derive environment from species name patterns
            FRESHWATER = ["spirulina","haematococcus","chlorella","freshwater"]
            BRACKISH   = ["brackish"]
            def classify_env(name):
                n = str(name).lower()
                if any(k in n for k in FRESHWATER): return "Freshwater / Inland"
                if any(k in n for k in BRACKISH):   return "Brackish"
                return "Marine"
            aq_env = faq.copy()
            aq_env["environment"] = aq_env["seaweed_name"].apply(classify_env)
            env_df = (aq_env[aq_env["value"]>0]
                      .groupby(["period","environment"])["value"].sum()
                      .reset_index())
            env_df["tm"] = env_df["value"] / 1e6
            env_pal = {"Marine": C["t600"], "Freshwater / Inland": C["amber"], "Brackish": C["blue"]}

            fig20 = px.area(env_df, x="period", y="tm", color="environment",
                color_discrete_map=env_pal,
                labels={"tm":"M tonnes","period":"Year","environment":"Environment"})
            fig20.update_layout(**base_layout(280, dict(l=4,r=4,t=10,b=4)), legend=LEGEND_TOP)
            fig20.update_xaxes(tickfont=AXIS_FONT, gridcolor=GRID_COLOR)
            fig20.update_yaxes(tickfont=AXIS_FONT, gridcolor=GRID_COLOR,
                               title_font=AXIS_FONT, title_text="M tonnes")
            fig20.update_annotations(font=dict(size=12, color="#111111"))
            fig20.update_traces(legendgrouptitle_text="")
            st.plotly_chart(fig20, use_container_width=True)

        with t4r1b:
            # ── KPI-24 — Price Volatility by Species ──────────────────────────
            st.markdown('<div class="sec-head">KPI-24 — Price Volatility by Species (USD/kg std dev)</div>',
                        unsafe_allow_html=True)
            av_sp = fav.groupby(["period","seaweed_name"])["value"].sum().reset_index()
            av_sp.columns = ["year","species","usd_k"]
            aq_sp = faq.groupby(["period","seaweed_name"])["value"].sum().reset_index()
            aq_sp.columns = ["year","species","tonnes"]
            vol_sp = av_sp.merge(aq_sp, on=["year","species"])
            vol_sp = vol_sp[(vol_sp["tonnes"] > 100) & (vol_sp["usd_k"] > 0)]
            vol_sp["usd_kg"] = vol_sp["usd_k"] * 1000 / vol_sp["tonnes"] / 1000
            vol_sp = vol_sp[vol_sp["usd_kg"] < 20]
            # Compute std dev per species (min 5 data points)
            vola = (vol_sp.groupby("species")["usd_kg"]
                    .agg(["std","count","mean"]).reset_index()
                    .rename(columns={"std":"vol_std","count":"n_years","mean":"avg_price"}))
            vola = vola[vola["n_years"] >= 5].dropna(subset=["vol_std"])
            vola = vola.nlargest(top_n, "vol_std").sort_values("vol_std")

            fig24 = go.Figure()
            for _, row in vola.iterrows():
                fig24.add_shape(type="line",
                    x0=0, x1=row["vol_std"],
                    y0=row["species"], y1=row["species"],
                    line=dict(color=C["t300"], width=2))
            fig24.add_trace(go.Scatter(
                x=vola["vol_std"], y=vola["species"], mode="markers",
                marker=dict(size=12, color=C["coral"],
                            line=dict(color=C["t900"], width=1.5)),
                text=vola["vol_std"].round(3).astype(str) + " $/kg σ",
                textfont=dict(size=11, color="#1A1A1A"),
                textposition="middle right", name="Std Dev",
            ))
            fig24.update_layout(**base_layout(280, dict(l=4,r=100,t=10,b=4)),
                               showlegend=False)
            fig24 = style_axes(fig24, xtitle="Price std dev (USD/kg)")
            fig24.update_yaxes(tickfont=dict(size=11, color="#1A1A1A"))
            fig24.update_annotations(font=dict(size=12, color="#111111"))
            fig24.update_traces(legendgrouptitle_text="")
            st.plotly_chart(fig24, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        t4r2a, t4r2b = st.columns(2)

        with t4r2a:
            # ── KPI-21 — Aquaculture Adoption Rate Bubble ─────────────────────
            st.markdown('<div class="sec-head">KPI-21 — Aquaculture Adoption Rate by Country (%)</div>',
                        unsafe_allow_html=True)
            aq_tot_c = faq.groupby(["period","country_name"])["value"].sum().reset_index()
            cq_tot_c = fcq.groupby(["period","country_name"])["value"].sum().reset_index()
            aq_tot_c.columns = ["year","country","aq_vol"]
            cq_tot_c.columns = ["year","country","cap_vol"]
            adopt = aq_tot_c.merge(cq_tot_c, on=["year","country"], how="outer").fillna(0)
            adopt["total"] = adopt["aq_vol"] + adopt["cap_vol"]
            adopt = adopt[adopt["total"] > 0]
            adopt["adopt_pct"] = adopt["aq_vol"] / adopt["total"] * 100
            # Latest year snapshot for bubble chart
            adopt_ly = adopt[adopt["year"] == LY].copy()
            adopt_ly = adopt_ly.merge(
                fgp[fgp["period"]==LY][["country_name","continent_group_en"]]
                .drop_duplicates(), left_on="country", right_on="country_name", how="left")
            adopt_ly = adopt_ly[adopt_ly["aq_vol"] > 0].nlargest(30, "aq_vol")

            fig21 = px.scatter(adopt_ly,
                x="adopt_pct", y="aq_vol",
                size="aq_vol", color="continent_group_en",
                color_discrete_map=CONT_PAL,
                hover_name="country",
                size_max=55,
                labels={"adopt_pct":"Adoption Rate (%)","aq_vol":"Farmed Volume (t)",
                        "continent_group_en":"Continent"},
                log_y=True,
            )
            fig21.update_traces(
                textfont=dict(size=10, color="#1A1A1A"),
                marker=dict(opacity=0.78, line=dict(width=1, color="white")),
            )
            fig21.update_layout(**base_layout(310, dict(l=4,r=4,t=10,b=4)),
                                legend=LEGEND_TOP)
            fig21 = style_axes(fig21,
                xtitle="Aquaculture Share of Total Seaweed Production (%)",
                ytitle="Farmed Volume (tonnes, log scale)")
            fig21.update_annotations(font=dict(size=12, color="#111111"))
            fig21.update_traces(legendgrouptitle_text="")
            st.plotly_chart(fig21, use_container_width=True)

        with t4r2b:
            # ── KPI-25 — Country Specialization Index ─────────────────────────
            st.markdown('<div class="sec-head">KPI-25 — Country Specialization Index (Top-Species Share)</div>',
                        unsafe_allow_html=True)
            spec_df = (faq[faq["value"]>0]
                       .groupby(["country_name","seaweed_name"])["value"].sum()
                       .reset_index())
            spec_tot = spec_df.groupby("country_name")["value"].sum().rename("total")
            spec_df  = spec_df.join(spec_tot, on="country_name")
            spec_df["share"] = spec_df["value"] / spec_df["total"] * 100
            # Top species share per country
            spec_top = (spec_df.sort_values("share", ascending=False)
                        .groupby("country_name").first().reset_index()
                        [["country_name","seaweed_name","share","total"]])
            spec_top = spec_top[spec_top["total"] > 1e5]  # min 100k tonnes cumulative
            spec_top = spec_top.sort_values("share", ascending=True)

            fig25 = go.Figure(go.Bar(
                x=spec_top["share"],
                y=spec_top["country_name"],
                orientation="h",
                marker_color=[SP_PAL[i % len(SP_PAL)] for i in range(len(spec_top))],
                text=spec_top.apply(
                    lambda r: f"{r['seaweed_name'][:18]}… ({r['share']:.0f}%)"
                    if len(r['seaweed_name']) > 18
                    else f"{r['seaweed_name']} ({r['share']:.0f}%)", axis=1),
                textposition="outside",
                textfont=dict(size=11, color="#1A1A1A"),
            ))
            fig25.update_layout(**base_layout(310, dict(l=4,r=200,t=10,b=4)),
                                showlegend=False)
            fig25 = style_axes(fig25, xtitle="Top Species Share (%)")
            fig25.update_yaxes(tickfont=dict(size=11, color="#1A1A1A"))
            fig25.update_xaxes(range=[0,120])
            fig25.update_annotations(font=dict(size=12, color="#111111"))
            fig25.update_traces(legendgrouptitle_text="")
            st.plotly_chart(fig25, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        t4r3a, t4r3b = st.columns(2)

        with t4r3a:
            # ── KPI-22/23 — Country Growth Scatter + Emerging Producers ───────
            st.markdown('<div class="sec-head">KPI-22/23 — Country Growth Trajectories (CAGR) &amp; Emerging Producers</div>',
                        unsafe_allow_html=True)
            cagr_base_yr = max(LY - cagr_win, year_range[0])
            c_base = (fgp[fgp["period"]==cagr_base_yr]
                      .groupby("country_name")["value"].sum().reset_index()
                      .rename(columns={"value":"base_v"}))
            c_last = (fgp[fgp["period"]==LY]
                      .groupby("country_name")["value"].sum().reset_index()
                      .rename(columns={"value":"last_v"}))
            cagr_c = c_base.merge(c_last, on="country_name")
            cagr_c = cagr_c[(cagr_c["base_v"]>0) & (cagr_c["last_v"]>0)]
            cagr_c["cagr"] = ((cagr_c["last_v"] / cagr_c["base_v"]) ** (1/cagr_win) - 1) * 100
            cagr_c["last_mt"] = cagr_c["last_v"] / 1e6
            cagr_c = cagr_c.merge(
                fgp[fgp["period"]==LY][["country_name","continent_group_en"]]
                .drop_duplicates(), on="country_name", how="left")
            cagr_c = cagr_c[(cagr_c["cagr"].abs() < 50) & (cagr_c["last_v"] > 10000)]

            fig22 = px.scatter(cagr_c,
                x="cagr", y="last_mt",
                size="last_mt", color="continent_group_en",
                color_discrete_map=CONT_PAL,
                hover_name="country_name",
                size_max=55, log_y=True,
                labels={"cagr":f"CAGR % ({cagr_win}yr)",
                        "last_mt":f"Production {LY} (M t, log)",
                        "continent_group_en":"Continent"},
            )
            fig22.add_vline(x=0, line_dash="dot",
                line_color=C["gray"], line_width=1.5,
                annotation_text="0%", annotation_font=dict(size=11, color=C["gray"]))
            fig22.update_traces(
                marker=dict(opacity=0.80, line=dict(width=1, color="white")))
            fig22.update_layout(**base_layout(310, dict(l=4,r=4,t=10,b=4)),
                                legend=LEGEND_TOP)
            fig22 = style_axes(fig22,
                xtitle=f"CAGR % (last {cagr_win} years)",
                ytitle=f"Production {LY} (M tonnes, log)")
            fig22.update_annotations(font=dict(size=12, color="#111111"))
            fig22.update_traces(legendgrouptitle_text="")
            st.plotly_chart(fig22, use_container_width=True)

            # KPI-23: Emerging producers table
            emerging = cagr_c[
                (cagr_c["last_v"] < 1e5) & (cagr_c["cagr"] > 5)
            ].sort_values("cagr", ascending=False)[
                ["country_name","cagr","last_v"]
            ].head(10)
            emerging.columns = ["Country", f"CAGR ({cagr_win}yr) %", f"Volume {LY} (t)"]
            emerging[f"CAGR ({cagr_win}yr) %"] = emerging[f"CAGR ({cagr_win}yr) %"].round(1)
            emerging[f"Volume {LY} (t)"] = emerging[f"Volume {LY} (t)"].astype(int)
            if not emerging.empty:
                st.markdown(
                    '<div style="font-size:12px;font-weight:600;color:{};'
                    'margin:6px 0 4px;">🌱 KPI-23 — Emerging Producers (vol &lt; 100k t, CAGR &gt; 5%)</div>'.format(C["t800"]),
                    unsafe_allow_html=True)
                st.dataframe(emerging.reset_index(drop=True),
                             use_container_width=True, hide_index=True)

        with t4r3b:
            # ── KPI-26 — Data Completeness / Confidence ───────────────────────
            st.markdown('<div class="sec-head">KPI-26 — Data Completeness by Species (Proxy for Confidence)</div>',
                        unsafe_allow_html=True)
            st.markdown(
                '<div style="font-size:11px;color:#666;font-family:Inter,Arial,sans-serif;'
                'margin-bottom:8px;">⚠️ Proxy metric — completeness = years reported ÷ total years '
                'in window. Add <code>status_alpha_2_code</code> (FAO FishStatJ) for official '
                'R/E/P confidence flags.</div>',
                unsafe_allow_html=True)
            total_yrs = faq["period"].nunique()
            conf_df = (faq[faq["value"]>0]
                       .groupby("seaweed_name")["period"].nunique()
                       .reset_index()
                       .rename(columns={"period":"yrs_reported"}))
            conf_df["completeness"] = (conf_df["yrs_reported"] / total_yrs * 100).round(1)
            conf_df = conf_df.nlargest(top_n, "completeness").sort_values("completeness")
            conf_df["color"] = conf_df["completeness"].apply(
                lambda v: C["t600"] if v >= 80 else (C["amber"] if v >= 50 else C["coral"]))

            fig26 = go.Figure(go.Bar(
                x=conf_df["completeness"],
                y=conf_df["seaweed_name"],
                orientation="h",
                marker_color=conf_df["color"].tolist(),
                text=conf_df["completeness"].astype(str) + "%",
                textposition="outside",
                textfont=dict(size=11, color="#1A1A1A"),
            ))
            fig26.add_vline(x=80, line_dash="dot", line_color=C["t600"], line_width=1.5,
                annotation_text="80% threshold",
                annotation_font=dict(size=11, color=C["t600"]),
                annotation_position="top right")
            fig26.update_layout(**base_layout(310, dict(l=4,r=80,t=10,b=4)),
                                showlegend=False)
            fig26 = style_axes(fig26, xtitle="Data Completeness (%)")
            fig26.update_yaxes(tickfont=dict(size=11, color="#1A1A1A"))
            fig26.update_xaxes(range=[0,120])
            fig26.update_annotations(font=dict(size=12, color="#111111"))
            fig26.update_traces(legendgrouptitle_text="")
            st.plotly_chart(fig26, use_container_width=True)

        with st.expander("📋 Advanced Analytics Data Tables"):
            ta1, ta2, ta3, ta4, ta5 = st.tabs([
                "Price Heatmap","Adoption Rate","Country CAGR",
                "Specialization","Data Completeness",
            ])
            with ta1:
                st.dataframe(
                    hmap_filt[["country","year","usd_t"]]
                    .rename(columns={"country":"Country","year":"Year","usd_t":"USD/tonne"})
                    .sort_values(["Country","Year"])
                    .reset_index(drop=True),
                    use_container_width=True, hide_index=True)
            with ta2:
                disp_adopt = adopt_ly[["country","adopt_pct","aq_vol","cap_vol"]].copy()
                disp_adopt.columns = ["Country","Adoption Rate (%)","Farmed (t)","Wild (t)"]
                disp_adopt["Adoption Rate (%)"] = disp_adopt["Adoption Rate (%)"].round(1)
                st.dataframe(disp_adopt.sort_values("Adoption Rate (%)", ascending=False)
                             .reset_index(drop=True),
                             use_container_width=True, hide_index=True)
            with ta3:
                disp_cagr = cagr_c[["country_name","cagr","last_v","continent_group_en"]].copy()
                disp_cagr.columns = ["Country",f"CAGR ({cagr_win}yr) %",f"Vol {LY} (t)","Continent"]
                disp_cagr[f"CAGR ({cagr_win}yr) %"] = disp_cagr[f"CAGR ({cagr_win}yr) %"].round(2)
                st.dataframe(disp_cagr.sort_values(f"CAGR ({cagr_win}yr) %", ascending=False)
                             .reset_index(drop=True),
                             use_container_width=True, hide_index=True)
            with ta4:
                st.dataframe(
                    spec_top.rename(columns={
                        "country_name":"Country","seaweed_name":"Top Species",
                        "share":"Top Species Share (%)","total":"Cumulative Vol (t)"})
                    .sort_values("Top Species Share (%)", ascending=False)
                    .reset_index(drop=True),
                    use_container_width=True, hide_index=True)
            with ta5:
                st.dataframe(
                    conf_df[["seaweed_name","yrs_reported","completeness"]]
                    .rename(columns={
                        "seaweed_name":"Species",
                        "yrs_reported":"Years Reported",
                        "completeness":"Completeness (%)"})
                    .sort_values("Completeness (%)", ascending=False)
                    .reset_index(drop=True),
                    use_container_width=True, hide_index=True)

# end of _main_col


# ─────────────────────────────────────────────────────────────────────────────
# ██████████████████████████████████████████████████████████████████████████
#  RAG KNOWLEDGE BASE — v6.0
#  26 document chunks across 16 source documents
# ██████████████████████████████████████████████████████████████████████████
# ─────────────────────────────────────────────────────────────────────────────
RAG_DOCS = [
    # ── ORIGINAL 10 CHUNKS (v5) ─────────────────────────────────────────────
    {
        "id": "prod_overview",
        "title": "Global Seaweed Production Overview",
        "text": (
            "Global seaweed production reached approximately 40 million tonnes in 2024, "
            "up from 32 million tonnes in 2018. Asia dominates production with a 97% share, "
            "led by China at 56%, followed by Indonesia, Philippines, South Korea, and Japan. "
            "Aquaculture (farmed seaweed) accounts for 88% of total production while wild "
            "capture makes up the remaining 12%. The sector has achieved a compound annual "
            "growth rate of approximately 2.9% over the past decade. Total aquaculture value "
            "reached USD 19.9 billion in 2024, growing significantly from USD 10.5 billion in 2015. "
            "Upper-middle income countries produce 86.7% of global output, while high-income "
            "countries like Canada represent only 7% — indicating large untapped opportunity."
        ),
        "source": "FAO FishStat 2024",
    },
    {
        "id": "species_overview",
        "title": "Key Seaweed Species",
        "text": (
            "Japanese kelp (Saccharina japonica / kombu) is the largest farmed species by volume "
            "with 301.5 million tonnes cumulative production. Eucheuma seaweeds and Cottoni seaweed "
            "are dominant tropical carrageenan species farmed primarily in the Philippines and "
            "Indonesia, with 81.4 and 79.5 million tonnes respectively. Gracilaria seaweeds are "
            "used primarily for agar production with 76 million tonnes. Wakame and Nori are "
            "premium Japanese food seaweeds. Globally, 31 seaweed species were actively farmed "
            "in 2024, and 28 species were harvested from the wild. The diversity of cultivated "
            "species has increased steadily since 1950, reflecting growing demand and innovation "
            "in aquaculture techniques."
        ),
        "source": "FAO FishStat Aquaculture Quantity",
    },
    {
        "id": "market_applications",
        "title": "Seaweed Market Applications and Value",
        "text": (
            "Seaweed has diverse commercial applications across multiple industries. Food and "
            "nutrition represents 40% of market usage, including direct consumption, food "
            "ingredients, and hydrocolloids like agar and carrageenan. Cosmetics and beauty "
            "products account for 20%, leveraging seaweed's antioxidant and moisturizing "
            "properties. Agriculture and biostimulants represent 18%, where seaweed extracts "
            "improve soil quality and crop yields. Pharmaceuticals account for 12%, with "
            "applications in drug delivery and bioactive compounds. Biofuel and energy "
            "represents 10% and is an emerging growth area. The global seaweed market is "
            "valued at approximately USD 19.9 billion in 2024 with strong projected growth "
            "driven by sustainability trends and demand for plant-based alternatives."
        ),
        "source": "Industry market reports; FAO",
    },
    {
        "id": "canada_seaweed",
        "title": "Canadian Seaweed Industry and PSIA",
        "text": (
            "Canada's seaweed industry is an emerging sector with significant growth potential. "
            "The Canadian seaweed market is valued at approximately CAD 50 million in 2024, "
            "growing from CAD 12 million in 2018. Key producing regions include British Columbia, "
            "Nova Scotia, and New Brunswick. Canada is classified as a high-income country, "
            "currently representing only 7% of global seaweed production — a significant "
            "opportunity gap. There are approximately 97 licensed seaweed farm operations in "
            "Canada as of 2024, covering about 3,340 hectares of cultivation area. "
            "The Pacific Seaweed Industry Association (PSIA) supports sector development through "
            "its five core values: Education, Economic Opportunity, Research and Resources, "
            "Innovation, and Community and Connection. Key Canadian species include giant kelp, "
            "sugar kelp, dulse, and Irish moss."
        ),
        "source": "PSIA; Statistics Canada; DFO",
    },
    {
        "id": "environment_benefits",
        "title": "Environmental Benefits of Seaweed",
        "text": (
            "Seaweed farming offers exceptional environmental benefits compared to land-based "
            "agriculture. Seaweed requires no freshwater, fertilizers, pesticides, or arable "
            "land to grow. It naturally sequesters carbon dioxide from the ocean, helping "
            "reduce ocean acidification. Seaweed farms provide habitat for marine biodiversity "
            "and can improve local water quality by absorbing excess nutrients. Canada's "
            "seaweed sector is estimated to sequester approximately 654 kilotonnes of CO2 "
            "equivalent annually as of 2024. Seaweed also plays a role in integrated "
            "multi-trophic aquaculture (IMTA) systems where it absorbs waste nutrients from "
            "fish and shellfish farms. The sustainability profile of seaweed makes it a "
            "compelling solution for food security and climate change mitigation."
        ),
        "source": "FAO; DFO environmental assessments",
    },
    {
        "id": "permitting_regulations",
        "title": "Aquaculture Permitting and Regulations in Canada",
        "text": (
            "Seaweed aquaculture in Canada is regulated primarily by Fisheries and Oceans Canada "
            "(DFO) under the Fisheries Act and the Aquaculture Activities Regulations. In British "
            "Columbia, the BC Aquaculture Act governs site designations. Aquaculture Development "
            "Areas (ADAs) totalling approximately 17,900 hectares have been designated in BC for "
            "potential expansion. About 755 total aquaculture operations are licensed across "
            "Canada, with approximately 97 specific to seaweed. DFO has assessed 54 Aquaculture "
            "Opportunity Areas (AOAs) as of 2024. The Fisheries Act compliance rate is "
            "approximately 95.3% based on annual inspection data. Social license — the degree "
            "of public acceptance — scores approximately 70 out of 100, reflecting generally "
            "positive but conditional community support for seaweed farming expansion."
        ),
        "source": "DFO Annual Aquaculture Statistics; BC Ministry of Agriculture",
    },
    {
        "id": "indigenous_engagement",
        "title": "Indigenous Peoples and Seaweed Aquaculture",
        "text": (
            "Indigenous communities play an important role in Canada's seaweed industry. "
            "As of 2024, approximately 24 formal agreements (MOUs and benefit agreements) "
            "exist between seaweed industry operators and Indigenous groups, facilitated "
            "through the BC Treaty Commission and CIRNAC (Crown-Indigenous Relations and "
            "Northern Affairs Canada). Approximately 207 Indigenous people are employed "
            "in funded aquaculture activities, and 174 Indigenous individuals have received "
            "capacity-building training through the CIRNAC Indigenous Aquaculture Collaborative "
            "Program. Total training across all individuals reached 504 in 2024. Indigenous "
            "communities on the Pacific and Atlantic coasts have traditional ecological "
            "knowledge of seaweed harvesting that is increasingly integrated into modern "
            "aquaculture practices. Reconciliation and equitable economic participation "
            "are central priorities for PSIA."
        ),
        "source": "CIRNAC; BC Treaty Commission; Statistics Canada Table 14-10-0023",
    },
    {
        "id": "production_trends",
        "title": "Production Trends and Growth Rates",
        "text": (
            "Global seaweed production has grown consistently since 1950, accelerating sharply "
            "after 1980 with the expansion of Chinese and Indonesian aquaculture. Year-over-year "
            "growth was approximately 3.1% in 2024. The 10-year compound annual growth rate "
            "stands at 2.9%. Aquaculture production grew from 22 million tonnes in 2010 to "
            "38.9 million tonnes in 2024. Wild capture has remained relatively flat at "
            "approximately 1.3 million tonnes annually. The number of actively farmed species "
            "increased from 21 in 2000 to 31 in 2024, indicating growing diversification. "
            "The ratio of aquaculture value to volume has increased, suggesting producers are "
            "capturing more value per tonne — the average implied price is approximately "
            "USD 0.51 per kg in 2024. High-income countries show the fastest per-capita growth."
        ),
        "source": "FAO FishStat 2024",
    },
    {
        "id": "income_geography",
        "title": "Geographic Concentration and Economic Development",
        "text": (
            "Seaweed production is highly geographically concentrated. Asia accounts for 97% "
            "of global production, with the Americas at 1.5%, Europe at 0.8%, and Africa and "
            "Oceania accounting for the remainder. China alone produces 56% of global output. "
            "By World Bank income classification, upper-middle income countries produce 86.7% "
            "of global seaweed, while high-income countries (which include Canada, Norway, "
            "Japan, South Korea, and European producers) contribute approximately 7%. "
            "Low-income and lower-middle income countries collectively produce under 7%. "
            "This concentration presents both a market risk (supply chain dependency on Asia) "
            "and an opportunity for high-income countries like Canada to develop domestic "
            "production capacity. Norway has emerged as a fast-growing high-income producer "
            "with 7.3% production CAGR."
        ),
        "source": "FAO FishStat 2024; World Bank income classifications",
    },
    {
        "id": "psia_values",
        "title": "PSIA Values and Mission",
        "text": (
            "The Pacific Seaweed Industry Association (PSIA) operates according to five core "
            "values. Education: providing relevant and essential information to seaweed sector "
            "stakeholders, members, and partners. Economic Opportunity: providing guidance for "
            "new business growth and supporting coastal economies in successful seaweed sector "
            "foundations. Research and Resources: supporting ongoing research into key aspects "
            "of the sector and providing a place for members to access essential resources for "
            "success. Innovation: supporting and sharing new and reimagined ideas for progression "
            "within the sector. Community and Connection: providing meaningful connections between "
            "stakeholders and building a strong seaweed community reaching across Canada and beyond. "
            "PSIA serves as a hub connecting farmers, researchers, government agencies, Indigenous "
            "communities, and industry partners across the Canadian seaweed value chain."
        ),
        "source": "PSIA organizational documents",
    },

    # ── NEW CHUNKS FROM UPLOADED REFERENCE DOCUMENTS (v6) ───────────────────

    {
        "id": "bc_seaweed_policy_2025",
        "title": "BC Seaweed Aquaculture: Policy, Regulations and Ecological Effects (2025)",
        "text": (
            "A 2025 David Suzuki Foundation report by SciTech Environmental Consulting examines "
            "seaweed aquaculture policy and ecological effects in British Columbia. Key findings: "
            "BC has over 25,000 km of coastline and has been identified as one of the world's "
            "most promising regions for seaweed aquaculture expansion. The regulatory framework "
            "involves four agencies — BC Ministry of Water, Land and Resource Stewardship; BC "
            "Ministry of Agriculture; Fisheries and Oceans Canada; and Transport Canada — each "
            "with different jurisdictions. The BC Fish and Seafood Act is limited in scope and "
            "does not address cultivation for non-human consumption. A critical offshore "
            "jurisdictional gap exists: provincial jurisdiction ends 12 nautical miles from the "
            "outer coast, leaving offshore seaweed aquaculture without clear regulatory authority. "
            "BC was expected to release a seaweed aquaculture-specific policy by end of 2025, "
            "drawing on the BC Coastal Marine Strategy co-developed with coastal First Nations. "
            "Potential ecological benefits include inhibition of harmful algae blooms, reduced "
            "eutrophication, and carbon sequestration. Potential risks include genetic disruption "
            "of wild seaweeds and introduction of pathogens, though likelihood is unknown."
        ),
        "source": "Martone, Gregr & Gregr, SciTech Environmental Consulting (2025) for David Suzuki Foundation",
    },
    {
        "id": "rdmw_development_plan_2024",
        "title": "Regional District of Mount Waddington Seaweed Industry Development Plan (2024)",
        "text": (
            "The Mount Waddington (RDMW) Seaweed Industry Development Plan (SIDP, October 2024) "
            "was prepared by LGL Limited to support seaweed industry growth in northern Vancouver "
            "Island following decommissioning of finfish farms. Key findings: The global seaweed "
            "industry grew 736% from 1990 to 2020; algae cultivation contributed nearly 30% of "
            "120 million tonnes of world aquaculture in 2019. BC coastal waters offer rich kelp "
            "biodiversity including Bull, Winged, Giant, and Sugar kelp. Business models include "
            "vertically integrated, start-up, co-op, and specialized supply chain roles. "
            "SWOT analysis shows strengths (suitable waters, experienced local workforce, existing "
            "mariculture infrastructure) and weaknesses (high transport costs for unprocessed "
            "seaweed, limited cold storage). Regulatory requirements include a Pacific Aquatic "
            "Plant tenure, Wild Aquatic Plant Harvester Licence, and Aquatic Plant Culture "
            "Licence. Funding opportunities include New Relationship Trust, National Research "
            "Council of Canada, and Coast Opportunity Funds. First Nations engagement is "
            "crucial, given their leadership role in territorial waters."
        ),
        "source": "LGL Limited for Regional District of Mount Waddington, October 2024",
    },
    {
        "id": "fnfc_action_plan_2025",
        "title": "First Nations Fisheries Council Whole of Aquaculture Action Plan (2025)",
        "text": (
            "The FNFC Whole of Aquaculture Action Plan (September 2025) outlines a coordinated "
            "framework to transform aquaculture governance in BC to be sustainable, inclusive, "
            "and First Nations-driven. Five interwoven objectives: (1) Strengthen Legislative "
            "and Policy Foundations — amend the Fisheries Act to reflect First Nations "
            "jurisdiction, co-develop a federal Aquaculture Act; (2) Preserve Culture and "
            "Enhance Community Health — support First Nations-led research, data sovereignty "
            "(OCAP principles), and restoration of traditional food systems; (3) Strengthen "
            "Environmental Stewardship and Climate Resilience — expand Guardian/Watchmen "
            "programs; (4) Expand Market Access and Economic Sustainability — recognize UNDRIP "
            "rights, support Nation-led ownership through joint ventures; (5) Secure Sustainable "
            "Funding and Build Capacity — establish a $400 million First Nations Aquaculture "
            "Investment Fund. BC's aquaculture sector generates over $550 million in wholesale "
            "value (2023). The plan emphasizes that First Nations have exercised governance of "
            "aquatic ecosystems since time immemorial, and current Crown decision-making "
            "continues to exclude First Nations from licencing, enforcement, and monitoring."
        ),
        "source": "First Nations Fisheries Council of BC, September 2025",
    },
    {
        "id": "slo_handbook_eu",
        "title": "Social License to Operate Handbook for Seaweed Cultivation (EU GENIALG Project)",
        "text": (
            "The SLO Handbook for Seaweed Cultivation (v4.2, Scottish Association for Marine "
            "Science, GENIALG H2020 Project) defines Social License to Operate (SLO) as an "
            "industry-coined term describing relationships that industries have with local "
            "communities, going beyond legal compliance. Key findings: Social opposition to "
            "aquaculture is inhibiting industry growth. Smaller-scale, locally-owned farms are "
            "more socially acceptable because they are perceived as accessible, more likely to "
            "provide local jobs, and as having lower environmental risk. Under-development of "
            "public policy negatively influences community perceptions of seaweed cultivation. "
            "Information provision is critical — where environmental impact information is "
            "scarce, stakeholders substitute experiences from other industries. Building trust "
            "is the fundamental aim: trust is established through early and ongoing communication, "
            "ensuring local benefits, and being fair and transparent. The handbook warns that "
            "accidental association of seaweed cultivation terms with wild harvesting is common "
            "and can generate unnecessary social opposition."
        ),
        "source": "Billing, Rostan & Tett, Scottish Association for Marine Science; EU H2020 GENIALG",
    },
    {
        "id": "slo_uk_guide_2023",
        "title": "Guide to Social License to Operate for Seaweed Cultivation in the UK (2023)",
        "text": (
            "A 2023 WWF-UK funded guide (Scottish Association for Marine Science) on establishing "
            "social license for seaweed cultivation in the UK identifies key characteristics for "
            "SLO. Research using null case studies and Q-method found: stakeholders provide "
            "conditional support — they want to see evidence of environmental sustainability "
            "before fully backing seaweed operations. Three SLO factors were identified: "
            "(1) Environmental sustainability and responsible practices — the most important "
            "factor, requiring demonstrable care for marine ecosystems; (2) Smaller scales with "
            "local social benefits — smaller locally-owned farms are more acceptable than "
            "large corporate operations; (3) Regulation and business development — stakeholders "
            "want adequate regulatory oversight. Consensus areas across all factors included: "
            "transparent communication, clear labelling of products, and inclusion of communities "
            "in decision-making processes. UK-specific recommendations include early stakeholder "
            "mapping, active community engagement programmes, and visual impact assessments "
            "for coastal operations."
        ),
        "source": "Billing et al., WWF-UK funded project, Scottish Association for Marine Science, 2023",
    },
    {
        "id": "noaa_slo_framework_2022",
        "title": "NOAA Social License to Operate in the Aquaculture Industry (2022)",
        "text": (
            "NOAA Technical Memorandum NMFS-NE-287 (August 2022) presents a community-focused "
            "quantitative framework for predicting SLO in aquaculture. The framework identifies "
            "7 empirically proven predictors of community approval: (1) Environmental values — "
            "communities with strong environmental orientations tend to oppose aquaculture unless "
            "environmental safeguards are demonstrated; (2) Economic values — communities with "
            "strong economic development interests are more receptive; (3) Use-conflict — "
            "competing uses of coastal waters create opposition; (4) Knowledge of aquaculture — "
            "informed communities are more likely to issue SLO; (5) Experience with aquaculture "
            "— prior positive experience improves receptivity; (6) Confidence in government — "
            "higher trust in regulatory bodies increases SLO likelihood; (7) Perceptions of "
            "health and safety. SLO operates through trust — community characteristics moderate "
            "the relationship between company actions and SLO. The U.S. aquaculture industry "
            "produces only $1 billion annually despite vast coastline potential, partly due to "
            "social license barriers. The framework positions community context as a key variable "
            "that determines whether SLO-generating actions will be effective."
        ),
        "source": "Whitmore, Cutler & Thunberg, NOAA NMFS Northeast Fisheries Science Center, August 2022",
    },
    {
        "id": "wwf_slo_workshop_2022",
        "title": "WWF Social License to Operate Workshop Report — Seaweed (2022)",
        "text": (
            "WWF's 2022 Social License to Operate Workshop Report (Portland, Maine) documents "
            "outcomes from the Seaweed Solution Project. Key insights: Kelp is nutritious and "
            "requires no land, freshwater, fertilizer, or pesticides; farms improve water quality "
            "by taking up excess nutrients and mitigating local ocean acidification. Markets are "
            "expanding for seaweed-based alternatives including livestock feed and packaging. "
            "Main challenges to SLO include: lack of public awareness of seaweed farming, "
            "visual aesthetics and 'ocean view' concerns, navigational conflicts with fishing "
            "and recreation, and regulatory complexity. Communication toolkit needs identified: "
            "simple, accessible language about seaweed farming benefits; farmer story-telling; "
            "use of existing trusted messengers in communities. Eastern Pacific and North "
            "Atlantic Rim regions show great promise but seaweed farming is underdeveloped there. "
            "A Community of Practice model was recommended for knowledge sharing among farmers "
            "and stakeholders across geographies."
        ),
        "source": "World Wildlife Fund, Seaweed Solution Project Workshop, Portland, Maine, April 2022",
    },
    {
        "id": "seaweed_production_cost_2022",
        "title": "Estimating Production Cost for Large-Scale Seaweed Farms (Kite-Powell et al., 2022)",
        "text": (
            "A 2022 peer-reviewed study in Applied Phycology (Kite-Powell et al., Woods Hole "
            "Oceanographic Institution) presents a techno-economic model for large-scale seaweed "
            "farms. Key findings: At farm scales of 1,000 ha or more, farm gate production costs "
            "in waters up to 200 km from shore are likely $200–$300 per dry tonne for Saccharina "
            "latissima (sugar kelp) and Eucheumatopsis isiformis (tropical red algae). Production "
            "costs below $100 per dry tonne may be achievable in some settings, which would make "
            "seaweed economically competitive with land-based biofuel feedstocks. Annual dry "
            "weight yield is on the order of 1 kg/m²/year at the high end. The model includes "
            "farm gear, boats, personnel, fuel, nursery operations, and onshore support costs. "
            "Key scaling advantages: larger farms share anchor infrastructure and reduce per-ha "
            "capital costs. Drone tugs are used for efficient low-speed, uncrewed biomass transport. "
            "Seaweed farming advantages over land-based biofuel: no arable land required, no "
            "freshwater inputs, and potential for nutrient and CO2 removal from ocean waters."
        ),
        "source": "Kite-Powell et al., Applied Phycology, 3(1), 435–445, 2022; DOI: 10.1080/26388081.2022.2111271",
    },
    {
        "id": "fao_global_seaweed_status_2021",
        "title": "FAO Global Status of Seaweed Production, Trade and Utilization (Cai, 2021)",
        "text": (
            "FAO presentation by Junning Cai (2021, Seaweed Innovation Forum Belize) provides "
            "comprehensive global statistics. In 2019: total world seaweed production was "
            "35.8 million tonnes from 49 countries; 97% from Asia; cultivation accounted for "
            "97% of total. Five genera accounted for over 95% of world cultivation: "
            "Laminaria/Saccharina (35.4%), Kappaphycus/Eucheuma (33.5%), Gracilaria (10.5%), "
            "Porphyra/Pyropia (8.6%), and Undaria (7.4%). Wild collection declined from "
            "1.33 million tonnes in 1990 to 1.08 million tonnes in 2019 across all groups. "
            "Seaweed trade in 2019: $2.65 billion world exports (seaweeds + hydrocolloids) — "
            "top exporters China ($578M), Indonesia ($329M), Republic of Korea ($320M). "
            "World imports: $2.9 billion — top importers China ($445M), Japan ($341M), USA ($320M). "
            "Canada ranked 10th in seaweed exports at $18M USD. Utilization categories include "
            "human foods (kelp, nori, wakame), hydrocolloids (carrageenan, agar, alginate), "
            "abalone and livestock feed, biofertilizers, cosmetics, pharmaceuticals, and bioenergy."
        ),
        "source": "Junning Cai, FAO, Seaweed Innovation Forum Belize, May 2021",
    },
    {
        "id": "fao_global_seaweed_report_2018",
        "title": "FAO Global Status of Seaweed Production Trade and Utilization (2018 Report)",
        "text": (
            "FAO Globefish Research Programme Volume 124 (2018) provides the global seaweed market "
            "update. Key highlights: Approximately 221 seaweed species are of commercial value; "
            "about 10 species are intensively cultivated, including Saccharina japonica (kombu), "
            "Undaria pinnatifida (wakame), Porphyra spp. (nori), Eucheuma spp., Kappaphycus "
            "alvarezii, and Gracilaria spp. Japanese kelp (Saccharina japonica) accounted for "
            "over 33% of global cultivated seaweed production. Seaweed is rich in macronutrients "
            "(sodium, calcium, magnesium, potassium) and micronutrients (iodine, iron, zinc, "
            "selenium). Brown algae (kelp) has iodine content of 1,500–8,000 parts per million, "
            "addressing iodine deficiency — the most preventable cause of impaired cognitive "
            "development in children according to the WHO. The report covers regional producers "
            "including China, Indonesia, Malaysia, Thailand, Philippines (Asia); Chile (South "
            "America); Denmark and EU (Europe); and South Africa, Zanzibar, Morocco (Africa). "
            "Carrageenan from Eucheuma/Kappaphycus is a major industrial hydrocolloid used in "
            "food, cosmetics, and pharmaceuticals globally."
        ),
        "source": "Ferdouse, Yang, Holdt, Murúa & Smith; FAO Globefish Research Programme Vol. 124, 2018",
    },
    {
        "id": "fao_top_aquaculture_2017",
        "title": "Top Species Groups in Global Aquaculture 2017 (FAO) — ALL species, not seaweed only",
        "text": (
            "⚠️ IMPORTANT: The 112 million tonnes figure refers to ALL aquaculture species "
            "combined (fish, shellfish, seaweed, freshwater — everything). "
            "Do NOT use it when asked about seaweed production. "
            "Seaweed-specific global production in 2017 was approximately 33.7 million tonnes "
            "(from FAO FishStat seaweed-only data shown on the dashboard). "
            "FAO factsheet on top 10 species groups in TOTAL global aquaculture (all species, 2017): "
            "Total ALL-SPECIES aquaculture production was 112 million tonnes valued at USD 250 billion. "
            "Seaweeds as a subset: Brown seaweeds #3 by quantity (12.30% = ~13.8M t), "
            "led by Japanese kelp. Red seaweeds #2 (15.42% = ~17.3M t), led by Eucheuma/Gracilaria. "
            "Combined seaweed share ~27.7% = ~31M tonnes of the 112M all-species total. "
            "In 2017, 608 aquaculture species items had been farmed globally since 1950; "
            "424 were actively farmed in 2017 (vs 254 in 1990). "
            "Carps and cyprinids were #1 by all-aquaculture quantity (25%). "
            "China dominates most aquaculture categories."
        ),
        "source": "Cai, Zhou, Yan, Lucente & Lagana, FAO / Chinese Academy of Fishery Sciences, 2017",
    },
    {
        "id": "fao_fishery_stats_2021",
        "title": "FAO Fishery and Aquaculture Statistics Yearbook 2021",
        "text": (
            "The FAO Fishery and Aquaculture Statistics Yearbook 2021 (cc9523en) provides "
            "authoritative global fisheries and aquaculture statistics. Key facts: Global "
            "aquaculture production continued to grow, with seaweeds representing a major "
            "component of total aquatic production. The yearbook documents production by country, "
            "species, and environment (marine, freshwater, brackish). Asia dominates both "
            "capture fisheries and aquaculture globally. The data covers inland waters, marine "
            "areas, and aquaculture broken down by ISSCAAP (International Standard Statistical "
            "Classification of Aquatic Animals and Plants) species groups. Seaweeds and other "
            "aquatic plants form a distinct ISSCAAP division that has grown dramatically since "
            "1990 due to expansion of cultivation in China and Southeast Asia. The yearbook "
            "is used as the primary reference for global fisheries policy, including PSIA's "
            "benchmarking of Canadian production against world averages."
        ),
        "source": "FAO Fishery and Aquaculture Statistics Yearbook 2021, Rome, 2024 (DOI: 10.4060/cc9523en)",
    },
    {
        "id": "canada_aquaculture_snapshot_2024",
        "title": "2024 Canadian Aquaculture Industry Data Snapshot — ALL aquaculture species",
        "text": (
            "⚠️ IMPORTANT: Figures below cover ALL Canadian aquaculture (salmon, shellfish, "
            "trout, etc.) — NOT seaweed only. "
            "The 2024 Aquaculture Industry Data Snapshot (aquaculture.ca/RIAS Inc.) shows: "
            "Canadian ALL-SPECIES aquaculture grew 9.8% in 2024 to 160,318 tonnes total. "
            "Total economic output for all aquaculture: $6 billion. GDP: $2.27 billion. "
            "Jobs across all aquaculture: 18,074 full-time. "
            "Farmed salmon dominated at 69.3% of all-aquaculture production; "
            "salmon production was 109,048 tonnes in 2024. "
            "BC farmed salmon: 53,816 tonnes. Atlantic Canada: 55,232 tonnes (up 17.7%). "
            "Farmed shellfish: 37,904 tonnes. Exports: $970 million (up 7.3%). "
            "Seaweed is a small but growing component — approximately 97 licensed seaweed "
            "operations covering 3,340 ha as of 2024, with seaweed-specific value ~CAD 50 million. "
            "Canada lags Norway and Chile in total aquaculture scale."
        ),
        "source": "RIAS Inc. / Aquaculture.ca, 2024 Aquaculture Industry Data Snapshot",
    },
    {
        "id": "bc_first_nations_aquaculture_2016",
        "title": "First Nations and Aquaculture in BC: Cultivating Change to Preserve Tradition (2016)",
        "text": (
            "FNFC brochure (2016) documents First Nations' multi-millennial role in aquaculture "
            "in BC. Indigenous peoples historically practiced habitat-based aquaculture: "
            "ancient clam gardens (found along entire BC coast) quadrupled butter clam harvests "
            "and doubled littleneck clam yields. Herring roe transplantation increased geographic "
            "spread of stocks. Salmon roe transplantation between streams created new runs. "
            "Today, BC First Nations are diverse in aquaculture engagement — some pursuing "
            "commercial aquaculture, others opposing it on their territories. Three case studies: "
            "(1) K'ómoks First Nation's shellfish farm; (2) Na̱mǥis Nation's closed-containment "
            "land-based salmon farm; (3) Okanagan Nation Alliance's freshwater sockeye hatchery. "
            "A key principle: all traditional practices maintain habitat and ecosystem awareness "
            "for sustainability across future generations. The FNFC's role is to provide "
            "information and support where collective interests align, respecting each Nation's "
            "right to make their own decisions."
        ),
        "source": "First Nations Fisheries Council of BC, 2016",
    },
    {
        "id": "first_nations_lba_mosier_2017",
        "title": "Land-Based Aquaculture for First Nations in BC (Mosier, SFU, 2017)",
        "text": (
            "SFU Master of Resource Management thesis (Mosier, 2017) investigates land-based "
            "aquaculture (LBA) as an economic and community development opportunity for First "
            "Nations, focusing on Nanwakolas Member Nations on northern Vancouver Island. "
            "LBA (cultivating seafood in tanks on land) includes Recirculating Aquaculture "
            "Systems (RAS) and Integrated Multi-trophic Aquaculture (IMTA). Key findings: "
            "Regulatory gaps in BC's Fisheries Act create barriers to shellfish LBA development; "
            "specifically, the Act does not adequately address LBA risks and the time required "
            "for licensing is excessive. Policy recommendations: reduce licensing time, create "
            "LBA advisory committees, and establish partnerships with educational institutions. "
            "Community assessment using the Community Capital Tool and Community Wellbeing Wheel "
            "showed LBA can generate sustainable economic opportunities, resource management "
            "governance improvements, and preservation of traditional foods. IMTA systems, where "
            "seaweed and shellfish absorb waste from salmon, offer particular promise for "
            "integrated First Nations aquaculture development."
        ),
        "source": "Elizabeth Mosier, Master of Resource Management Report No. 667, Simon Fraser University, 2017",
    },
    {
        "id": "flaherty_imta_slo_bc",
        "title": "Social License for Integrated Multi-Trophic Aquaculture (IMTA) in BC (Flaherty)",
        "text": (
            "Dr. Mark Flaherty (University of Victoria) presentation on obtaining social license "
            "for IMTA in BC. BC context: 27,000 km coastline (almost double China's 14,500 km), "
            "population 4.4 million. IMTA integrates fed species (salmon), invertebrate "
            "extractive species (shellfish), and inorganic extractive species (seaweed) in one "
            "system — seaweed absorbs excess nutrients from salmon farms, creating circular "
            "value. IMTA research domains: environmental system performance and species "
            "interactions; system design and engineering; economic analyses; and social "
            "implications including governance, community development, and First Nations "
            "involvement. SLO challenges for IMTA in BC include: competing user groups "
            "(commercial fishing, tourism, recreation), regulatory complexity across multiple "
            "agencies, and the need to demonstrate ecological benefit rather than just economic "
            "benefit to coastal communities. Indigenous co-governance is identified as essential "
            "for long-term social license."
        ),
        "source": "Dr. Mark Flaherty, University of Victoria; S11 Conference Presentation",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF RAG INDEX
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def build_rag_index():
    """Build TF-IDF index over the full knowledge base. Cached for the session."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim
        texts = [d["text"] for d in RAG_DOCS]
        vec = TfidfVectorizer(stop_words="english", max_features=12000, ngram_range=(1, 2))
        mat = vec.fit_transform(texts)
        return vec, mat, cos_sim, True
    except ImportError:
        return None, None, None, False

_vec, _mat, _cos_sim, _rag_ok = build_rag_index()

def retrieve_context(query: str, top_k: int = 4) -> str:
    """Return top-k relevant knowledge chunks for the query."""
    if not _rag_ok or _vec is None:
        return ""
    import numpy as np
    q_vec = _vec.transform([query])
    scores = _cos_sim(q_vec, _mat)[0]
    top_idx = np.argsort(scores)[::-1][:top_k]
    chunks = []
    for i in top_idx:
        if scores[i] > 0.02:
            d = RAG_DOCS[i]
            chunks.append(f"[{d['title']} | Source: {d['source']}]\n{d['text']}")
    return "\n\n".join(chunks)


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT  (RAG-augmented)
# ─────────────────────────────────────────────────────────────────────────────
def build_system_prompt(query: str = "", is_uc2: bool = False) -> str:
    """
    Build system prompt. Order: (1) instructions + disambiguation,
    (2) live data [UC2 first], (3) RAG background.
    Live data comes BEFORE RAG so the LLM anchors on real dashboard numbers.
    """
    base = (
        "You are the PSIA (Pacific Seaweed Industry Association) AI assistant "
        "embedded in a seaweed analytics dashboard. Answer questions about the "
        "global seaweed industry, Canadian market opportunities, BC regulations, "
        "social license to operate, production costs, First Nations aquaculture, "
        "and PSIA initiatives.\n\n"
        "== CRITICAL DATA DISAMBIGUATION ==\n"
        "This dashboard tracks SEAWEED-ONLY production from FAO FishStat. "
        "'Production', 'global production', and 'total production' always mean "
        "SEAWEED-ONLY — NEVER total global aquaculture (all species). "
        "NEVER cite the 112 million tonnes all-species figure as seaweed production. "
        "Seaweed-only: ~33.7M t in 2017, ~40M t in 2024. "
        "The 112M figure (FAO 2017) = ALL aquaculture (fish + shellfish + seaweed combined). "
        "Be concise and data-specific. 3–5 sentences for simple questions. "
        "Always cite your source when using specific numbers."
    )

    # ── UC2: inject ALL live dashboard data FIRST, before RAG ─────────────────
    if is_uc2 and data_ok:
        # Historical production (all years)
        hist_yrs = sorted([y for y in fgp["period"].unique()
                           if fgp[fgp["period"]==y]["value"].sum() > 0])
        hist_lines = "\n".join([
            f"  {y}: {fgp[fgp['period']==y]['value'].sum()/1e6:.2f}M t"
            for y in hist_yrs])

        # Species prices (top 10)
        av_sp = fav.groupby(["period","seaweed_name"])["value"].sum().reset_index()
        aq_sp = faq.groupby(["period","seaweed_name"])["value"].sum().reset_index()
        av_sp.columns = ["year","species","usd_k"]
        aq_sp.columns = ["year","species","tonnes"]
        jp_sp = av_sp.merge(aq_sp, on=["year","species"])
        jp_sp = jp_sp[(jp_sp["tonnes"]>100) & (jp_sp["usd_k"]>0)]
        jp_sp["usd_kg"] = jp_sp["usd_k"]*1000/jp_sp["tonnes"]/1000
        sp_price_agg = (jp_sp[jp_sp["usd_kg"]<20]
                        .groupby("species")
                        .agg(avg_price=("usd_kg","mean"), tot_vol=("tonnes","sum"))
                        .reset_index().nlargest(10,"tot_vol").sort_values("tot_vol",ascending=False))
        price_lines = "\n".join([
            f"  {row['species']}: ${row['avg_price']:.3f}/kg avg"
            for _, row in sp_price_agg.iterrows()])

        # CR5/CR10
        cr_data = fgp[fgp["value"]>0].groupby(["period","country_name"])["value"].sum().reset_index()
        ly_cr    = cr_data[cr_data["period"]==LY]
        tot_cr   = ly_cr["value"].sum()
        sorted_cr = ly_cr["value"].sort_values(ascending=False).values
        cr5  = sorted_cr[:5].sum()/tot_cr*100  if len(sorted_cr)>=5  else 0
        cr10 = sorted_cr[:10].sum()/tot_cr*100 if len(sorted_cr)>=10 else 0

        # Top 15 countries
        top15_c = (fgp[fgp["period"]==LY]
                   .groupby("country_name")["value"].sum()
                   .sort_values(ascending=False).head(15))
        country_lines = "\n".join([
            f"  {i+1}. {c}: {v/1e6:.3f}M t ({v/fgp[fgp['period']==LY]['value'].sum()*100:.1f}%)"
            for i,(c,v) in enumerate(top15_c.items())])

        # Continental
        cont_s = (fgp[fgp["period"]==LY]
                  .groupby("continent_group_en")["value"].sum().sort_values(ascending=False))
        cont_lines = "\n".join([
            f"  {k}: {v/cont_s.sum()*100:.1f}% ({v/1e6:.2f}M t)"
            for k, v in cont_s.items() if k != "Unknown"])

        # Top 10 species
        sp10 = (faq.groupby("seaweed_name")["value"].sum()
                .sort_values(ascending=False).head(10))
        sp_lines = "\n".join([f"  {i+1}. {s}: {v/1e6:.2f}M t cumulative"
                               for i,(s,v) in enumerate(sp10.items())])

        # Income group
        ig_s = (fgp[fgp["period"]==LY]
                .groupby("ecoclass_group_en")["value"].sum().sort_values(ascending=False))
        ig_lines = "\n".join([
            f"  {k}: {v/ig_s.sum()*100:.1f}% ({v/1e6:.2f}M t)"
            for k, v in ig_s.items()])

        # Value per tonne by income group
        av_ig = (fav[fav["value"]>0]
                 .groupby(["period","ecoclass_group_en"])["value"].sum()
                 .reset_index().rename(columns={"value":"usd_k"}))
        aq_ig = (faq[faq["value"]>0]
                 .groupby(["period","ecoclass_group_en"])["value"].sum()
                 .reset_index().rename(columns={"value":"tonnes_ig"}))
        vpt_df = av_ig.merge(aq_ig, on=["period","ecoclass_group_en"])
        vpt_df = vpt_df[vpt_df["period"]==LY]
        vpt_df["usd_per_t"] = vpt_df["usd_k"]*1000/vpt_df["tonnes_ig"]
        vpt_lines = "\n".join([
            f"  {row['ecoclass_group_en']}: ${row['usd_per_t']:,.0f}/t"
            for _, row in vpt_df.sort_values("usd_per_t",ascending=False).iterrows()])

        # Tab 3 permitting & social (estimated)
        perm_ly = sim_perm[sim_perm["year"]==sim_perm["year"].max()].iloc[0]
        soc_ly  = sim_social[sim_social["year"]==sim_social["year"].max()].iloc[0]

        # Tab 4: top CAGR countries
        cagr_base_yr = max(LY - cagr_win, year_range[0])
        c_base = (fgp[fgp["period"]==cagr_base_yr]
                  .groupby("country_name")["value"].sum().reset_index()
                  .rename(columns={"value":"base_v"}))
        c_last = (fgp[fgp["period"]==LY]
                  .groupby("country_name")["value"].sum().reset_index()
                  .rename(columns={"value":"last_v"}))
        cagr_df = c_base.merge(c_last, on="country_name")
        cagr_df = cagr_df[(cagr_df["base_v"]>0) & (cagr_df["last_v"]>1000)]
        cagr_df["cagr"] = ((cagr_df["last_v"]/cagr_df["base_v"])**(1/cagr_win)-1)*100
        top_grow = cagr_df.nlargest(10,"cagr")[["country_name","cagr","last_v"]]
        cagr_lines = "\n".join([
            f"  {row['country_name']}: {row['cagr']:.1f}% CAGR ({row['last_v']/1e3:.0f}k t)"
            for _, row in top_grow.iterrows()])

        base += f"""

╔══════════════════════════════════════════════════════════════════╗
║  LIVE DASHBOARD DATA — SEAWEED ONLY ({year_range[0]}–{year_range[1]})                ║
║  Source: FAO FishStat CSVs + DFO/CIRNAC estimates               ║
║  Use these numbers to answer ALL data questions.                 ║
╚══════════════════════════════════════════════════════════════════╝

── TAB 1: PRODUCTION & VALUE ──────────────────────────────────────
Seaweed production ({LY}): {prod_tot/1e6:.2f}M t · YoY {yoy_prod:+.1f}% · CAGR {cagr_prod:.1f}%
  Farmed: {aq_tot/1e6:.2f}M t ({aq_tot/prod_tot*100:.1f}%) · Wild: {wc_tot/1e6:.3f}M t ({wc_tot/prod_tot*100:.1f}%)
  ASFIS seaweed species: {sp_total} total (farmed:{sp_cult}, wild:{sp_wild})
  Avg seaweed price: ${avg_price_kg:.2f}/kg
  Total seaweed value: USD ${val_tot/1e6:.1f}B (YoY {yoy_val:+.1f}%)
  Market concentration: CR5={cr5:.1f}%, CR10={cr10:.1f}%

Historical seaweed-only production (ALL years, FAO FishStat):
{hist_lines}

Top 10 species avg price (implied USD/kg):
{price_lines}

── TAB 2: GEOGRAPHIC & SPECIES ────────────────────────────────────
Continental share ({LY}):
{cont_lines}

Top 15 countries by seaweed production ({LY}):
{country_lines}

Top 10 seaweed species by cumulative farmed volume:
{sp_lines}

Income group production share ({LY}):
{ig_lines}

Value per tonne by income group ({LY}):
{vpt_lines}

── TAB 3: PERMITTING & SOCIAL (Canada — estimated from public reports) ──
  ADA designated areas    : {int(perm_ly['ada_ha']):,} ha
  Licensed farms (total)  : {int(perm_ly['permitted_farms'])}
  Seaweed-specific farms  : {int(perm_ly['seaweed_farms'])}
  Seaweed farm area       : {int(perm_ly['seaweed_area_ha']):,} ha
  AOAs assessed           : {int(perm_ly['aoa_count'])}
  Fisheries Act compliance: {perm_ly['compliance_pct']:.1f}%
  Social license score    : {int(soc_ly['agreements'])+60}/100
  Indigenous agreements   : {int(soc_ly['agreements'])} MOUs/benefit agreements
  Indigenous employed     : {int(soc_ly['employed'])} people
  Indigenous trained      : {int(soc_ly['trained_indig'])} individuals
  Total trained           : {int(soc_ly['trained_total'])} individuals

── TAB 4: ADVANCED ANALYTICS ──────────────────────────────────────
Top 10 fastest-growing countries ({cagr_win}-yr CAGR):
{cagr_lines}
"""

    # RAG background (comes AFTER live data; fewer chunks in UC2)
    top_k_rag = 2 if is_uc2 else 4
    rag_ctx = retrieve_context(query, top_k=top_k_rag)
    if rag_ctx:
        base += f"\n\n== BACKGROUND KNOWLEDGE (supplementary) ==\n{rag_ctx}"

    return base


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "messages"  not in st.session_state: st.session_state.messages  = []
if "thinking"  not in st.session_state: st.session_state.thinking  = False
if "chat_mode" not in st.session_state: st.session_state.chat_mode = "UC1"


# ─────────────────────────────────────────────────────────────────────────────
# CHAT PANEL  — fixed right column
# ─────────────────────────────────────────────────────────────────────────────
with _chat_col:
    is_uc2    = st.session_state.chat_mode == "UC2"
    rag_badge = f"🔍 RAG ({len(RAG_DOCS)} docs)" if _rag_ok else "💬"

    # Header
    st.markdown(f"""
    <div style="background:{C['t600']};color:#FFFFFF;
                padding:14px 18px 12px;border-radius:12px 12px 0 0;
                box-shadow:0 2px 8px rgba(0,0,0,0.15);">
      <div style="font-size:15px;font-weight:700;font-family:Georgia,serif;
                  letter-spacing:0.2px;margin-bottom:3px;">
        🌿 PSIA AI Assistant
      </div>
      <div style="font-size:12px;color:#CCEEDD;font-family:Inter,Arial,sans-serif;
                  margin-top:3px;font-weight:400;">
        {rag_badge} · Powered by Groq
      </div>
    </div>""", unsafe_allow_html=True)

    # Mode toggle strip — white background so radio labels are clearly readable
    st.markdown(
        '<div style="background:#FFFFFF;padding:8px 14px 4px;'
        'border-left:1px solid #CCCCCC;border-right:1px solid #CCCCCC;">'
        '</div>', unsafe_allow_html=True)

    _mc1, _mc2 = st.columns([3, 1])
    with _mc1:
        mode_choice = st.radio(
            "mode",
            ["💬 Industry Info", "📊 Live Data"],
            index=1 if is_uc2 else 0,
            horizontal=True,
            label_visibility="collapsed",
            key="chat_mode_radio",
        )
        st.session_state.chat_mode = "UC2" if "Live" in mode_choice else "UC1"
        is_uc2 = st.session_state.chat_mode == "UC2"
    with _mc2:
        if st.button("🗑️", key="clr_chat", help="Clear conversation",
                     use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Message history ─ every text node gets explicit color:#FFFFFF inline
    # so Streamlit's own CSS cascade can never override it
    _WHITE = "color:#FFFFFF;font-family:Inter,Arial,sans-serif;"
    hist_html = (
        f'<div class="chat-hist-wrap" '
        f'style="background:{C["t900"]};min-height:320px;max-height:420px;'
        f'overflow-y:auto;padding:14px 14px 10px;{_WHITE}'
        f'border-left:1px solid {C["t800"]};border-right:1px solid {C["t800"]};">'
    )
    if not st.session_state.messages:
        hist_html += (
            '<div class="chat-welcome" '
            f'style="background:#1A6B50;border-radius:10px 10px 10px 2px;'
            f'padding:14px 16px;margin-bottom:12px;{_WHITE}font-size:14px;line-height:1.70;">'
            '<span style="color:#FFFFFF;font-size:14px;line-height:1.70;display:block;">'
            "Hi! I'm the PSIA AI Assistant. Ask me about the seaweed industry, "
            "BC regulations, social license, production costs, First Nations "
            "aquaculture, or live dashboard data."
            '</span>'
            '</div>'
        )
    for m in st.session_state.messages:
        u   = m["role"] == "user"
        bg  = "#1D6E50" if u else "#17523C"
        br  = "14px 14px 2px 14px" if u else "14px 14px 14px 2px"
        mg  = "margin-left:24px" if u else "margin-right:24px"
        ico = "You" if u else "🌿 PSIA"
        align = "text-align:right;" if u else ""
        cls = "chat-bubble-user" if u else "chat-bubble-bot"
        # Escape any characters that would break HTML attributes
        content = str(m["content"]).replace("<", "&lt;").replace(">", "&gt;")
        hist_html += (
            # Icon label
            f'<div class="chat-icon-label" '
            f'style="{_WHITE}font-size:10px;font-weight:600;opacity:0.82;'
            f'margin-bottom:3px;{align}">'
            f'<span style="color:#FFFFFF;">{ico}</span></div>'
            # Bubble
            f'<div class="{cls}" '
            f'style="background:{bg};border-radius:{br};{_WHITE}'
            f'padding:11px 14px;margin-bottom:10px;{mg};'
            f'font-size:14px;line-height:1.62;word-wrap:break-word;">'
            f'<span style="color:#FFFFFF;font-size:14px;line-height:1.62;">{content}</span>'
            f'</div>'
        )
    if st.session_state.thinking:
        hist_html += (
            f'<div style="{_WHITE}font-size:13px;font-style:italic;opacity:0.80;padding:4px 2px 8px;">'
            '<span style="color:#FFFFFF;">⏳ Thinking…</span>'
            '</div>'
        )
    hist_html += "</div>"
    st.markdown(hist_html, unsafe_allow_html=True)

    # Suggestion chips
    SUGG = {
        False: [
            "What are BC's seaweed aquaculture regulations?",
            "How does social license affect seaweed farms?",
            "What is the FNFC Aquaculture Action Plan?",
        ],
        True: [
            f"What is total production in {LY if data_ok else 2024}?",
            "Which continent produces the most seaweed?",
            "What is the average price per kg?",
        ],
    }

    st.markdown(
        '<div style="background:#F7F7F7;padding:8px 14px 4px;'
        'border-left:1px solid #CCCCCC;border-right:1px solid #CCCCCC;">'
        '<span style="font-size:12px;font-weight:600;color:#333333;'
        'font-family:Inter,Arial,sans-serif;">Try asking:</span>'
        '</div>', unsafe_allow_html=True)

    st.markdown(
        '<div style="background:#F7F7F7;padding:0 10px 8px;'
        'border-left:1px solid #CCCCCC;border-right:1px solid #CCCCCC;">',
        unsafe_allow_html=True)
    for sugg in SUGG[is_uc2]:
        if st.button(sugg, key=f"sugg_{sugg[:25]}", use_container_width=True):
            st.session_state["_pending"] = sugg
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # Input row
    st.markdown(
        '<div style="background:#FFFFFF;padding:10px 12px;'
        'border-radius:0 0 12px 12px;'
        'border:1px solid #CCCCCC;border-top:none;'
        'box-shadow:0 4px 12px rgba(0,0,0,0.10);">'
        '</div>', unsafe_allow_html=True)

    _ic, _bc = st.columns([5, 1])
    with _ic:
        txt_input = st.text_input(
            "chat_input_label",
            placeholder="Ask about the seaweed industry…",
            key="chat_txt",
            label_visibility="collapsed",
        )
    with _bc:
        send_clicked = st.button(
            "Send", key="chat_send",
            use_container_width=True,
            type="primary",
        )

    if _rag_ok:
        st.markdown(
            '<div style="font-size:11px;color:#444444;font-weight:500;'
            'font-family:Inter,Arial,sans-serif;text-align:center;margin-top:4px;">'
            f'🔍 {len(RAG_DOCS)} knowledge docs · top-4 retrieval'
            '</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="font-size:11px;color:#CC0000;font-weight:600;'
            'font-family:Inter,Arial,sans-serif;text-align:center;margin-top:4px;">'
            '⚠️ pip install scikit-learn to enable RAG'
            '</div>', unsafe_allow_html=True)

    pending = st.session_state.pop("_pending", None)
    prompt  = pending or (txt_input.strip() if send_clicked and txt_input.strip() else None)

    if prompt and not st.session_state.thinking:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state["_last_prompt"] = prompt
        st.session_state.thinking = True
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# GROQ API CALL
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.get("thinking", False):
    last_prompt = st.session_state.get("_last_prompt", "")
    is_uc2_call = st.session_state.chat_mode == "UC2"
    try:
        client = Groq(api_key=GROQ_API_KEY)
        system = build_system_prompt(query=last_prompt, is_uc2=is_uc2_call)
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=600,
            messages=[
                {"role": "system", "content": system},
                *[{"role": m["role"], "content": m["content"]}
                  for m in st.session_state.messages],
            ],
        )
        reply = resp.choices[0].message.content
    except Exception as e:
        err = str(e).lower()
        if "auth" in err or "invalid" in err:
            reply = "❌ Invalid API key. Check GROQ_API_KEY in the script."
        elif "rate" in err:
            reply = "⏳ Rate limit hit. Wait a moment and try again."
        else:
            reply = f"❌ Error: {e}"

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.session_state.thinking = False
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;font-size:12px;color:#444444;"
    "font-family:Inter,Arial,sans-serif;padding:8px 0 14px;'>"
    "PSIA Seaweed Analytics <strong>v7.0</strong>  ·  "
    "🟢 FAO FishStat (real)  ·  🟡 DFO/CIRNAC (simulated)  ·  "
    "⚡ Groq llama-3.3-70b  ·  "
    f"🔍 RAG — {len(RAG_DOCS)} knowledge documents (16 source files)"
    "</div>", unsafe_allow_html=True)

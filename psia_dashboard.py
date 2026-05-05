"""
PSIA Seaweed Industry Analytics Dashboard  — v4.0
===================================================
  - Sidebar: filters only
  - AI chatbox: collapsible panel at bottom of main content (click to expand/close)
  - Groq API key hardcoded
  - All chart/graph fonts dark and readable
  - 3 main tabs: Production & Value | Geographic & Species | Permitting & Social

Setup:
  pip install streamlit plotly pandas numpy groq
  Place 4 CSVs inside  data/
  streamlit run psia_dashboard.py
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
# COLOURS
# ─────────────────────────────────────────────────────────────────────────────
C = {
    "t900": "#04342C", "t800": "#085041", "t600": "#0F6E56",
    "t400": "#1D9E75", "t300": "#5DCAA7", "t100": "#9FE1CB", "t50": "#E1F5EE",
    "amber": "#C97D10", "coral": "#B84020", "blue": "#1E5FA8",
    "gray": "#555550",  "green": "#1B5E20",
    "txt":  "#1A1A1A",   # main readable text on charts
    "txt2": "#444440",   # secondary axis / labels
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

# ─────────────────────────────────────────────────────────────────────────────
# CHART LAYOUT DEFAULTS  — dark readable fonts
# ─────────────────────────────────────────────────────────────────────────────
CHART_FONT   = dict(family="Arial, sans-serif", size=12, color=C["txt"])
AXIS_FONT    = dict(family="Arial, sans-serif", size=11, color=C["txt2"])
TITLE_FONT   = dict(family="Arial, sans-serif", size=12, color=C["txt"])
GRID_COLOR   = "#E0E0E0"
AXIS_LINE    = dict(color="#CCCCCC", width=1)

# Default legend style — apply per chart, never inside base_layout
LEGEND_TOP = dict(
    orientation="h", y=1.12, x=0,
    font=dict(size=11, color=C["txt"]),
    bgcolor="rgba(255,255,255,0.85)",
    bordercolor="#DDDDDD", borderwidth=1,
)
LEGEND_BOTTOM = dict(
    orientation="h", y=-0.42, x=0,
    font=dict(size=10, color=C["txt"]),
    bgcolor="rgba(255,255,255,0.85)",
    bordercolor="#DDDDDD", borderwidth=1,
    title_text="",
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
    )

def style_axes(fig, xtitle="", ytitle="", y2title="", y_range=None):
    """Apply consistent readable axis styling."""
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
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
[data-testid="stSidebar"] {{ background: {C['t900']}; }}
[data-testid="stSidebar"] * {{ color: white !important; }}

/* KPI cards */
.kpi-card {{
    background: white; border-radius: 10px; padding: 14px 18px;
    border: 1px solid #D8EDE5; border-top: 3px solid {C['t400']};
    margin-bottom: 4px; box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}}
.kpi-card.amber {{ border-top-color: {C['amber']}; }}
.kpi-card.coral {{ border-top-color: {C['coral']}; }}
.kpi-card.blue  {{ border-top-color: {C['blue']};  }}
.kpi-label {{
    font-size: 9.5px; color: #777; text-transform: uppercase;
    letter-spacing: 1.1px; margin-bottom: 5px;
    font-family: Arial, sans-serif;
}}
.kpi-value {{
    font-size: 24px; font-weight: 700; color: {C['t600']};
    line-height: 1.1; font-family: Georgia, serif;
}}
.kpi-delta  {{ font-size: 10.5px; color: {C['green']}; margin-top: 4px; font-family: Arial, sans-serif; }}
.kpi-delta.neg {{ color: {C['coral']}; }}

/* Section headings */
.sec-head {{
    font-family: Georgia, serif; font-size: 14px; font-weight: 600;
    color: {C['t800']}; padding-bottom: 5px;
    border-bottom: 2px solid {C['t50']}; margin-bottom: 10px;
}}

/* Source tags */
.tag     {{ display:inline-block; background:{C['t50']}; color:{C['t800']};
            border-radius:4px; padding:2px 7px; font-size:10px;
            font-family:Arial,sans-serif; margin-right:4px; margin-bottom:4px; }}
.tag.sim {{ background:#FFF4E5; color:#7A4F00; }}
.tag.ext {{ background:#E8F0FE; color:#1A3A6B; }}

/* Collapsible chat panel */
.chat-panel {{
    background: {C['t900']}; border-radius: 14px 14px 0 0;
    padding: 0; margin-top: 20px;
    border: 1px solid {C['t800']};
}}
.chat-header {{
    background: {C['t800']}; border-radius: 14px 14px 0 0;
    padding: 10px 18px; cursor: pointer;
    display: flex; justify-content: space-between; align-items: center;
}}
.chat-body {{
    padding: 12px 16px 14px;
}}

/* Chat bubbles */
.bubble-user {{
    background: {C['t400']}; color: white;
    border-radius: 14px 14px 2px 14px;
    padding: 10px 14px; margin: 6px 0 6px 60px;
    font-size: 13px; font-family: Arial, sans-serif; line-height: 1.5;
    word-wrap: break-word;
}}
.bubble-bot {{
    background: rgba(255,255,255,0.10); color: #E0F5EE;
    border-radius: 14px 14px 14px 2px;
    padding: 10px 14px; margin: 6px 60px 6px 0;
    font-size: 13px; font-family: Arial, sans-serif; line-height: 1.5;
    word-wrap: break-word;
}}
.bubble-think {{
    color: rgba(255,255,255,0.45); font-style: italic;
    padding: 6px 14px; font-size: 12px; font-family: Arial, sans-serif;
}}
.chat-icon {{
    font-size: 9px; color: rgba(255,255,255,0.4);
    font-family: Arial, sans-serif; margin-bottom: 2px;
}}
.chat-icon.right {{ text-align: right; }}

/* Suggestion buttons inside chat */
div.chat-sugg button {{
    background: rgba(255,255,255,0.10) !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
    color: white !important;
    border-radius: 8px; font-size: 12px; padding: 6px 12px;
    text-align: left; white-space: normal; height: auto;
    width: 100%; margin-bottom: 4px;
}}
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
# SIDEBAR  — filters only
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center;padding:16px 0 12px;">
      <div style="font-size:32px;margin-bottom:6px;">🌿</div>
      <div style="font-family:Georgia,serif;font-size:18px;font-weight:700;">
        PSIA Dashboard</div>
      <div style="font-size:9.5px;color:{C['t100']};margin-top:2px;">
        Pacific Seaweed Industry Association</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        f"<div style='font-size:9.5px;letter-spacing:1.1px;text-transform:uppercase;"
        f"color:{C['t100']};margin-bottom:8px;'>Dashboard Filters</div>",
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
        f"<div style='font-size:9px;color:{C['t100']};line-height:1.8;'>"
        "🟢 FAO FishStat — real data<br>"
        "🟡 DFO / CIRNAC — simulated<br>"
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
            f'<div style="font-size:9px;color:#aaa;margin-top:5px;'
            f'font-family:Arial,sans-serif;">Source: {src}</div>'
            f'</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:{C['t900']};color:white;padding:18px 24px;
            border-radius:10px;margin-bottom:16px;">
  <div style="display:flex;justify-content:space-between;
              align-items:center;flex-wrap:wrap;gap:10px;">
    <div>
      <div style="font-family:Georgia,serif;font-size:21px;font-weight:700;
                  margin-bottom:4px;">
        🌿 PSIA Seaweed Industry Analytics Dashboard
      </div>
      <div style="font-size:11px;opacity:0.75;">
        Production · Economic Value · Species · Geographic · Permitting · Social KPIs
      </div>
    </div>
    <div style="text-align:right;font-family:Arial,sans-serif;">
      <div style="font-size:9.5px;opacity:0.55;text-transform:uppercase;
                  letter-spacing:1px;">Viewing period</div>
      <div style="font-size:20px;font-weight:700;">{year_range[0]}–{year_range[1]}</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊 Production & Value",
    "🌍 Geographic & Species",
    "🏛️ Permitting & Social",
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
                    '<span class="tag">G4-OP1 · G4-OP2 · G4-OP3 · G4-OP4 · G4-OP5</span>',
                    unsafe_allow_html=True)

        c1,c2,c3,c4,c5 = st.columns(5)
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

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts row 1 ─────────────────────────────────────────────────────────
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
            fig.update_annotations(font=dict(size=12, color=C["txt"]))
            fig.update_xaxes(tickfont=AXIS_FONT, gridcolor=GRID_COLOR,
                             linecolor=AXIS_LINE["color"])
            fig.update_yaxes(tickfont=AXIS_FONT, gridcolor=GRID_COLOR,
                             linecolor=AXIS_LINE["color"])
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
            st.plotly_chart(fig2, use_container_width=True)

        # Charts row 2 ─────────────────────────────────────────────────────────
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
                    line=dict(color=C["t100"], width=2.5))
            fig4.add_trace(go.Scatter(
                x=sp_price["price_per_kg"], y=sp_price["species"],
                mode="markers",
                marker=dict(size=13, color=C["t400"],
                            line=dict(color=C["t800"], width=1.5)),
                text=sp_price["price_per_kg"].round(2).astype(str) + " $/kg",
                textposition="middle right",
                textfont=dict(size=11, color=C["txt"]),
                name="$/kg",
            ))
            fig4.update_layout(**base_layout(310, dict(l=4,r=80,t=16,b=8)), showlegend=False)
            fig4 = style_axes(fig4, xtitle="USD per kg")
            fig4.update_yaxes(tickfont=dict(size=11, color=C["txt"]))
            st.plotly_chart(fig4, use_container_width=True)

        # ASFIS species diversity ───────────────────────────────────────────────
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
        st.plotly_chart(fig5, use_container_width=True)

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
                    '<span class="tag">KPI-7 · KPI-10 · KPI-17</span>',
                    unsafe_allow_html=True)

        gc1,gc2,gc3,gc4 = st.columns(4)
        card(gc1, "KPI-7 · Asia Share",          f"{asia_pct:.1f}%",
             f"of global {LY} production")
        card(gc2, "KPI-7 · Active Countries",    str(n_cnt),
             f"producing nations · {LY}")
        card(gc3, "KPI-17 · Upper-Mid Income",   f"{um_pct:.1f}%",
             f"High-income (Canada tier): {hi_pct:.1f}%")
        card(gc4, "KPI-10 · Active Species",
             str(int(faq[(faq["period"]==LY)&(faq["value"]>0)]["seaweed_name"].nunique())),
             f"farmed varieties · {LY}", accent="amber")

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
                textfont=dict(size=12, color=C["txt"]),
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
            st.plotly_chart(fig6, use_container_width=True)

        with gb:
            st.markdown(f'<div class="sec-head">KPI-10 — Top {top_n} Species by Volume</div>',
                        unsafe_allow_html=True)
            sp_df = (faq.groupby("seaweed_name")["value"].sum().reset_index()
                     .nlargest(top_n, "value").sort_values("value", ascending=True))
            sp_df["tm"] = (sp_df["value"] / 1e6).round(2)
            bc = (SP_PAL * 3)[:len(sp_df)]; bc.reverse()

            fig7 = go.Figure(go.Bar(
                x=sp_df["tm"], y=sp_df["seaweed_name"], orientation="h",
                marker=dict(color=bc, line=dict(color="white", width=0.5)),
                text=sp_df["tm"].apply(lambda v: f"{v:.1f}M t"),
                textposition="outside",
                textfont=dict(size=11, color=C["txt"]),
            ))
            fig7.update_layout(**base_layout(320, dict(l=4,r=80,t=10,b=4)), legend=LEGEND_TOP)
            fig7 = style_axes(fig7, xtitle="Million tonnes (cumulative)")
            fig7.update_yaxes(tickfont=dict(size=11, color=C["txt"]))
            st.plotly_chart(fig7, use_container_width=True)

        gi, gj = st.columns(2)

        with gi:
            st.markdown('<div class="sec-head">KPI-17 — Production by Income Group</div>',
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
            st.plotly_chart(fig8, use_container_width=True)

        with gj:
            st.markdown('<div class="sec-head">Species Composition Shift (Top 6)</div>',
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
            st.plotly_chart(fig9, use_container_width=True)

        with st.expander("📋 Geographic & Species Data Tables"):
            tg1, tg2, tg3 = st.tabs(["Continental Share","Top Species","Income Group"])
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


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PERMITTING & SOCIAL
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<span class="tag sim">🟡 Simulated data</span>'
                '<span class="tag ext">Sources: DFO · BC Gov · CIRNAC · Stats Canada</span>',
                unsafe_allow_html=True)
    st.caption("Simulated from public DFO/CIRNAC reports — replace with scraped data.")

    pf = sim_perm[(sim_perm["year"]>=year_range[0])&(sim_perm["year"]<=year_range[1])]
    sf = sim_social[(sim_social["year"]>=year_range[0])&(sim_social["year"]<=year_range[1])]
    pf_ly = pf[pf["year"]==pf["year"].max()] if not pf.empty else pd.DataFrame()
    sf_ly = sf[sf["year"]==sf["year"].max()] if not sf.empty else pd.DataFrame()

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
        st.plotly_chart(fpm2, use_container_width=True)

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
        st.plotly_chart(fso, use_container_width=True)

    with ss2:
        st.markdown('<div class="sec-head">G4-SO5 — Total Training Provided</div>',
                    unsafe_allow_html=True)
        ftr = go.Figure(go.Bar(
            x=sf["year"], y=sf["trained_total"],
            marker_color=[C["t400"] if y < 2022 else C["t600"] for y in sf["year"]],
            text=sf["trained_total"],
            textposition="outside",
            textfont=dict(size=11, color=C["txt"]),
        ))
        ftr.update_layout(**base_layout(275, dict(l=4,r=4,t=10,b=30)), showlegend=False)
        ftr = style_axes(ftr, ytitle="Individuals trained")
        st.plotly_chart(ftr, use_container_width=True)

    with st.expander("📋 Permitting & Social Data Tables"):
        tp1, tp2 = st.tabs(["Permitting","Social / Indigenous"])
        with tp1:
            st.dataframe(pf.rename(columns={
                "year":"Year","ada_ha":"ADA (ha)",
                "permitted_farms":"Permitted Farms","aoa_count":"AOAs",
                "compliance_pct":"Compliance (%)",
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
        """)


# ─────────────────────────────────────────────────────────────────────────────
# GROQ SYSTEM PROMPTS
# ─────────────────────────────────────────────────────────────────────────────
def build_snapshot():
    if not data_ok:
        return {}
    cont_s = (fgp[fgp["period"]==LY]
              .groupby("continent_group_en")["value"].sum()
              .sort_values(ascending=False))
    cont_lines = "\n".join([f"  {k}: {v/cont_s.sum()*100:.1f}%"
                            for k,v in cont_s.items() if k != "Unknown"])
    sp5 = (faq.groupby("seaweed_name")["value"].sum()
           .sort_values(ascending=False).head(5))
    sp_lines = "\n".join([f"  {k}: {v/1e6:.1f}M t" for k,v in sp5.items()])
    ig_s = (fgp[fgp["period"]==LY]
            .groupby("ecoclass_group_en")["value"].sum()
            .sort_values(ascending=False))
    ig_lines = "\n".join([f"  {k}: {v/ig_s.sum()*100:.1f}%"
                          for k,v in ig_s.items()])
    return dict(
        LY=LY, yr_range=year_range, cagr_win=cagr_win,
        prod_tot=prod_tot, yoy_prod=yoy_prod, cagr_prod=cagr_prod,
        aq_tot=aq_tot, wc_tot=wc_tot,
        sp_cult=sp_cult, sp_wild=sp_wild, sp_total=sp_total,
        val_tot=val_tot, yoy_val=yoy_val, avg_price_kg=avg_price_kg,
        cont_lines=cont_lines, sp_lines=sp_lines, ig_lines=ig_lines,
    )

def build_system_prompt(is_uc1):
    base = (
        "You are the PSIA (Pacific Seaweed Industry Association) AI assistant "
        "on an analytics dashboard. Help stakeholders and researchers understand "
        "the global seaweed industry. Be concise and data-specific. "
        "2–4 sentences for simple questions."
    )
    if is_uc1:
        return base + """

INDUSTRY KNOWLEDGE:
- Global production ~40M t/year (2024), 97% from Asia; China ~56%
- 88% farmed; 12% wild capture; CAGR ~2.9%/decade
- Aquaculture value ~$20B USD; Canadian market ~CAD $50M
- Top species: Japanese kelp, Eucheuma, Cottoni, Gracilaria, Wakame, Nori
- Applications: Food 40% · Cosmetics 20% · Agriculture 18% · Pharma 12% · Biofuel 10%
- Seaweed needs no freshwater, fertilizer, or land; sequesters CO2
- Canada = high-income tier, only 7% of global production — large opportunity
- PSIA values: Education, Economic Opportunity, Research, Innovation, Community
- Permitting: BC Aquaculture Act, DFO licences, ~755 farms, 54 AOAs, 95% compliance
- Indigenous: CIRNAC programs, ~24 agreements, ~207 employed, ~174 trained (2024)
"""
    snap = build_snapshot()
    if not snap:
        return base + "\nData not loaded."
    return base + f"""

LIVE DASHBOARD DATA ({snap['yr_range'][0]}–{snap['yr_range'][1]}):
Production ({snap['LY']}): {snap['prod_tot']/1e6:.2f}M t · YoY {snap['yoy_prod']:.1f}% · CAGR {snap['cagr_prod']:.1f}%
  Farmed: {snap['aq_tot']/1e6:.2f}M t · Wild: {snap['wc_tot']/1e6:.3f}M t
  ASFIS species: {snap['sp_total']} (farmed:{snap['sp_cult']}, wild:{snap['sp_wild']})
  Avg price proxy: ${snap['avg_price_kg']:.2f}/kg
Value ({snap['LY']}): USD ${snap['val_tot']/1e6:.1f}B · YoY {snap['yoy_val']:.1f}%
Continents:
{snap['cont_lines']}
Top species:
{snap['sp_lines']}
Income groups:
{snap['ig_lines']}
Permitting (sim): ADA 17,900ha · farms 755 · AOAs 54 · compliance 95.3%
Social (sim): agreements 24 · employed 207 · trained 174 · total trained 504
"""


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thinking"  not in st.session_state:
    st.session_state.thinking = False
if "chat_open" not in st.session_state:
    st.session_state.chat_open = True
if "chat_mode" not in st.session_state:
    st.session_state.chat_mode = "UC1"

# ─────────────────────────────────────────────────────────────────────────────
# COLLAPSIBLE CHAT PANEL  — bottom of main content
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)

# ── Toggle bar ────────────────────────────────────────────────────────────────
icon  = "▼" if not st.session_state.chat_open else "▲"
label = f"💬  AI Assistant  {icon}  {'— click to open' if not st.session_state.chat_open else '— click to close'}"
n_msg = len(st.session_state.messages)
badge = f"  ·  {n_msg} message{'s' if n_msg != 1 else ''}" if n_msg > 0 else "  ·  Powered by Groq llama-3.3-70b"

st.markdown(f"""
<div style="background:{C['t800']};color:white;padding:11px 20px;
            border-radius:{'10px 10px 0 0' if st.session_state.chat_open else '10px'};
            display:flex;justify-content:space-between;align-items:center;
            font-family:Arial,sans-serif;font-size:13.5px;font-weight:600;">
  <span>{label}</span>
  <span style="font-size:11px;opacity:0.65;font-weight:400;">{badge}</span>
</div>""", unsafe_allow_html=True)

if st.button("🟢 Click here to open / close the AI Assistant",
             key="chat_toggle",
             help="Toggle the AI chat panel",
             use_container_width=True):
    st.session_state.chat_open = not st.session_state.chat_open
    st.rerun()

# ── Chat body (only shown when open) ─────────────────────────────────────────
if st.session_state.chat_open:
    with st.container():
        st.markdown(f"""
        <div style="background:{C['t900']};border-radius:0 0 10px 10px;
                    border:1px solid {C['t800']};border-top:none;padding:16px 20px 18px;">
        """, unsafe_allow_html=True)

        # Mode + clear row
        col_mode, col_clr = st.columns([4, 1])
        with col_mode:
            mode_choice = st.radio(
                "Chat mode",
                ["💬 UC1 — Industry Info", "📊 UC2 — Live Data Query"],
                index=0 if st.session_state.chat_mode == "UC1" else 1,
                horizontal=True,
                label_visibility="collapsed",
            )
            st.session_state.chat_mode = "UC1" if "UC1" in mode_choice else "UC2"
        with col_clr:
            if st.button("🗑️ Clear", key="clear_chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

        is_uc1 = st.session_state.chat_mode == "UC1"
        st.markdown(
            f"<div style='font-size:11px;color:rgba(255,255,255,0.55);"
            f"font-family:Arial,sans-serif;margin-bottom:10px;'>"
            f"{'📚 Industry knowledge — species · markets · PSIA values · permitting' if is_uc1 else '📊 Live data — production · value · geographic · species · social KPIs'}"
            "</div>", unsafe_allow_html=True)

        # Chat history
        history_html = '<div style="max-height:380px;overflow-y:auto;padding-right:6px;margin-bottom:12px;">'
        if len(st.session_state.messages) == 0:
            history_html += (
                '<div style="text-align:center;padding:30px 20px;">'
                f'<div style="font-size:28px;margin-bottom:8px;">🌿</div>'
                f'<div style="font-size:13px;color:rgba(255,255,255,0.6);'
                f'font-family:Arial,sans-serif;line-height:1.6;">'
                'Hi! I\'m your PSIA seaweed industry AI assistant.<br>'
                'Ask me about production data, species, market trends, or PSIA initiatives.'
                '</div></div>'
            )
        for m in st.session_state.messages:
            is_user = m["role"] == "user"
            bubble_bg  = C["t400"] if is_user else "rgba(255,255,255,0.10)"
            txt_color  = "white"
            b_radius   = "14px 14px 2px 14px" if is_user else "14px 14px 14px 2px"
            margin     = "margin-left:80px" if is_user else "margin-right:80px"
            icon_align = "text-align:right;" if is_user else ""
            icon       = "🧑" if is_user else "🌿"
            history_html += (
                f'<div style="font-size:9.5px;color:rgba(255,255,255,0.38);'
                f'font-family:Arial,sans-serif;margin-bottom:3px;{icon_align}">{icon}</div>'
                f'<div style="background:{bubble_bg};color:{txt_color};'
                f'border-radius:{b_radius};padding:10px 14px;margin-bottom:8px;'
                f'{margin};font-size:13px;font-family:Arial,sans-serif;'
                f'line-height:1.55;word-wrap:break-word;">{m["content"]}</div>'
            )
        if st.session_state.thinking:
            history_html += (
                '<div style="color:rgba(255,255,255,0.45);font-style:italic;'
                'font-family:Arial,sans-serif;font-size:12px;padding:6px 14px;">'
                '⏳ Thinking...</div>'
            )
        history_html += "</div>"
        st.markdown(history_html, unsafe_allow_html=True)

        # Suggestion chips (only on empty chat)
        SUGG = {
            True:  ["What drives global seaweed market growth?",
                    "Which seaweed species are most commercially valuable?",
                    "What opportunities exist for Canada's seaweed sector?"],
            False: [f"What is total production in {LY if data_ok else '2024'}?",
                    "Which continent produces the most seaweed?",
                    "What is the average implied price per kg?"],
        }
        if len(st.session_state.messages) == 0:
            st.markdown(
                "<div style='font-size:11px;color:rgba(255,255,255,0.5);"
                "font-family:Arial,sans-serif;margin-bottom:6px;'>💡 Try asking:</div>",
                unsafe_allow_html=True)
            s1, s2, s3 = st.columns(3)
            for col, sugg in zip([s1, s2, s3], SUGG[is_uc1]):
                with col:
                    if st.button(sugg, key=f"sugg_{sugg}", use_container_width=True):
                        st.session_state["_pending"] = sugg
                        st.rerun()

        # Chat input
        user_input = st.chat_input("Ask about the seaweed industry or dashboard data...")
        pending    = st.session_state.pop("_pending", None)
        prompt     = pending or user_input

        if prompt and not st.session_state.thinking:
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.thinking = True
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

# ── Groq API call (fires on rerun when thinking=True) ────────────────────────
if st.session_state.get("thinking", False):
    try:
        client = Groq(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=600,
            messages=[
                {"role": "system",
                 "content": build_system_prompt(st.session_state.chat_mode == "UC1")},
                *[{"role": m["role"], "content": m["content"]}
                  for m in st.session_state.messages],
            ],
        )
        reply = resp.choices[0].message.content
    except Exception as e:
        err = str(e).lower()
        if "auth" in err or "api_key" in err or "invalid" in err:
            reply = "❌ Invalid API key. Check GROQ_API_KEY in the script."
        elif "rate" in err:
            reply = "⏳ Rate limit hit. Wait a few seconds and try again."
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
    "<div style='text-align:center;font-size:10.5px;color:#999;"
    "font-family:Arial,sans-serif;padding:6px 0 4px;'>"
    "PSIA Seaweed Analytics v4.0  ·  "
    "🟢 FAO FishStat (real)  ·  🟡 DFO/CIRNAC (simulated)  ·  "
    "⚡ AI powered by Groq llama-3.3-70b (free)"
    "</div>", unsafe_allow_html=True)

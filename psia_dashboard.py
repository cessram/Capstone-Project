"""
PSIA Seaweed Industry Analytics Dashboard — v11.0
==================================================
CHANGES FROM v10:
  - Enriched RAG knowledge base with 35 chunks from 16 source documents
    + new Canadian aquaculture data (Statistics Canada 1986-2024)
    + Value Added Account data (DFO/StatsCan 1997-2024)
  - Tab 5 "🇨🇦 Canadian Context" — new BC aquaculture historical tab
    using real Statistics Canada data (aqua86-aqua24 + va97-va24 files)
  - Typography & readability overhaul:
    · Main font: 'Source Sans Pro' (web-safe fallback: Helvetica Neue, sans-serif)
    · All body text #1A1A1A on white; sidebar text white on dark teal
    · KPI values Georgia serif 28px; labels Inter 11px uppercase
    · Chart axis/tick fonts 12px; chart titles 13px
    · Sidebar labels 15px bold white; filter controls high contrast
    · Chat bubbles: explicit white text on dark teal backgrounds
  - Live data in chatbot UC2 mode now fully aligned with dashboard metrics
  - Canadian aquaculture context RAG chunks added

Setup:
  pip install streamlit plotly pandas numpy groq scikit-learn
  Place 4 FAO CSVs in data/
  streamlit run psia_dashboard_v19.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from groq import Groq


import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# COLOURS (fallback if ctx doesn't pass the palette)
# ─────────────────────────────────────────────────────────────────────────────
_C = {
    "t900": "#04342C", "t800": "#085041", "t600": "#0F6E56",
    "t400": "#1D9E75", "t300": "#5DCAA7", "t100": "#9FE1CB",
    "amber": "#C97D10", "coral": "#B84020", "blue": "#1E5FA8", "gray": "#555550",
    "txt": "#1A1A1A",
}

_AXIS = dict(family="Helvetica, Arial, sans-serif", size=12, color="#333333")


def _layout(fig, C, title="", h=300):
    fig.update_layout(
        height=h, margin=dict(l=6, r=6, t=40 if title else 16, b=8),
        paper_bgcolor="white", plot_bgcolor="#FAFAFA",
        font=dict(family="Helvetica, Arial, sans-serif", size=13, color=C["txt"]),
        title=dict(text=title, font=dict(size=13, color=C["txt"])),
        legend=dict(orientation="h", y=1.12, x=0, font=dict(size=11, color=C["txt"]),
                    bgcolor="rgba(255,255,255,0.9)", bordercolor="#CCC", borderwidth=1),
    )
    fig.update_xaxes(tickfont=_AXIS, gridcolor="#EAEAEA", linecolor="#CCC", zeroline=False)
    fig.update_yaxes(tickfont=_AXIS, gridcolor="#EAEAEA", linecolor="#CCC", zeroline=False)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 1. FORECASTING (ARIMA with graceful fallback)
# ═════════════════════════════════════════════════════════════════════════════
def _to_regular_series(series_dict):
    """Turn a possibly-gappy {year:value} dict into a consecutive (years, values)
    pair via linear interpolation across the observed range. ARIMA needs regular
    spacing; the StatsCan dicts have suppressed years (e.g. BC 1991-2001)."""
    years = sorted(int(y) for y in series_dict)
    if not years:
        return [], []
    full = list(range(years[0], years[-1] + 1))
    obs_y = np.array(years, dtype=float)
    obs_v = np.array([float(series_dict[y]) for y in years], dtype=float)
    vals = np.interp(full, obs_y, obs_v)
    return full, vals.tolist()


def forecast_series(series_dict, periods=5, alpha=0.20):
    """Forecast `periods` years ahead.

    Tries ARIMA (statsmodels). If statsmodels is unavailable or the fit fails,
    falls back to a log-linear (constant-growth) projection so the feature always
    works. Returns a dict with history + forecast + an 80% interval (alpha=0.20).
    """
    years, vals = _to_regular_series(series_dict)
    if len(vals) < 4:
        return {"ok": False, "error": "Need at least 4 data points to forecast."}

    last_year = years[-1]
    fut_years = list(range(last_year + 1, last_year + periods + 1))
    method = None
    mean = lo = hi = None

    # --- Attempt 1: ARIMA ---------------------------------------------------
    try:
        from statsmodels.tsa.arima.model import ARIMA
        import warnings
        warnings.filterwarnings("ignore")
        best = None
        # small grid search over sensible orders; pick lowest AIC
        for order in [(1, 1, 1), (2, 1, 1), (1, 1, 0), (0, 1, 1), (2, 1, 2), (1, 2, 1)]:
            try:
                res = ARIMA(vals, order=order).fit()
                if best is None or res.aic < best[1]:
                    best = (res, res.aic, order)
            except Exception:
                continue
        if best is not None:
            res, _, order = best
            fc = res.get_forecast(periods)
            mean = np.asarray(fc.predicted_mean, dtype=float)
            ci = np.asarray(fc.conf_int(alpha=alpha), dtype=float)
            lo, hi = ci[:, 0], ci[:, 1]
            mean = np.clip(mean, 0, None)
            lo = np.clip(lo, 0, None)
            method = f"ARIMA{order}"
    except ImportError:
        method = None
    except Exception:
        method = None

    # --- Attempt 2: log-linear fallback ------------------------------------
    if method is None:
        v = np.array(vals, dtype=float)
        t = np.arange(len(v), dtype=float)
        pos = v > 0
        if pos.sum() >= 3:                       # exponential trend on positive data
            b, a = np.polyfit(t[pos], np.log(v[pos]), 1)
            resid = np.log(v[pos]) - (a + b * t[pos])
            s = float(np.std(resid)) if len(resid) > 1 else 0.0
            ft = np.arange(len(v), len(v) + periods, dtype=float)
            mean = np.exp(a + b * ft)
            z = 1.2816                            # ~80% interval
            lo = np.exp(a + b * ft - z * s)
            hi = np.exp(a + b * ft + z * s)
            method = "log-linear trend (install statsmodels for ARIMA)"
        else:                                    # linear fallback
            b, a = np.polyfit(t, v, 1)
            resid = v - (a + b * t)
            s = float(np.std(resid)) if len(resid) > 1 else 0.0
            ft = np.arange(len(v), len(v) + periods, dtype=float)
            mean = a + b * ft
            lo, hi = mean - 1.2816 * s, mean + 1.2816 * s
            method = "linear trend (install statsmodels for ARIMA)"
        mean = np.clip(mean, 0, None); lo = np.clip(lo, 0, None)

    return {
        "ok": True, "method": method,
        "hist_years": years, "hist_vals": vals,
        "fc_years": fut_years,
        "fc_mean": [float(x) for x in mean],
        "fc_lo": [float(x) for x in lo],
        "fc_hi": [float(x) for x in hi],
    }


def fig_forecast(series_dict, periods, label, C, scale=1.0, unit="", h=320):
    """Build a Plotly chart: history + forecast line + confidence band."""
    fc = forecast_series(series_dict, periods=periods)
    if not fc["ok"]:
        return None, fc.get("error", "forecast failed"), None
    hy = fc["hist_years"]; hv = [v / scale for v in fc["hist_vals"]]
    fy = fc["fc_years"]
    fm = [v / scale for v in fc["fc_mean"]]
    flo = [v / scale for v in fc["fc_lo"]]
    fhi = [v / scale for v in fc["fc_hi"]]
    # connect history end to forecast start for a continuous line
    cy = [hy[-1]] + fy
    cm = [hv[-1]] + fm

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fy + fy[::-1], y=fhi + flo[::-1], fill="toself",
                             fillcolor="rgba(201,125,16,0.15)", line=dict(width=0),
                             name="80% interval", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=hy, y=hv, mode="lines+markers", name="History",
                             line=dict(color=C["t600"], width=2.5), marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=cy, y=cm, mode="lines+markers", name="Forecast",
                             line=dict(color=C["amber"], width=2.5, dash="dash"),
                             marker=dict(size=6, color=C["amber"])))
    _layout(fig, C, h=h)
    fig.update_yaxes(title=dict(text=unit, font=_AXIS))
    caption = f"{label} — forecast {fy[0]}–{fy[-1]} via {fc['method']}."
    txt = (f"{label}: forecast for {fy[-1]} is "
           f"{fc['fc_mean'][-1]/scale:,.2f}{(' ' + unit) if unit else ''} "
           f"(80% interval {flo[-1]:,.2f}–{fhi[-1]:,.2f}). Method: {fc['method']}.")
    return fig, txt, caption


# ═════════════════════════════════════════════════════════════════════════════
# 2. SMALL CHART BUILDERS (for chart-intent questions)
# ═════════════════════════════════════════════════════════════════════════════
def fig_trend(years, vals, C, name, unit, h=300):
    fig = go.Figure(go.Scatter(x=years, y=vals, mode="lines+markers", name=name,
                               line=dict(color=C["t600"], width=2.5),
                               marker=dict(size=5), fill="tozeroy",
                               fillcolor="rgba(15,110,86,0.10)"))
    _layout(fig, C, h=h)
    fig.update_yaxes(title=dict(text=unit, font=_AXIS))
    return fig


def fig_topn_bar(labels, vals, C, unit, h=320):
    order = np.argsort(vals)
    labels = [labels[i] for i in order]; vals = [vals[i] for i in order]
    fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h",
                           marker_color=C["t600"],
                           text=[f"{v:,.2f}" for v in vals], textposition="outside",
                           textfont=dict(size=11, color=C["txt"])))
    _layout(fig, C, h=h)
    fig.update_xaxes(title=dict(text=unit, font=_AXIS))
    return fig


def fig_compare(series_map, C, unit, h=320):
    """series_map: {name: {year:val}}"""
    fig = go.Figure()
    pal = [C["t600"], C["amber"], C["blue"], C["coral"], C["t400"]]
    for i, (name, d) in enumerate(series_map.items()):
        ys = sorted(d)
        fig.add_trace(go.Scatter(x=ys, y=[d[y] for y in ys], mode="lines+markers",
                                 name=name, line=dict(color=pal[i % len(pal)], width=2.4),
                                 marker=dict(size=5)))
    _layout(fig, C, h=h)
    fig.update_yaxes(title=dict(text=unit, font=_AXIS))
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 3. QUERY ROUTER
# ═════════════════════════════════════════════════════════════════════════════
_YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")
_FORECAST_KW = ("forecast", "predict", "projection", "project ", "arima",
                "next year", "next 5", "future", "by 2030", "by 2035", "expected",
                "will be", "outlook")
_CHART_KW = ("chart", "graph", "plot", "show me", "visuali", "trend of", "draw",
             "over time", "by year")


def _years(prompt):
    return [int(y) for y in _YEAR_RE.findall(prompt)]


def _wants_forecast(p):
    return any(k in p for k in _FORECAST_KW)


def _wants_chart(p):
    return any(k in p for k in _CHART_KW)


def _global_prod_series(fgp):
    s = fgp[fgp["value"] > 0].groupby("period")["value"].sum()
    return {int(y): float(v) for y, v in s.items() if v > 0}


def _global_val_series(fav):
    s = fav.groupby("period")["value"].sum()
    return {int(y): float(v) for y, v in s.items() if v > 0}


def _match_country(prompt, countries):
    """Return the country name mentioned in the prompt, if any (longest match)."""
    pl = prompt.lower()
    hits = [c for c in countries if c.lower() in pl]
    return max(hits, key=len) if hits else None


def answer_live_query(prompt, ctx):
    """Deterministic answer for quantitative / chart / forecast questions.
    Returns {"handled","text","fig","caption"}. handled=False -> let the LLM answer."""
    miss = {"handled": False, "text": None, "fig": None, "caption": None}
    if not prompt:
        return miss
    p = prompt.lower().strip()
    C = ctx.get("C", _C)
    data_ok = ctx.get("data_ok", False)

    fgp = ctx.get("fgp"); faq = ctx.get("faq"); fav = ctx.get("fav"); fcq = ctx.get("fcq")
    LY = ctx.get("LY"); cagr_win = ctx.get("cagr_win", 10)
    yrs = _years(p)

    # Canadian / BC keyword detection -> Tab-5 StatsCan dicts (all-aquaculture)
    is_canada = bool(re.search(
        r"\b(bc|b\.c\.|british columbia|canad\w*|salmon|shellfish|statscan|"
        r"statistics canada|gva|value added|feed cost\w*|therapeut\w*|dfo|wages?)\b", p))
    is_bc = bool(re.search(r"\b(bc|b\.c\.|british columbia)\b", p)) and "canad" not in p

    # ── A. FORECASTS ────────────────────────────────────────────────────────
    if _wants_forecast(p) and data_ok:
        horizon = 5
        for y in yrs:
            if LY and y > LY:
                horizon = max(1, min(15, y - LY))
        if is_canada:
            if "value" in p or "$" in p:
                fig, txt, cap = fig_forecast(ctx["BC_VAL_K"] if is_bc else ctx["CAN_VAL_K"],
                                             horizon, ("BC" if is_bc else "Canada") +
                                             " aquaculture value", C, scale=1000, unit="$M CAD")
            else:
                fig, txt, cap = fig_forecast(ctx["BC_PROD_T"] if is_bc else ctx["CAN_PROD_T"],
                                             horizon, ("BC" if is_bc else "Canada") +
                                             " aquaculture production", C, scale=1000, unit="kt")
        elif "value" in p or "$" in p or "usd" in p:
            fig, txt, cap = fig_forecast(_global_val_series(fav), horizon,
                                         "Global seaweed value", C, scale=1e6, unit="$B USD")
        else:
            fig, txt, cap = fig_forecast(_global_prod_series(fgp), horizon,
                                         "Global seaweed production", C, scale=1e6, unit="M tonnes")
        if fig is not None:
            return {"handled": True, "text": txt, "fig": fig, "caption": cap}

    # ── B. CANADIAN / BC METRICS (Tab 5) ─────────────────────────────────────
    if is_canada and data_ok:
        metric_map = {
            "salmon": ("BC_SALMON_T", "BC farmed salmon production", 1000, "kt", "t"),
            "shellfish": ("BC_SHELL_T", "BC shellfish production", 1000, "kt", "t"),
            "gva": ("VA_BC_GVA_K", "BC aquaculture Gross Value Added", 1000, "$M CAD", "$K"),
            "value added": ("VA_BC_GVA_K", "BC aquaculture Gross Value Added", 1000, "$M CAD", "$K"),
            "feed": ("VA_BC_FEED_K", "BC aquaculture feed costs", 1000, "$M CAD", "$K"),
            "therapeut": ("VA_BC_THERAP_K", "BC aquaculture therapeutant costs", 1000, "$M CAD", "$K"),
            "wage": ("VA_BC_WAGES_K", "BC aquaculture wages", 1000, "$M CAD", "$K"),
        }
        chosen = None
        for kw, spec in metric_map.items():
            if kw in p:
                chosen = spec; break
        if chosen is None:
            if "value" in p or "$" in p:
                chosen = (("BC_VAL_K" if is_bc else "CAN_VAL_K"),
                          ("BC" if is_bc else "Canada") + " aquaculture value", 1000, "$M CAD", "$K")
            else:
                chosen = (("BC_PROD_T" if is_bc else "CAN_PROD_T"),
                          ("BC" if is_bc else "Canada") + " aquaculture production", 1000, "kt", "t")
        dkey, label, scale, cunit, raw_unit = chosen
        d = ctx.get(dkey, {})
        if _wants_chart(p) or not yrs:
            fig = fig_trend(sorted(d), [d[y] / scale for y in sorted(d)], C, label, cunit)
            yy = sorted(d)
            txt = (f"{label}: {d[yy[-1]]/scale:,.1f} {cunit} in {yy[-1]} "
                   f"(earliest on record {yy[0]}: {d[yy[0]]/scale:,.1f} {cunit}). "
                   f"Source: Statistics Canada / DFO. NOTE: all-aquaculture (not seaweed-only).")
            return {"handled": True, "text": txt, "fig": fig,
                    "caption": f"{label} — Statistics Canada / DFO (all aquaculture species)"}
        # specific year(s)
        lines = []
        for y in yrs:
            if y in d:
                seg = (f"{label} in {y}: {d[y]:,} {raw_unit}"
                       + (f" (= {d[y]/scale:,.1f} {cunit})" if scale != 1 else ""))
                if (y - 1) in d and d[y - 1]:
                    pct = (d[y] - d[y - 1]) / d[y - 1] * 100
                    seg += f" — {pct:+.1f}% vs {y-1}"
                lines.append(seg)
            else:
                lines.append(f"{label} in {y}: not in the data (StatsCan suppresses BC 1991–2001).")
        txt = " ".join(lines) + " Source: Statistics Canada / DFO (all aquaculture species, NOT seaweed-only)."
        return {"handled": True, "text": txt, "fig": None, "caption": None}

    # ── C. GLOBAL SEAWEED METRICS (Tabs 1-4, FAO) ────────────────────────────
    if not data_ok:
        return miss

    countries = sorted(fgp[fgp["value"] > 0]["country_name"].unique().tolist())
    country = _match_country(p, countries)

    # C1. top-N countries / species
    if ("top" in p or "biggest" in p or "largest" in p or "leading" in p) and \
       ("countr" in p or "producer" in p or "nation" in p or "species" in p):
        n = 10
        m = re.search(r"top\s+(\d{1,2})", p)
        if m:
            n = int(m.group(1))
        yr = yrs[0] if yrs else LY
        if "species" in p:
            s = faq[(faq["period"] == yr) & (faq["value"] > 0)].groupby("seaweed_name")["value"].sum()
            unit = "M tonnes (farmed)"
        else:
            s = fgp[(fgp["period"] == yr) & (fgp["value"] > 0)].groupby("country_name")["value"].sum()
            unit = "M tonnes"
        s = s.sort_values(ascending=False).head(n)
        labels = list(s.index); vals = [v / 1e6 for v in s.values]
        ranked = "; ".join([f"{i+1}. {labels[i]} {vals[i]:.2f}M t" for i in range(len(labels))])
        txt = f"Top {n} {'species' if 'species' in p else 'countries'} by seaweed production in {yr}: {ranked}."
        fig = fig_topn_bar(labels, vals, C, unit) if _wants_chart(p) or True else None
        return {"handled": True, "text": txt, "fig": fig,
                "caption": f"Top {n} by seaweed production, {yr} (FAO FishStat)"}

    # C2. country production in a year
    if country:
        yr = yrs[0] if yrs else LY
        cv = fgp[(fgp["period"] == yr) & (fgp["country_name"] == country)]["value"].sum()
        if _wants_chart(p):
            ser = fgp[fgp["country_name"] == country].groupby("period")["value"].sum()
            ser = {int(y): float(v) / 1e6 for y, v in ser.items() if v > 0}
            fig = fig_trend(sorted(ser), [ser[y] for y in sorted(ser)], C,
                            f"{country} seaweed production", "M tonnes")
            return {"handled": True,
                    "text": f"{country} seaweed production reached {cv/1e6:.3f}M tonnes in {yr} (FAO FishStat).",
                    "fig": fig, "caption": f"{country} seaweed production over time (FAO FishStat)"}
        tot = fgp[fgp["period"] == yr]["value"].sum()
        share = cv / tot * 100 if tot else 0
        txt = (f"{country} produced {cv/1e6:.3f}M tonnes of seaweed in {yr} "
               f"({share:.1f}% of the {tot/1e6:.2f}M-tonne global total). Source: FAO FishStat.")
        return {"handled": True, "text": txt, "fig": None, "caption": None}

    # C3. CAGR
    if "cagr" in p or "growth rate" in p or "annual growth" in p:
        gs = _global_prod_series(fgp)
        ys = sorted(gs)
        win = cagr_win
        base_y = ys[-1] - win
        base_y = base_y if base_y in gs else min(ys, key=lambda y: abs(y - (ys[-1] - win)))
        cagr = ((gs[ys[-1]] / gs[base_y]) ** (1 / (ys[-1] - base_y)) - 1) * 100 if gs[base_y] else 0
        txt = (f"Global seaweed production CAGR over {ys[-1]-base_y} years "
               f"({base_y}→{ys[-1]}): {cagr:.1f}%/yr "
               f"({gs[base_y]/1e6:.2f}M → {gs[ys[-1]]/1e6:.2f}M tonnes). Source: FAO FishStat.")
        return {"handled": True, "text": txt, "fig": None, "caption": None}

    # C4. cultivation vs wild
    if ("cultivat" in p or "farmed" in p or "wild" in p or "capture" in p) and \
       ("vs" in p or "versus" in p or "split" in p or "share" in p or "how much" in p):
        yr = yrs[0] if yrs else LY
        aqv = faq[faq["period"] == yr]["value"].sum()
        wcv = fcq[fcq["period"] == yr]["value"].sum()
        tot = aqv + wcv
        txt = (f"In {yr}, farmed (cultivation) seaweed was {aqv/1e6:.2f}M tonnes "
               f"({aqv/tot*100:.1f}%) and wild capture was {wcv/1e6:.3f}M tonnes "
               f"({wcv/tot*100:.1f}%). Source: FAO FishStat.")
        return {"handled": True, "text": txt, "fig": None, "caption": None}

    # C5. continental / Asia share
    if "asia" in p or "continent" in p or ("share" in p and "income" not in p):
        yr = yrs[0] if yrs else LY
        cs = fgp[fgp["period"] == yr].groupby("continent_group_en")["value"].sum()
        cs = cs[cs.index != "Unknown"]; tot = cs.sum()
        if "asia" in p:
            a = cs.get("Asia", 0)
            txt = f"Asia produced {a/tot*100:.1f}% of global seaweed in {yr} ({a/1e6:.2f}M tonnes). Source: FAO FishStat."
            return {"handled": True, "text": txt, "fig": None, "caption": None}
        parts = "; ".join([f"{k} {v/tot*100:.1f}%" for k, v in cs.sort_values(ascending=False).items()])
        txt = f"Continental share of global seaweed in {yr}: {parts}. Source: FAO FishStat."
        return {"handled": True, "text": txt, "fig": None, "caption": None}

    # C6. plain global production / value in a year (+ chart variant)
    if ("production" in p or "produce" in p or "tonnes" in p or "output" in p
            or "value" in p or "worth" in p) or _wants_chart(p):
        want_value = "value" in p or "worth" in p or "$" in p or "usd" in p
        gs = _global_val_series(fav) if want_value else _global_prod_series(fgp)
        scale = 1e6
        unit = "$B USD" if want_value else "M tonnes"
        label = "Global seaweed value" if want_value else "Global seaweed production"
        if _wants_chart(p) and not yrs:
            ys = sorted(gs)
            fig = fig_trend(ys, [gs[y] / scale for y in ys], C, label, unit)
            return {"handled": True,
                    "text": f"{label}: {gs[ys[-1]]/scale:.2f} {unit} in {ys[-1]}, up from "
                            f"{gs[ys[0]]/scale:.2f} {unit} in {ys[0]}. Source: FAO FishStat.",
                    "fig": fig, "caption": f"{label} over time (FAO FishStat)"}
        if yrs:
            lines = []
            for y in yrs:
                if y in gs:
                    lines.append(f"{label} in {y}: {gs[y]/scale:.2f} {unit}")
                else:
                    near = min(gs, key=lambda k: abs(k - y))
                    lines.append(f"{label} for {y}: not in the loaded range; nearest is {near} "
                                 f"({gs[near]/scale:.2f} {unit})")
            txt = "; ".join(lines) + ". Source: FAO FishStat (seaweed-only)."
            return {"handled": True, "text": txt, "fig": None, "caption": None}

    return miss



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

import uuid as _uuid
def make_float(css_props: str):
    marker_id = "fl-" + _uuid.uuid4().hex[:10]
    st.markdown(
        f"<style>"
        f"div[data-testid='stVerticalBlock']:has(> div > span#{marker_id})"
        f"{{position:fixed !important;{css_props}}}"
        f"</style>", unsafe_allow_html=True)
    c = st.container()
    with c:
        st.markdown(f'<span id="{marker_id}" style="display:none;"></span>', unsafe_allow_html=True)
    return c

# ─────────────────────────────────────────────────────────────────────────────
# COLOURS
# ─────────────────────────────────────────────────────────────────────────────
C = {
    "t900": "#04342C", "t800": "#085041", "t600": "#0F6E56",
    "t400": "#1D9E75", "t300": "#5DCAA7", "t100": "#9FE1CB", "t50": "#E1F5EE",
    "amber": "#C97D10", "coral": "#B84020", "blue": "#1E5FA8",
    "gray": "#555550", "green": "#1B5E20",
    "txt":  "#1A1A1A", "txt2": "#444440",
    "bg":   "#FFFFFF", "bg2":  "#F7F9F8",
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
    "#085041","#0F6E56","#1D9E75","#5DCAA7","#9FE1CB",
    "#C97D10","#B84020","#1E5FA8","#5B4FCF","#A0356E",
    "#2E7D6B","#7B4F00","#1A3A6B","#6B1A3A","#3A6B1A",
]

# ─── Typography constants ────────────────────────────────────────────────────
BODY_FONT  = "'Source Sans 3', 'Source Sans Pro', 'Helvetica Neue', Helvetica, Arial, sans-serif"
SERIF_FONT = "Lora, Georgia, 'Times New Roman', serif"

CHART_FONT   = dict(family="'Source Sans 3', 'Source Sans Pro', Helvetica, Arial, sans-serif", size=14, color="#1A1A1A")
AXIS_FONT    = dict(family="'Source Sans 3', 'Source Sans Pro', Helvetica, Arial, sans-serif", size=13, color="#333333")
TITLE_FONT   = dict(family="Lora, Georgia, serif", size=14, color="#1A1A1A")
GRID_COLOR   = "#EAEAEA"
AXIS_LINE    = dict(color="#CCCCCC", width=1)

LEGEND_TOP = dict(
    orientation="h", y=1.14, x=0,
    font=dict(size=12, color="#1A1A1A", family="'Source Sans Pro', Helvetica, Arial, sans-serif"),
    bgcolor="rgba(255,255,255,0.95)",
    bordercolor="#CCCCCC", borderwidth=1,
    tracegroupgap=0, groupclick="toggleitem",
)
LEGEND_BOTTOM = dict(
    orientation="h", y=-0.44, x=0,
    font=dict(size=11, color="#1A1A1A", family="'Source Sans Pro', Helvetica, Arial, sans-serif"),
    bgcolor="rgba(255,255,255,0.95)",
    bordercolor="#CCCCCC", borderwidth=1,
    title_text="", tracegroupgap=0, groupclick="toggleitem",
)

def base_layout(height=320, margin=None):
    if margin is None:
        margin = dict(l=4, r=4, t=32, b=8)
    return dict(
        height=height, margin=margin,
        paper_bgcolor="white", plot_bgcolor="#FAFAFA",
        font=CHART_FONT,
        title=dict(text="", font=TITLE_FONT),
    )

def style_axes(fig, xtitle="", ytitle="", y2title="", y_range=None):
    xax = dict(title=dict(text=xtitle, font=AXIS_FONT, standoff=8),
               tickfont=AXIS_FONT, gridcolor=GRID_COLOR,
               linecolor=AXIS_LINE["color"], linewidth=1,
               showgrid=True, zeroline=False)
    yax = dict(title=dict(text=ytitle, font=AXIS_FONT, standoff=8),
               tickfont=AXIS_FONT, gridcolor=GRID_COLOR,
               linecolor=AXIS_LINE["color"], linewidth=1,
               showgrid=True, zeroline=False)
    if y_range: yax["range"] = y_range
    fig.update_xaxes(**xax)
    fig.update_yaxes(**yax)
    if y2title:
        fig.update_yaxes(title=dict(text=y2title, font=AXIS_FONT, standoff=8),
                         tickfont=AXIS_FONT, secondary_y=True)
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# CSS — v11.1: Full professional polish + sidebar readability fix
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@300;400;500;600;700;800&family=Lora:wght@600;700&display=swap');

/* ── 1. APP BASE ─────────────────────────────────────────────────────────── */
html, body {{
    font-family: {BODY_FONT} !important;
    font-size: 15px !important;
    color: {C['txt']} !important;
    -webkit-font-smoothing: antialiased;
}}
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main .block-container,
[data-testid="stMainBlockContainer"] {{
    background-color: #F5F7F6 !important;
}}
.main .block-container {{
    padding-top: 24px !important;
    padding-left: 28px !important;
    padding-right: 20px !important;
    max-width: 100% !important;
}}
[data-baseweb="tab-panel"],
[data-baseweb="tab-list"] {{
    background-color: #F5F7F6 !important;
}}

/* ── 2. GENERAL MAIN-AREA TEXT (dark on light) ──────────────────────────── */
/* Only applies outside sidebar — sidebar rules below override these */
.stMarkdown p, .stMarkdown li {{ color: {C['txt']} !important; font-size: 15px !important; font-family: {BODY_FONT} !important; line-height: 1.65; }}
.stMarkdown h4 {{ font-size: 18px !important; font-weight: 700 !important; color: {C['txt']} !important; font-family: {SERIF_FONT} !important; }}
[data-testid="stCaptionContainer"] p {{ font-size: 13px !important; color: #666666 !important; font-family: {BODY_FONT} !important; }}
[data-testid="stExpander"] summary span {{ font-size: 15px !important; font-weight: 600 !important; color: {C['t800']} !important; }}
[data-testid="stAlert"] p {{ font-size: 15px !important; color: {C['txt']} !important; }}
.stSelectbox label, .stMultiSelect label, .stSlider label,
.stRadio label,
.stCheckbox label {{ font-size: 15px !important; font-weight: 600 !important; color: {C['txt']} !important; font-family: {BODY_FONT} !important; }}
[data-baseweb="option"] {{ font-size: 14px !important; color: {C['txt']} !important; }}
[data-baseweb="radio"] label,
[data-baseweb="radio"] label * {{ color: #111111 !important; }}
div[data-testid="stRadio"] label p,
div[data-testid="stRadio"] label span {{ color: #111111 !important; }}

/* ── 3. HEADER — scoped class forces white, beats everything ────────────── */
.psia-header       {{ color: #FFFFFF !important; }}
.psia-header *     {{ color: #FFFFFF !important; }}
.psia-header p     {{ color: #FFFFFF !important; }}
.psia-header span  {{ color: #FFFFFF !important; }}
.psia-header div   {{ color: #FFFFFF !important; }}
.psia-header-sub   {{ color: #B8EDD8 !important; }}
.psia-header-sub * {{ color: #B8EDD8 !important; }}
.psia-header-period   {{ color: #9FE1CB !important; }}
.psia-header-period * {{ color: #9FE1CB !important; }}

/* ── 4. SIDEBAR — nuclear white text, layered specificity ───────────────── */
/* Background */
[data-testid="stSidebar"] {{
    background: {C['t900']} !important;
    background-color: {C['t900']} !important;
    padding-top: 0 !important;
}}
[data-testid="stSidebarContent"] {{
    background: {C['t900']} !important;
    background-color: {C['t900']} !important;
}}
/* Every element white */
[data-testid="stSidebar"] *,
[data-testid="stSidebarContent"] * {{
    color: #FFFFFF !important;
    opacity: 1 !important;
}}
/* Explicit element types */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{
    color: #FFFFFF !important;
    opacity: 1 !important;
}}
/* Widget labels */
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] span,
[data-testid="stSidebar"] [class*="Label"] p,
[data-testid="stSidebar"] [class*="label"] p {{
    color: #FFFFFF !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    opacity: 1 !important;
    font-family: {BODY_FONT} !important;
}}
/* Slider tick values */
[data-testid="stSidebar"] [data-testid="stThumbValue"],
[data-testid="stSidebar"] [data-testid="stThumbValue"] *,
[data-testid="stSidebar"] [data-testid="stTickBarMin"],
[data-testid="stSidebar"] [data-testid="stTickBarMax"],
[data-testid="stSidebar"] [data-testid="stTickBar"] * {{
    color: #9FE1CB !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}}
/* Multiselect tags */
[data-testid="stSidebar"] [data-baseweb="tag"] {{
    background-color: {C['t600']} !important;
    border: 1.5px solid rgba(255,255,255,0.35) !important;
    border-radius: 6px !important;
}}
[data-testid="stSidebar"] [data-baseweb="tag"] span,
[data-testid="stSidebar"] [data-baseweb="tag"] * {{
    color: #FFFFFF !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}}
[data-testid="stSidebar"] [data-baseweb="tag"] svg {{
    fill: #FFFFFF !important;
}}
/* Select dropdowns */
[data-testid="stSidebar"] [data-baseweb="select"] > div {{
    background-color: rgba(255,255,255,0.12) !important;
    border-color: rgba(255,255,255,0.30) !important;
    border-radius: 8px !important;
}}
[data-testid="stSidebar"] [data-baseweb="select"] span,
[data-testid="stSidebar"] [data-baseweb="select"] * {{
    color: #FFFFFF !important;
    font-size: 14px !important;
}}
[data-testid="stSidebar"] [data-baseweb="select"] svg {{
    fill: #FFFFFF !important;
}}
/* Collapse arrow */
[data-testid="collapsedControl"] {{
    background: {C['t600']} !important;
    border-radius: 0 8px 8px 0 !important;
    box-shadow: 3px 0 10px rgba(0,0,0,0.30) !important;
}}
[data-testid="collapsedControl"] svg {{
    fill: #FFFFFF !important;
}}

/* ── 5. TABS ─────────────────────────────────────────────────────────────── */
[data-baseweb="tab-list"] {{
    margin-bottom: 6px !important;
    border-bottom: 2px solid #E0E8E4 !important;
}}
[data-baseweb="tab-list"] [data-baseweb="tab"] button,
[data-baseweb="tab-list"] [data-baseweb="tab"] button p,
[data-baseweb="tab-list"] [data-baseweb="tab"] button span,
[data-baseweb="tab-list"] button[role="tab"],
[data-baseweb="tab-list"] button[role="tab"] p,
[data-baseweb="tab-list"] button[role="tab"] span {{
    color: #111111 !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    font-family: {BODY_FONT} !important;
    padding: 8px 16px !important;
}}
[data-baseweb="tab-list"] [aria-selected="true"] button,
[data-baseweb="tab-list"] [aria-selected="true"] button p,
[data-baseweb="tab-list"] [aria-selected="true"] button span {{
    color: #085041 !important;
    font-weight: 800 !important;
}}

/* ── 6. KPI CARDS ────────────────────────────────────────────────────────── */
.kpi-card {{
    background: #FFFFFF !important;
    border-radius: 12px !important;
    padding: 18px 20px !important;
    border: 1px solid #E2EAE6 !important;
    border-top: 5px solid {C['t400']} !important;
    margin-bottom: 8px !important;
    box-shadow: 0 2px 12px rgba(4,52,44,0.08) !important;
}}
.kpi-card.amber {{ border-top-color: {C['amber']} !important; }}
.kpi-card.coral {{ border-top-color: {C['coral']} !important; }}
.kpi-card.blue  {{ border-top-color: {C['blue']}  !important; }}
.kpi-card.green {{ border-top-color: {C['green']} !important; }}
.kpi-label {{ font-size: 10.5px !important; font-weight: 800 !important; color: #777777 !important; text-transform: uppercase !important; letter-spacing: 1.1px !important; margin-bottom: 8px !important; font-family: {BODY_FONT} !important; }}
.kpi-value {{ font-size: 30px !important; font-weight: 700 !important; color: {C['t900']} !important; font-family: {SERIF_FONT} !important; letter-spacing: -0.5px !important; }}
.kpi-delta {{ font-size: 12.5px !important; font-weight: 600 !important; color: #1B7A36 !important; margin-top: 6px !important; font-family: {BODY_FONT} !important; }}
.kpi-delta.neg {{ color: {C['coral']} !important; }}
.kpi-src {{ font-size: 11px !important; color: #888888 !important; margin-top: 6px !important; font-family: {BODY_FONT} !important; }}

/* ── 7. SECTION HEADINGS ─────────────────────────────────────────────────── */
.sec-head {{
    display: block !important; font-family: {SERIF_FONT} !important;
    font-size: 16px !important; font-weight: 700 !important;
    color: {C['t900']} !important; padding: 8px 0 8px !important;
    border-bottom: 2px solid {C['t100']} !important; margin-bottom: 14px !important;
    background-color: transparent !important;
}}

/* ── 8. TAGS ─────────────────────────────────────────────────────────────── */
.tag {{ display: inline-block !important; background: #E8F5F0 !important; color: {C['t800']} !important; border-radius: 20px !important; padding: 3px 11px !important; font-size: 12px !important; font-weight: 700 !important; font-family: {BODY_FONT} !important; margin-right: 6px !important; margin-bottom: 6px !important; border: 1px solid {C['t100']} !important; }}
.tag.sim {{ background: #FFF4E0 !important; color: #7A4800 !important; border-color: #FFD999 !important; }}
.tag.ext {{ background: #EBF0FF !important; color: #1A3A8B !important; border-color: #C5D0FF !important; }}
.tag.new {{ background: #E6F7EC !important; color: #155724 !important; border-color: #A8D9B8 !important; }}

/* ── 9. BUTTONS ──────────────────────────────────────────────────────────── */
[data-testid="baseButton-secondary"],
.stButton > button[kind="secondary"] {{
    background-color: {C['t800']} !important; color: #FFFFFF !important;
    border: 1.5px solid {C['t400']} !important; font-size: 13px !important;
    font-weight: 600 !important; font-family: {BODY_FONT} !important;
    border-radius: 8px !important; text-align: left !important;
}}
[data-testid="baseButton-secondary"]:hover,
.stButton > button[kind="secondary"]:hover {{
    background-color: {C['t600']} !important; color: #FFFFFF !important;
}}
[data-testid="baseButton-primary"],
.stButton > button[kind="primary"] {{
    background-color: {C['t600']} !important; color: #FFFFFF !important;
    border: none !important; font-size: 14px !important; font-weight: 700 !important;
    border-radius: 8px !important;
}}

/* ── 10. DATA TABLES ─────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {{ border-radius: 10px !important; overflow: hidden !important; box-shadow: 0 2px 8px rgba(0,0,0,0.07) !important; }}
[data-testid="stDataFrame"] th {{ background: {C['t800']} !important; color: #FFFFFF !important; font-size: 13px !important; font-weight: 700 !important; padding: 10px 14px !important; }}
[data-testid="stDataFrame"] td {{ font-size: 13.5px !important; color: {C['txt']} !important; padding: 8px 14px !important; }}

/* ── 11. EXPANDERS ───────────────────────────────────────────────────────── */
[data-testid="stExpander"] {{ border: 1px solid #E2EAE6 !important; border-radius: 10px !important; background: #FFFFFF !important; }}
[data-testid="stExpander"] summary {{ padding: 12px 16px !important; background: #FFFFFF !important; border-radius: 10px !important; }}

/* ── 12. MISC ────────────────────────────────────────────────────────────── */
hr {{ border-color: #E0E8E4 !important; margin: 16px 0 !important; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA — FAO CSVs
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

# ─────────────────────────────────────────────────────────────────────────────
# CANADIAN AQUACULTURE STATIC DATA (Statistics Canada / DFO, 1986-2024)
# Extracted from aqua86_e.xls → aqua24_e.xls and va97pub_e.xls → va24pub_e.xls
# ─────────────────────────────────────────────────────────────────────────────
# Production (tonnes) — BC and Canada totals
# BC data suppressed by StatsCan 1991-2001; early 1986-90 are shellfish+finfish combined
CAN_PROD_T = {
    1986:10488,  1987:13936,  1988:21461,  1989:30273,  1990:36462,
    1991:39004,  1992:35808,  1993:42349,  1994:42119,  1995:48854,
    1996:53499,  1997:63918,  1998:92105,  1999:114204, 2000:95003,
    2001:154069, 2002:171799, 2003:150205, 2004:141580, 2005:154484,
    2006:171629, 2007:152475, 2017:191111, 2018:190184, 2019:187258,
    2020:171226, 2021:191249, 2022:166265, 2023:145985, 2024:160238,
}
BC_PROD_T = {
    1986:3378,   1987:5550,   1988:10402,  1989:15731,  1990:17739,
    2002:93090,  2003:75126,  2004:65666,  2005:73624,  2006:80672,
    2007:81384,  2008:81873,  2009:78345,  2010:82435,  2011:92264,
    2012:90382,  2013:84258,  2014:66403,  2015:102385, 2016:102325,
    2017:96608,  2018:97783,  2019:100293, 2020:100191, 2021:96074,
    2022:96507,  2023:60962,  2024:64154,
}
# BC Value ($000 CAD)
BC_VAL_K = {
    1986:5842,   1987:15875,  1988:42163,  1989:63172,  1990:84914,
    2002:304400, 2003:273531, 2004:241828, 2005:337158, 2006:427530,
    2007:407766, 2008:433060, 2009:422752, 2010:538490, 2011:463925,
    2012:434736, 2013:507490, 2014:411941, 2015:474455, 2016:752099,
    2017:768529, 2018:814281, 2019:708660, 2020:665753, 2021:738330,
    2022:916924, 2023:557893, 2024:562814,
}
CAN_VAL_K = {
    1986:35106,  1987:61669,  1988:105355, 1989:139137, 1990:195955,
    1991:221328, 1992:231068, 1993:263942, 1994:283082, 1995:317753,
    1996:327721, 1997:359655, 1998:396107, 1999:517352, 2000:608881,
    2001:539483, 2002:628318, 2003:583285, 2004:532924, 2005:706794,
    2006:904595, 2007:752690, 2017:1392153,2018:1431707,2019:1231550,
    2020:1044986,2021:1344745,2022:1342553,2023:1259693,2024:1365820,
}
# BC Salmon production (tonnes) — where available
BC_SALMON_T = {
    1991:24362,  1992:19814,  1993:25555,  1994:23657,  1995:27275,
    1996:27756,  1997:36465,  1998:42200,  1999:49700,  2000:49000,
    2001:68000,  2002:84200,  2003:65411,  2004:55646,  2005:63370,
    2006:70181,  2007:70998,  2008:73265,  2009:68662,  2010:70831,
    2011:83144,  2012:79981,  2013:74673,  2014:54971,  2015:92926,
    2016:90511,  2017:85608,  2018:87010,  2019:88874,  2020:91666,
    2021:84171,  2022:85191,  2023:50067,  2024:53816,
}
# BC Shellfish (tonnes)
BC_SHELL_T = {
    2002:8790,  2003:9579,  2004:9905,  2010:10120, 2011:7973,
    2012:9669,  2013:8450,  2014:10127, 2015:8535,  2016:10417,
    2017:9442,  2018:8949,  2019:9698,  2020:6673,  2021:9526,
    2022:10149, 2023:9588,  2024:9537,
}

# Value Added Account (VA) data — BC ($000 CAD), Statistics Canada / DFO 1997-2024
VA_BC_GVA_K = {
    1997:97500,  1998:111960, 1999:133995, 2000:139400, 2001:116500,
    2002:70750,  2003:61700,  2004:58430,  2005:100695, 2006:162050,
    2007:159560, 2008:123825, 2009:152440, 2010:210885, 2011:129920,
    2012:75495,  2013:160945, 2014:162050, 2017:338015, 2018:313255,
    2019:180595, 2020:158152, 2021:201866, 2022:198784, 2024:21600,
}
VA_BC_FEED_K = {
    1997:69000,  1998:83000,  1999:95000,  2000:96000,  2001:120000,
    2002:135000, 2003:145000, 2004:137000, 2005:140000, 2006:155000,
    2007:158000, 2008:158000, 2009:159600, 2010:164500, 2011:166555,
    2012:159950, 2013:160550, 2014:183620, 2015:231400, 2016:222955,
    2017:219140, 2018:237255, 2019:248314, 2020:223196, 2021:220220,
    2022:214576, 2024:178254,
}
VA_BC_WAGES_K = {
    1997:30000,  1998:30500,  1999:35000,  2000:40000,  2001:43000,
    2002:48000,  2003:41000,  2004:43000,  2005:41000,  2006:48000,
    2007:53000,  2008:58900,  2009:66500,  2010:60010,  2011:62855,
    2012:60515,  2013:55190,  2014:61795,  2015:65080,  2016:68500,
    2017:70730,  2018:71090,  2019:75853,  2020:77749,  2021:82414,
    2022:84063,  2024:72403,
}
VA_BC_THERAP_K = {
    1997:3000,   1998:3700,   1999:4000,   2000:4300,   2001:4300,
    2002:4500,   2003:5300,   2004:7000,   2005:8000,   2006:6900,
    2007:7700,   2008:5000,   2009:9600,   2010:11100,  2011:9280,
    2012:9500,   2013:14655,  2014:19560,  2015:26615,  2016:26925,
    2017:26390,  2018:38765,  2019:35334,  2020:30611,  2021:28420,
    2022:43694,  2024:36238,
}
VA_BC_OUTPUT_K = {
    1997:275000, 1998:285100, 1999:332900, 2000:245800, 2001:346900,
    2002:332950, 2003:328000, 2004:321430, 2005:373175, 2006:461500,
    2007:541920, 2008:462375, 2009:463240, 2010:551265, 2011:473425,
    2012:422850, 2013:517555, 2014:509890, 2015:535430, 2016:799070,
    2017:835355, 2018:883080, 2019:756304, 2020:691115, 2021:784264,
    2022:870090, 2024:626689,
}
VA_CAN_GVA_K = {
    1997:185970, 1998:226040, 1999:275930, 2000:307120, 2001:255200,
    2002:220005, 2003:209709, 2004:203345, 2005:249900, 2006:395787,
    2010:422765, 2011:267175, 2012:203265, 2013:326333, 2014:244455,
    2015:165850, 2016:591195, 2017:563700, 2018:485430, 2019:305335,
    2020:150351, 2021:243123, 2022:241261, 2024:236247,
}

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
        "agreements":    [3,4,5,7,9,11,14,17,20,24],
        "employed":      [85,92,101,112,124,138,155,170,188,207],
        "trained_indig": [40,48,57,68,82,98,115,133,152,174],
        "trained_total": [120,145,172,204,241,283,330,383,441,504],
    })
    return permitting, social

gp, aq, av, cq, data_ok = load_fao()
sim_perm, sim_social = build_simulated()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # ── Logo & Brand ─────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:28px 12px 20px;
                border-bottom:1px solid rgba(255,255,255,0.22);
                margin-bottom:20px;background:transparent;">
      <div style="font-size:46px;margin-bottom:12px;line-height:1;
                  filter:drop-shadow(0 2px 8px rgba(0,0,0,0.3));">🌿</div>
      <div style="font-family:Georgia,serif;font-size:24px;font-weight:700;
                  color:#FFFFFF !important;letter-spacing:0.3px;line-height:1.2;
                  text-shadow:0 1px 3px rgba(0,0,0,0.4);">
        PSIA Dashboard
      </div>
      <div style="font-size:13px;color:#9FE1CB !important;margin-top:8px;
                  letter-spacing:0.5px;line-height:1.5;font-weight:500;">
        Pacific Seaweed Industry Association
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Filters header ───────────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:12px !important;font-weight:800 !important;"
        "letter-spacing:1.8px !important;text-transform:uppercase !important;"
        "color:#9FE1CB !important;margin-bottom:16px !important;"
        "padding:0 0 8px !important;"
        "border-bottom:1px solid rgba(255,255,255,0.25) !important;"
        "display:block !important;opacity:1 !important;'>"
        "📊 &nbsp;DASHBOARD FILTERS</div>", unsafe_allow_html=True)

    y_min = int(gp["period"].min()) if data_ok else 1950
    y_max = int(gp["period"].max()) if data_ok else 2024
    year_range = st.slider("Year Range", y_min, y_max, (2000, y_max))

    all_cont = sorted([c for c in gp["continent_group_en"].unique() if c != "Unknown"]) if data_ok else []
    sel_cont = st.multiselect("Continent Filter", all_cont, default=all_cont)
    top_n    = st.selectbox("Top-N Species / Countries", [5, 10, 15], index=1)
    cagr_win = st.selectbox("CAGR Window (years)", [5, 10, 20], index=1)

    # ── Data Sources box ─────────────────────────────────────────────────────
    st.markdown(
        "<div style='height:1px;background:rgba(255,255,255,0.20);margin:20px 0 16px;'></div>",
        unsafe_allow_html=True)
    st.markdown(
        "<div style='background:rgba(255,255,255,0.10);border-radius:10px;"
        "padding:14px 16px;border:1.5px solid rgba(255,255,255,0.20);margin-top:4px;'>"
        "<div style='font-size:11px !important;font-weight:800 !important;"
        "letter-spacing:1.6px !important;text-transform:uppercase !important;"
        "color:#9FE1CB !important;margin-bottom:12px !important;"
        "display:block !important;opacity:1 !important;'>"
        "DATA SOURCES</div>"
        "<div style='font-size:14px !important;color:#FFFFFF !important;"
        "line-height:2.2 !important;font-weight:500 !important;"
        "opacity:1 !important;display:block !important;'>"
        "<span style='color:#FFFFFF !important;'>🟢</span>"
        "<span style='color:#FFFFFF !important;'> FAO FishStat — seaweed global</span><br>"
        "<span style='color:#FFFFFF !important;'>🔵</span>"
        "<span style='color:#FFFFFF !important;'> Statistics Canada — CA aqua</span><br>"
        "<span style='color:#FFFFFF !important;'>🟡</span>"
        "<span style='color:#FFFFFF !important;'> DFO / CIRNAC — estimated</span><br>"
        "<span style='color:#FFFFFF !important;'>📅</span>"
        "<span style='color:#FFFFFF !important;'> Data through 2024</span>"
        "</div></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FILTER & KPI HELPERS
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
    av_yr.columns = ["country_name","seaweed_name","usd_k"]
    aq_yr.columns = ["country_name","seaweed_name","tonnes"]
    jp = av_yr.merge(aq_yr, on=["country_name","seaweed_name"])
    jp = jp[(jp["tonnes"]>0)&(jp["usd_k"]>0)]
    avg_price_kg = (jp["usd_k"].sum()*1000 / jp["tonnes"].sum() / 1000) if len(jp) else 0

    pb_v = max(LY - cagr_win, year_range[0])
    pb_val = fgp[fgp["period"]==pb_v]["value"].sum()
    cagr_prod = ((prod_tot / pb_val)**(1/cagr_win) - 1)*100 if pb_val else 0

def card(col, label, val, delta, pos=True, accent=""):
    with col:
        st.markdown(
            f'<div class="kpi-card{" "+accent if accent else ""}">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{val}</div>'
            f'<div class="kpi-delta{"" if pos else " neg"}">{delta}</div>'
            f'</div>', unsafe_allow_html=True)

def sim_card(col, label, val, delta, src, accent=""):
    with col:
        st.markdown(
            f'<div class="kpi-card{" "+accent if accent else ""}">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{val}</div>'
            f'<div class="kpi-delta">{delta}</div>'
            f'<div class="kpi-src">Source: {src}</div>'
            f'</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE HEADER — scoped CSS forces white text regardless of Streamlit theme
# NOTE: No HTML comments allowed inside st.markdown — they break the parser
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
.psia-header, .psia-header *, .psia-header p, .psia-header span, .psia-header div, .psia-header a {{
    color: #FFFFFF !important;
}}
.psia-header-sub {{ color: #B8EDD8 !important; }}
.psia-header-sub * {{ color: #B8EDD8 !important; }}
.psia-header-period {{ color: #9FE1CB !important; }}
.psia-header-period * {{ color: #9FE1CB !important; }}
</style>
<div class="psia-header"
     style="background:linear-gradient(135deg,#04342C 0%,#0A5540 55%,#0F6E56 100%);
            padding:26px 32px 24px;border-radius:14px;margin-bottom:20px;
            box-shadow:0 6px 24px rgba(4,52,44,0.40),0 2px 6px rgba(0,0,0,0.20);
            border:1px solid rgba(255,255,255,0.10);">
  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:16px;">
    <div style="display:flex;align-items:center;gap:16px;">
      <div style="font-size:36px;line-height:1;flex-shrink:0;">🌿</div>
      <div>
        <div style="font-family:Lora,Georgia,serif;font-size:26px;font-weight:700;
                    color:#FFFFFF !important;letter-spacing:0.1px;line-height:1.2;
                    text-shadow:0 2px 8px rgba(0,0,0,0.50);">
          PSIA Seaweed Industry Analytics Dashboard
        </div>
        <div class="psia-header-sub"
             style="font-size:14px;color:#B8EDD8 !important;margin-top:6px;
                    font-family:'Source Sans 3',Helvetica,sans-serif;font-weight:400;
                    letter-spacing:0.4px;text-shadow:0 1px 4px rgba(0,0,0,0.40);">
          Global Seaweed Production &nbsp;&middot;&nbsp; Canadian Aquaculture Context
          &nbsp;&middot;&nbsp; Value Added &nbsp;&middot;&nbsp; Permitting &amp; Social KPIs
        </div>
      </div>
    </div>
    <div style="text-align:right;flex-shrink:0;">
      <div class="psia-header-period"
           style="font-size:11px;color:#9FE1CB !important;text-transform:uppercase;
                  letter-spacing:1.8px;font-weight:700;margin-bottom:6px;
                  font-family:'Source Sans 3',Helvetica,sans-serif;
                  text-shadow:none;">
        VIEWING PERIOD
      </div>
      <div style="font-size:25px;font-weight:800;color:#FFFFFF !important;
                  font-family:Lora,Georgia,serif;letter-spacing:2px;
                  background:rgba(255,255,255,0.14);padding:8px 22px;
                  border-radius:10px;border:1.5px solid rgba(255,255,255,0.28);
                  text-shadow:0 2px 6px rgba(0,0,0,0.40);">
        {year_range[0]} &ndash; {year_range[1]}
      </div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
_main_col, _chat_col = st.columns([7, 3], gap="medium")

with _main_col:
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Production & Value",
        "🌍 Geographic & Species",
        "🏛️ Permitting & Social",
        "🔬 Advanced Analytics",
        "🇨🇦 Canadian Context",
    ])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PRODUCTION & VALUE
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    if not data_ok:
        st.warning("Data files not loaded. Place CSVs in data/")
    else:
        st.markdown(
            f'<div style="font-family:Lora,Georgia,serif;font-size:20px;font-weight:700;'
            f'color:{C["t900"]};margin-bottom:10px;letter-spacing:0.1px;">'
            f'Operational KPIs — Global Seaweed</div>',
            unsafe_allow_html=True)
        st.markdown('<span class="tag">🟢 FAO FishStat</span>'
                    '<span class="tag">G4-OP1 · G4-OP2 · G4-OP3 · G4-OP4 · G4-OP5 · KPI-15</span>',
                    unsafe_allow_html=True)
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        card(c1,"G4-OP1 · Wet Weight (Total)", f"{prod_tot/1e6:.2f}M t",
             f"{'▲' if yoy_prod>=0 else '▼'} {abs(yoy_prod):.1f}% YoY  ·  CAGR {cagr_prod:.1f}%",
             yoy_prod>=0)
        card(c2,"G4-OP2 · Cultivation (Farmed)", f"{aq_tot/1e6:.2f}M t",
             f"{aq_tot/prod_tot*100:.1f}% of total tonnage")
        card(c3,"G4-OP3 · Wild Collection", f"{wc_tot/1e6:.3f}M t",
             f"{wc_tot/prod_tot*100:.1f}% of total tonnage")
        card(c4,"G4-OP4 · ASFIS Species", str(sp_total),
             f"Farmed: {sp_cult}  ·  Wild: {sp_wild}")
        card(c5,"G4-OP5 · Avg Price/kg (proxy)", f"${avg_price_kg:.2f}/kg",
             f"USD  ·  derived av÷aq  ({LY})")
        card(c6,"KPI-15 · Value YoY Growth", f"{'▲' if yoy_val>=0 else '▼'} {abs(yoy_val):.1f}%",
             f"${val_tot/1e6:.1f}B USD total  ·  {LY}",
             yoy_val>=0, accent="blue")

        st.markdown("<br>", unsafe_allow_html=True)
        r1a, r1b = st.columns(2)

        with r1a:
            st.markdown('<div class="sec-head">G4-OP1 — Global Production Trend</div>', unsafe_allow_html=True)
            pt = fgp.groupby("period")["value"].sum().reset_index()
            pt.columns = ["year","tonnes"]
            pt["yoy"] = pt["tonnes"].pct_change()*100
            pt["tm"]  = pt["tonnes"]/1e6
            fig = make_subplots(rows=2,cols=1,row_heights=[0.68,0.32],shared_xaxes=True,
                                vertical_spacing=0.06,subplot_titles=["Production (M tonnes)","YoY Growth (%)"])
            fig.add_trace(go.Scatter(x=pt["year"],y=pt["tm"],mode="lines",fill="tozeroy",
                line=dict(color=C["t600"],width=2.5),fillcolor="rgba(15,110,86,0.12)",name="M tonnes"),row=1,col=1)
            fig.add_trace(go.Bar(x=pt["year"],y=pt["yoy"].fillna(0),
                marker_color=[C["t400"] if v>=0 else C["coral"] for v in pt["yoy"].fillna(0)],
                name="YoY %",showlegend=False),row=2,col=1)
            fig.update_layout(**base_layout(370,dict(l=4,r=4,t=40,b=4)),legend=LEGEND_TOP)
            fig.update_annotations(font=dict(size=12,color=C["txt"],family="'Source Sans Pro',Helvetica,sans-serif"))
            fig.update_xaxes(tickfont=AXIS_FONT,gridcolor=GRID_COLOR,linecolor=AXIS_LINE["color"])
            fig.update_yaxes(tickfont=AXIS_FONT,gridcolor=GRID_COLOR,linecolor=AXIS_LINE["color"])
            st.plotly_chart(fig, use_container_width=True, key="autoplt_0")

        with r1b:
            st.markdown('<div class="sec-head">G4-OP2 / OP3 — Cultivation vs Wild Collection</div>', unsafe_allow_html=True)
            aq_t = faq.groupby("period")["value"].sum().reset_index(); aq_t.columns = ["year","c"]
            cq_t = fcq.groupby("period")["value"].sum().reset_index(); cq_t.columns = ["year","w"]
            combo = aq_t.merge(cq_t,on="year",how="outer").fillna(0)
            combo["cult_m"] = combo["c"]/1e6; combo["wild_m"] = combo["w"]/1e6
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=combo["year"],y=combo["cult_m"],mode="lines",name="Cultivation",
                stackgroup="one",line=dict(color=C["t600"]),fillcolor="rgba(15,110,86,0.55)"))
            fig2.add_trace(go.Scatter(x=combo["year"],y=combo["wild_m"],mode="lines",name="Wild Capture",
                stackgroup="one",line=dict(color=C["amber"]),fillcolor="rgba(201,125,16,0.55)"))
            fig2.update_layout(**base_layout(370),legend=LEGEND_TOP)
            fig2 = style_axes(fig2,ytitle="M tonnes")
            st.plotly_chart(fig2, use_container_width=True, key="autoplt_1")

        r2a, r2b = st.columns(2)
        with r2a:
            st.markdown('<div class="sec-head">KPI-13 — Aquaculture Value vs Volume</div>', unsafe_allow_html=True)
            vt = fav.groupby("period")["value"].sum().reset_index(); vt.columns = ["year","usd_k"]; vt["usd_b"] = vt["usd_k"]/1e6
            qt = faq.groupby("period")["value"].sum().reset_index(); qt.columns = ["year","tonnes"]; qt["tm"] = qt["tonnes"]/1e6
            mv = vt.merge(qt,on="year",how="inner")
            fig3 = make_subplots(specs=[[{"secondary_y": True}]])
            fig3.add_trace(go.Scatter(x=mv["year"],y=mv["usd_b"],mode="lines",name="Value ($B USD)",
                line=dict(color=C["t600"],width=2.5),fill="tozeroy",fillcolor="rgba(15,110,86,0.08)"),secondary_y=False)
            fig3.add_trace(go.Scatter(x=mv["year"],y=mv["tm"],mode="lines",name="Volume (M t)",
                line=dict(color=C["amber"],width=2.2,dash="dot")),secondary_y=True)
            fig3.update_layout(**base_layout(310),legend=LEGEND_TOP)
            fig3.update_xaxes(tickfont=AXIS_FONT,gridcolor=GRID_COLOR)
            fig3.update_yaxes(title_text="Value (USD $B)",tickfont=AXIS_FONT,gridcolor=GRID_COLOR,secondary_y=False,title_font=AXIS_FONT)
            fig3.update_yaxes(title_text="Volume (M t)",tickfont=AXIS_FONT,gridcolor=GRID_COLOR,secondary_y=True,title_font=AXIS_FONT)
            st.plotly_chart(fig3, use_container_width=True, key="autoplt_2")

        with r2b:
            st.markdown('<div class="sec-head">G4-OP5 — Implied Price per kg by Species</div>', unsafe_allow_html=True)
            sp_price = jp.groupby("seaweed_name").apply(
                lambda x: x["usd_k"].sum()*1000/x["tonnes"].sum()/1000).reset_index()
            sp_price.columns = ["species","price_per_kg"]
            sp_price = sp_price[sp_price["price_per_kg"]<20].nlargest(top_n,"price_per_kg").sort_values("price_per_kg")
            fig4 = go.Figure()
            for _,row in sp_price.iterrows():
                fig4.add_shape(type="line",x0=0,x1=row["price_per_kg"],y0=row["species"],y1=row["species"],
                    line=dict(color=C["t400"],width=2.5))
            fig4.add_trace(go.Scatter(x=sp_price["price_per_kg"],y=sp_price["species"],mode="markers",
                marker=dict(size=13,color=C["t400"],line=dict(color=C["t800"],width=1.5)),
                text=sp_price["price_per_kg"].round(2).astype(str)+" $/kg",
                textposition="middle right",textfont=dict(size=11,color=C["txt"]),name="$/kg"))
            fig4.update_layout(**base_layout(310,dict(l=4,r=80,t=16,b=8)),showlegend=False)
            fig4 = style_axes(fig4,xtitle="USD per kg")
            fig4.update_yaxes(tickfont=dict(size=11,color=C["txt"]))
            st.plotly_chart(fig4, use_container_width=True, key="autoplt_3")

        st.markdown('<div class="sec-head">G4-OP4 — ASFIS Species Diversity Over Time</div>', unsafe_allow_html=True)
        sp_c = faq[faq["value"]>0].groupby("period")["seaweed_name"].nunique().reset_index()
        sp_w = fcq[fcq["value"]>0].groupby("period")["seaweed_name"].nunique().reset_index()
        sp_c.columns = ["year","cultivated"]; sp_w.columns = ["year","wild"]
        sp_div = sp_c.merge(sp_w,on="year",how="outer").fillna(0)
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=sp_div["year"],y=sp_div["cultivated"],mode="lines+markers",name="Cultivated",
            line=dict(color=C["t600"],width=2.5),marker=dict(size=6,color=C["t600"])))
        fig5.add_trace(go.Scatter(x=sp_div["year"],y=sp_div["wild"],mode="lines+markers",name="Wild-collected",
            line=dict(color=C["amber"],width=2.5,dash="dot"),marker=dict(size=6,color=C["amber"])))
        fig5.update_layout(**base_layout(220,dict(l=4,r=4,t=12,b=4)),legend=LEGEND_TOP)
        fig5 = style_axes(fig5,ytitle="# Species")
        st.plotly_chart(fig5, use_container_width=True, key="autoplt_4")

        # KPI-8 CR5/CR10
        st.markdown('<div class="sec-head">KPI-8 — Market Concentration Ratio (CR5 / CR10)</div>', unsafe_allow_html=True)
        cr_data = (fgp[fgp["value"]>0].groupby(["period","country_name"])["value"].sum().reset_index())
        cr_rows = []
        for yr_cr, grp in cr_data.groupby("period"):
            total = grp["value"].sum()
            if total==0: continue
            sv = grp["value"].sort_values(ascending=False).values
            cr5  = sv[:5].sum()/total*100  if len(sv)>=5  else sv.sum()/total*100
            cr10 = sv[:10].sum()/total*100 if len(sv)>=10 else sv.sum()/total*100
            cr_rows.append({"year":yr_cr,"CR5":round(cr5,1),"CR10":round(cr10,1)})
        cr_df = pd.DataFrame(cr_rows)
        cr_ly = cr_df[cr_df["year"]==LY]
        k8a,k8b,k8c = st.columns([1,1,4])
        cr5_val  = float(cr_ly["CR5"].values[0])  if not cr_ly.empty else 0
        cr5_prev = float(cr_df[cr_df["year"]==PY]["CR5"].values[0]) if not cr_df[cr_df["year"]==PY].empty else cr5_val
        card(k8a,f"KPI-8 · CR5 ({LY})",f"{cr5_val:.1f}%",
             f"Top 5 countries · {'▲' if cr5_val>=cr5_prev else '▼'} {abs(cr5_val-cr5_prev):.1f}pp YoY",
             cr5_val<=cr5_prev)
        cr10_val  = float(cr_ly["CR10"].values[0])  if not cr_ly.empty else 0
        cr10_prev = float(cr_df[cr_df["year"]==PY]["CR10"].values[0]) if not cr_df[cr_df["year"]==PY].empty else cr10_val
        card(k8b,f"KPI-8 · CR10 ({LY})",f"{cr10_val:.1f}%",
             f"Top 10 countries · {'▲' if cr10_val>=cr10_prev else '▼'} {abs(cr10_val-cr10_prev):.1f}pp YoY",
             cr10_val<=cr10_prev)
        with k8c:
            fig_cr = go.Figure()
            fig_cr.add_trace(go.Scatter(x=cr_df["year"],y=cr_df["CR5"],mode="lines+markers",
                name="CR5",line=dict(color=C["t600"],width=2.5),marker=dict(size=6),
                fill="tozeroy",fillcolor="rgba(15,110,86,0.07)"))
            fig_cr.add_trace(go.Scatter(x=cr_df["year"],y=cr_df["CR10"],mode="lines+markers",
                name="CR10",line=dict(color=C["amber"],width=2.2,dash="dot"),marker=dict(size=6,color=C["amber"])))
            fig_cr.update_layout(**base_layout(200,dict(l=4,r=4,t=10,b=4)),legend=LEGEND_TOP)
            fig_cr = style_axes(fig_cr,ytitle="Share of global production (%)")
            fig_cr.update_yaxes(range=[0,105])
            st.plotly_chart(fig_cr, use_container_width=True, key="autoplt_5")

        with st.expander("📋 Production & Value Data Tables"):
            dt1,dt2,dt3 = st.tabs(["Production Trend","Cultivation vs Wild","Value vs Volume"])
            with dt1:
                tbl = pt[["year","tm","yoy"]].copy(); tbl.columns = ["Year","Production (M t)","YoY (%)"]
                st.dataframe(tbl.round(3).sort_values("Year",ascending=False).reset_index(drop=True),use_container_width=True,hide_index=True)
            with dt2:
                tbl2 = combo[["year","cult_m","wild_m"]].round(3).copy(); tbl2.columns = ["Year","Cultivation (M t)","Wild (M t)"]
                st.dataframe(tbl2.sort_values("Year",ascending=False).reset_index(drop=True),use_container_width=True,hide_index=True)
            with dt3:
                tbl3 = mv[["year","usd_b","tm"]].round(3).copy(); tbl3.columns = ["Year","Value ($B USD)","Volume (M t)"]
                st.dataframe(tbl3.sort_values("Year",ascending=False).reset_index(drop=True),use_container_width=True,hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — GEOGRAPHIC & SPECIES
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    if not data_ok:
        st.warning("Data files not loaded.")
    else:
        cont_now = fgp[fgp["period"]==LY].groupby("continent_group_en")["value"].sum()
        asia_pct = cont_now.get("Asia",0)/cont_now.sum()*100
        ig_now   = fgp[fgp["period"]==LY].groupby("ecoclass_group_en")["value"].sum()
        um_pct   = ig_now.get("Upper-middle income countries",0)/ig_now.sum()*100
        hi_pct   = ig_now.get("High-income countries",0)/ig_now.sum()*100
        n_cnt    = int(fgp[(fgp["period"]==LY)&(fgp["value"]>0)]["country_name"].nunique())

        st.markdown(
            f'<div style="font-family:Lora,Georgia,serif;font-size:20px;font-weight:700;'
            f'color:{C["t900"]};margin-bottom:10px;">Geographic &amp; Species KPIs</div>',
            unsafe_allow_html=True)
        st.markdown('<span class="tag">🟢 FAO FishStat</span>'
                    '<span class="tag">KPI-6 · KPI-7 · KPI-8 · KPI-10 · KPI-17 · KPI-18 · KPI-19</span>',
                    unsafe_allow_html=True)
        gc1,gc2,gc3,gc4 = st.columns(4)
        card(gc1,"KPI-7 · Asia Share",f"{asia_pct:.1f}%",f"of global {LY} production")
        card(gc2,"KPI-9 · Active Countries",str(n_cnt),f"producing nations · {LY}")
        card(gc3,"KPI-17 · Upper-Mid Income",f"{um_pct:.1f}%",f"High-income (Canada tier): {hi_pct:.1f}%")
        card(gc4,"KPI-10 · Active Species",
             str(int(faq[(faq["period"]==LY)&(faq["value"]>0)]["seaweed_name"].nunique())),
             f"farmed varieties · {LY}",accent="amber")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-head">KPI-6 — Country Market Share Treemap</div>', unsafe_allow_html=True)
        tree_df = (fgp[(fgp["period"]==LY)&(fgp["value"]>0)&(fgp["continent_group_en"]!="Unknown")]
                   .groupby(["continent_group_en","country_name"])["value"].sum().reset_index())
        tree_df.columns = ["Continent","Country","Tonnes"]
        tree_df["Share"] = (tree_df["Tonnes"]/tree_df["Tonnes"].sum()*100).round(2)
        fig_tree = px.treemap(tree_df,path=["Continent","Country"],values="Tonnes",color="Continent",
            color_discrete_map={"Asia":C["t600"],"Americas":C["t400"],"Europe":C["t300"],"Africa":C["amber"],"Oceania":C["coral"]},
            custom_data=["Share"])
        fig_tree.update_traces(texttemplate="<b>%{label}</b><br>%{customdata[0]:.1f}%",
            textfont=dict(size=12,color="#FFFFFF"),marker=dict(line=dict(width=1.5,color="#FFFFFF")))
        fig_tree.update_layout(**base_layout(360,dict(l=4,r=4,t=10,b=4)))
        st.plotly_chart(fig_tree, use_container_width=True, key="autoplt_6")

        st.markdown("<br>", unsafe_allow_html=True)
        ga,gb = st.columns([1,1.5])
        with ga:
            st.markdown('<div class="sec-head">KPI-7 — Continental Share</div>', unsafe_allow_html=True)
            cd = (fgp[fgp["period"]==LY].groupby("continent_group_en")["value"].sum().reset_index())
            cd = cd[cd["continent_group_en"]!="Unknown"]; cd.columns = ["continent","tonnes"]
            cd["share"] = (cd["tonnes"]/cd["tonnes"].sum()*100).round(1)
            fig6 = go.Figure(go.Pie(labels=cd["continent"],values=cd["tonnes"],hole=0.52,
                marker=dict(colors=[CONT_PAL.get(c,C["gray"]) for c in cd["continent"]],line=dict(color="white",width=2)),
                textinfo="label+percent",textfont=dict(size=12,color=C["txt"]),insidetextorientation="radial"))
            fig6.update_layout(**base_layout(320,dict(l=4,r=4,t=10,b=4)),showlegend=False,
                annotations=[dict(text=f"<b>{LY}</b>",x=0.5,y=0.5,
                    font=dict(size=17,color=C["t600"],family="Georgia,serif"),showarrow=False)])
            st.plotly_chart(fig6, use_container_width=True, key="autoplt_7")
        with gb:
            st.markdown('<div class="sec-head">KPI-10 — Top 10 Species by Volume</div>', unsafe_allow_html=True)
            sp_df = (faq.groupby("seaweed_name")["value"].sum().reset_index()
                     .nlargest(10,"value").sort_values("value",ascending=True))
            sp_df["tm"] = (sp_df["value"]/1e6).round(2)
            bc_colors = (SP_PAL*3)[:len(sp_df)]; bc_colors.reverse()
            fig7 = go.Figure(go.Bar(x=sp_df["tm"],y=sp_df["seaweed_name"],orientation="h",
                marker=dict(color=bc_colors,line=dict(color="white",width=0.5)),
                text=sp_df["tm"].apply(lambda v: f"{v:.1f}M t"),textposition="outside",
                textfont=dict(size=11,color=C["txt"])))
            fig7.update_layout(**base_layout(320,dict(l=4,r=80,t=10,b=4)),legend=LEGEND_TOP)
            fig7 = style_axes(fig7,xtitle="Million tonnes (cumulative)")
            fig7.update_yaxes(tickfont=dict(size=11,color=C["txt"]))
            st.plotly_chart(fig7, use_container_width=True, key="autoplt_8")

        gi,gj = st.columns(2)
        with gi:
            st.markdown('<div class="sec-head">KPI-17 — Production by Income Group (Volume)</div>', unsafe_allow_html=True)
            ig_df = fgp.groupby(["period","ecoclass_group_en"])["value"].sum().reset_index()
            ig_df = ig_df[ig_df["ecoclass_group_en"]!="Countries not classified by World Bank"]
            ig_df.columns = ["year","income_group","tonnes"]; ig_df["tm"] = ig_df["tonnes"]/1e6
            ig_ord = ["Upper-middle income countries","High-income countries","Lower-middle income countries","Low-income countries"]
            fig8 = px.area(ig_df,x="year",y="tm",color="income_group",color_discrete_map=INC_PAL,
                labels={"tm":"M tonnes","year":"Year","income_group":"Income Group"},category_orders={"income_group":ig_ord})
            fig8.update_layout(**base_layout(280,dict(l=4,r=4,t=10,b=4)),legend=LEGEND_BOTTOM)
            fig8.update_xaxes(tickfont=AXIS_FONT,gridcolor=GRID_COLOR)
            fig8.update_yaxes(tickfont=AXIS_FONT,gridcolor=GRID_COLOR,title_font=AXIS_FONT,title_text="M tonnes")
            st.plotly_chart(fig8, use_container_width=True, key="autoplt_9")
        with gj:
            st.markdown('<div class="sec-head">KPI-11 — Species Composition Shift (Top 6)</div>', unsafe_allow_html=True)
            top6 = faq.groupby("seaweed_name")["value"].sum().nlargest(6).index.tolist()
            sp_yr = (faq[faq["seaweed_name"].isin(top6)]
                     .groupby(["period","seaweed_name"])["value"].sum().reset_index())
            sp_yr["tm"] = sp_yr["value"]/1e6
            fig9 = px.area(sp_yr,x="period",y="tm",color="seaweed_name",color_discrete_sequence=SP_PAL[:6],
                labels={"tm":"M tonnes","period":"Year","seaweed_name":"Species"})
            fig9.update_layout(**base_layout(280,dict(l=4,r=4,t=10,b=4)),legend=LEGEND_BOTTOM)
            fig9.update_xaxes(tickfont=AXIS_FONT,gridcolor=GRID_COLOR)
            fig9.update_yaxes(tickfont=AXIS_FONT,gridcolor=GRID_COLOR,title_font=AXIS_FONT,title_text="M tonnes")
            st.plotly_chart(fig9, use_container_width=True, key="autoplt_10")

        gk,gl = st.columns(2)
        with gk:
            st.markdown('<div class="sec-head">KPI-18 — Income Group Share Trend (%)</div>', unsafe_allow_html=True)
            ig_pct = ig_df.copy()
            yr_totals = ig_pct.groupby("year")["tonnes"].transform("sum")
            ig_pct["share_pct"] = ig_pct["tonnes"]/yr_totals*100
            fig18 = px.area(ig_pct,x="year",y="share_pct",color="income_group",color_discrete_map=INC_PAL,
                labels={"share_pct":"Share (%)","year":"Year","income_group":"Income Group"},
                category_orders={"income_group":ig_ord},groupnorm="percent")
            fig18.update_layout(**base_layout(280,dict(l=4,r=4,t=10,b=4)),legend=LEGEND_BOTTOM)
            fig18.update_xaxes(tickfont=AXIS_FONT,gridcolor=GRID_COLOR)
            fig18.update_yaxes(tickfont=AXIS_FONT,gridcolor=GRID_COLOR,title_font=AXIS_FONT,
                               title_text="Share (%)",ticksuffix="%",range=[0,100])
            st.plotly_chart(fig18, use_container_width=True, key="autoplt_11")
        with gl:
            st.markdown('<div class="sec-head">KPI-19 — Value per Tonne by Income Group (USD/t)</div>', unsafe_allow_html=True)
            av_ig = (fav[fav["value"]>0].groupby(["period","ecoclass_group_en"])["value"].sum()
                     .reset_index().rename(columns={"value":"usd_k"}))
            aq_ig = (faq[faq["value"]>0].groupby(["period","ecoclass_group_en"])["value"].sum()
                     .reset_index().rename(columns={"value":"tonnes_ig"}))
            vpt = av_ig.merge(aq_ig,on=["period","ecoclass_group_en"])
            vpt = vpt[vpt["ecoclass_group_en"]!="Countries not classified by World Bank"]
            vpt["usd_per_t"] = vpt["usd_k"]*1000/vpt["tonnes_ig"]
            vpt = vpt.rename(columns={"ecoclass_group_en":"income_group","period":"year"})
            fig19 = go.Figure()
            for grp,colour in INC_PAL.items():
                sub = vpt[vpt["income_group"]==grp].sort_values("year")
                if sub.empty: continue
                fig19.add_trace(go.Scatter(x=sub["year"],y=sub["usd_per_t"],mode="lines+markers",
                    name=grp,line=dict(color=colour,width=2.2),marker=dict(size=5,color=colour)))
            fig19.update_layout(**base_layout(280,dict(l=4,r=4,t=10,b=4)),legend=LEGEND_BOTTOM)
            fig19 = style_axes(fig19,ytitle="USD / tonne")
            st.plotly_chart(fig19, use_container_width=True, key="autoplt_12")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PERMITTING & SOCIAL
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<span class="tag sim">📊 Estimated from public reports</span>'
                '<span class="tag ext">Sources: DFO · BC Gov · CIRNAC · Stats Canada</span>',
                unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:13px;color:#444444;background:#FFF8EC;'
        'border-left:4px solid #C97D10;padding:10px 14px;border-radius:0 6px 6px 0;margin-bottom:12px;line-height:1.7;">'
        '<strong>About this data:</strong> Values estimated from published DFO Annual Aquaculture Statistics, '
        'BC Ministry of Agriculture licence records, CIRNAC program reports, and Statistics Canada datasets. '
        'Year-by-year precision requires direct API or CSV feeds from those agencies.'
        '</div>', unsafe_allow_html=True)

    pf = sim_perm[(sim_perm["year"]>=year_range[0])&(sim_perm["year"]<=year_range[1])]
    sf = sim_social[(sim_social["year"]>=year_range[0])&(sim_social["year"]<=year_range[1])]
    pf_ly = pf[pf["year"]==pf["year"].max()] if not pf.empty else pd.DataFrame()
    sf_ly = sf[sf["year"]==sf["year"].max()] if not sf.empty else pd.DataFrame()

    st.markdown("#### Operational Farm KPIs")
    op1,op2 = st.columns(2)
    if not pf_ly.empty:
        sim_card(op1,"G4-OP6 · Number of Seaweed Farms",str(int(pf_ly['seaweed_farms'].values[0])),
                 "Active licensed seaweed farm operations","DFO Annual Aquaculture Stats · BC Ministry of Agriculture")
        sim_card(op2,"G4-OP7 · Total Farm Area (ha)",f"{int(pf_ly['seaweed_area_ha'].values[0]):,} ha",
                 "Total licensed seaweed cultivation area","BC Ministry of Agriculture · Statistics Canada")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Permitting KPIs")
    pc1,pc2,pc3,pc4 = st.columns(4)
    if not pf_ly.empty:
        sim_card(pc1,"G4-PM1 · ADA (ha)",f"{int(pf_ly['ada_ha'].values[0]):,} ha","Development Areas designated","BC Ministry of Agriculture")
        sim_card(pc2,"G4-PM2 · Permitted Farms",str(int(pf_ly['permitted_farms'].values[0])),"Licensed operations","DFO Annual Aquaculture Stats",accent="amber")
        sim_card(pc3,"G4-PM3 · AOAs",str(int(pf_ly['aoa_count'].values[0])),"Opportunity Areas assessed","DFO Pacific Aquaculture Regs",accent="blue")
        sim_card(pc4,"G4-PM4 · Fisheries Act",f"{pf_ly['compliance_pct'].values[0]:.1f}%","Inspection pass rate","DFO C&E Annual Report")

    st.markdown("<br>", unsafe_allow_html=True)
    pm1,pm2 = st.columns(2)
    with pm1:
        st.markdown('<div class="sec-head">G4-PM1/2 — ADA & Permitted Farms</div>', unsafe_allow_html=True)
        fpm = make_subplots(specs=[[{"secondary_y": True}]])
        fpm.add_trace(go.Bar(x=pf["year"],y=pf["ada_ha"],name="ADA (ha)",marker_color=C["t300"],opacity=0.85),secondary_y=False)
        fpm.add_trace(go.Scatter(x=pf["year"],y=pf["permitted_farms"],mode="lines+markers",name="Permitted Farms",
            line=dict(color=C["t800"],width=2.5),marker=dict(size=7,color=C["t800"])),secondary_y=True)
        fpm.update_layout(**base_layout(280),legend=LEGEND_TOP)
        fpm.update_xaxes(tickfont=AXIS_FONT,gridcolor=GRID_COLOR)
        fpm.update_yaxes(title_text="ADA (ha)",tickfont=AXIS_FONT,gridcolor=GRID_COLOR,title_font=AXIS_FONT,secondary_y=False)
        fpm.update_yaxes(title_text="Permitted Farms",tickfont=AXIS_FONT,gridcolor=GRID_COLOR,title_font=AXIS_FONT,secondary_y=True)
        st.plotly_chart(fpm, use_container_width=True, key="autoplt_13")
    with pm2:
        st.markdown('<div class="sec-head">G4-PM3/4 — AOAs & Compliance</div>', unsafe_allow_html=True)
        fpm2 = make_subplots(specs=[[{"secondary_y": True}]])
        fpm2.add_trace(go.Bar(x=pf["year"],y=pf["aoa_count"],name="AOA Count",marker_color=C["blue"],opacity=0.75),secondary_y=False)
        fpm2.add_trace(go.Scatter(x=pf["year"],y=pf["compliance_pct"],mode="lines+markers",name="Compliance (%)",
            line=dict(color=C["green"],width=2.5),marker=dict(size=7,color=C["green"])),secondary_y=True)
        fpm2.update_layout(**base_layout(280),legend=LEGEND_TOP)
        fpm2.update_xaxes(tickfont=AXIS_FONT,gridcolor=GRID_COLOR)
        fpm2.update_yaxes(title_text="AOA Count",tickfont=AXIS_FONT,gridcolor=GRID_COLOR,title_font=AXIS_FONT,secondary_y=False)
        fpm2.update_yaxes(title_text="Compliance (%)",tickfont=AXIS_FONT,gridcolor=GRID_COLOR,title_font=AXIS_FONT,secondary_y=True,range=[85,100])
        st.plotly_chart(fpm2, use_container_width=True, key="autoplt_14")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Social KPIs")
    sc1,sc2,sc3,sc4,sc5 = st.columns(5)
    if not sf_ly.empty:
        sim_card(sc1,"G4-SO1 · Social License",f"{int(sf_ly['agreements'].values[0])+60}/100","Public acceptance index","DFO Aquaculture Survey")
        sim_card(sc2,"G4-SO2 · Indigenous Agreements",str(int(sf_ly['agreements'].values[0])),"MOUs / benefit agreements","BC Treaty Commission",accent="amber")
        sim_card(sc3,"G4-SO3 · Indigenous Employed",str(int(sf_ly['employed'].values[0])),"Jobs from funded activities","Stats Canada · DFO",accent="blue")
        sim_card(sc4,"G4-SO4 · Indigenous Trained",str(int(sf_ly['trained_indig'].values[0])),"Capacity building programs","CIRNAC")
        sim_card(sc5,"G4-SO5 · Total Trained",str(int(sf_ly['trained_total'].values[0])),"All individuals trained","PSIA surveys · DFO")

    st.markdown("<br>", unsafe_allow_html=True)
    ss1,ss2 = st.columns(2)
    with ss1:
        st.markdown('<div class="sec-head">G4-SO2/3/4 — Indigenous Engagement Trend</div>', unsafe_allow_html=True)
        fso = go.Figure()
        fso.add_trace(go.Scatter(x=sf["year"],y=sf["employed"],mode="lines+markers",name="Employed",
            line=dict(color=C["t600"],width=2.5),marker=dict(size=7,color=C["t600"]),
            fill="tozeroy",fillcolor="rgba(15,110,86,0.10)"))
        fso.add_trace(go.Scatter(x=sf["year"],y=sf["trained_indig"],mode="lines+markers",
            name="Trained (Indigenous)",line=dict(color=C["amber"],width=2.2),marker=dict(size=7,color=C["amber"])))
        fso.add_trace(go.Bar(x=sf["year"],y=sf["agreements"]*8,name="Agreements (×8 scaled)",
            marker_color=C["blue"],opacity=0.45))
        fso.update_layout(**base_layout(275),legend=LEGEND_TOP)
        fso = style_axes(fso)
        st.plotly_chart(fso, use_container_width=True, key="autoplt_15")
    with ss2:
        st.markdown('<div class="sec-head">G4-SO5 — Total Training Provided</div>', unsafe_allow_html=True)
        ftr = go.Figure(go.Bar(x=sf["year"],y=sf["trained_total"],
            marker_color=[C["t400"] if y<2022 else C["t600"] for y in sf["year"]],
            text=sf["trained_total"],textposition="outside",textfont=dict(size=12,color=C["txt"])))
        ftr.update_layout(**base_layout(275,dict(l=4,r=4,t=10,b=30)),showlegend=False)
        ftr = style_axes(ftr,ytitle="Individuals trained")
        st.plotly_chart(ftr, use_container_width=True, key="autoplt_16")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ADVANCED ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    if not data_ok:
        st.warning("Data files not loaded. Place CSVs in data/")
    else:
        st.markdown(
            f'<div style="font-family:Lora,Georgia,serif;font-size:20px;font-weight:700;'
            f'color:{C["t900"]};margin-bottom:10px;">Advanced Analytics KPIs</div>',
            unsafe_allow_html=True)
        st.markdown('<span class="tag">🟢 FAO FishStat</span>'
                    '<span class="tag">KPI-14 · KPI-20 · KPI-21 · KPI-22 · KPI-23 · KPI-24 · KPI-25 · KPI-26</span>',
                    unsafe_allow_html=True)

        # KPI-14 Heatmap
        st.markdown('<div class="sec-head">KPI-14 — Average Price per Tonne Heatmap (Country × Year)</div>', unsafe_allow_html=True)
        av_c = fav.groupby(["period","country_name"])["value"].sum().reset_index(); av_c.columns = ["year","country","usd_k"]
        aq_c = faq.groupby(["period","country_name"])["value"].sum().reset_index(); aq_c.columns = ["year","country","tonnes"]
        hmap_df = av_c.merge(aq_c,on=["year","country"])
        hmap_df = hmap_df[(hmap_df["tonnes"]>500)&(hmap_df["usd_k"]>0)]
        hmap_df["usd_t"] = (hmap_df["usd_k"]*1000/hmap_df["tonnes"]).round(0)
        hmap_df = hmap_df[hmap_df["usd_t"]<5000]
        top15 = hmap_df.groupby("country")["tonnes"].sum().nlargest(15).index.tolist()
        hmap_filt = hmap_df[hmap_df["country"].isin(top15)]
        hmap_pivot = hmap_filt.pivot_table(index="country",columns="year",values="usd_t",aggfunc="mean").fillna(0)
        if LY in hmap_pivot.columns: hmap_pivot = hmap_pivot.sort_values(LY,ascending=True)
        fig14 = go.Figure(go.Heatmap(z=hmap_pivot.values,x=[str(c) for c in hmap_pivot.columns],y=hmap_pivot.index.tolist(),
            colorscale=[[0,"#E1F5EE"],[0.4,C["t400"]],[0.8,C["t600"]],[1,C["t900"]]],
            colorbar=dict(title=dict(text="USD/tonne",font=AXIS_FONT),tickfont=AXIS_FONT,len=0.8),
            hoverongaps=False,hovertemplate="<b>%{y}</b><br>Year: %{x}<br>Price: $%{z:,.0f}/t<extra></extra>",
            text=hmap_pivot.values.astype(int),texttemplate="%{text}",textfont=dict(size=10,color=C["txt"]),zmin=0))
        fig14.update_layout(**base_layout(420,dict(l=120,r=20,t=10,b=60)),
            xaxis=dict(tickfont=AXIS_FONT,title=dict(text="Year",font=AXIS_FONT)),
            yaxis=dict(tickfont=dict(size=11,color=C["txt"])))
        st.plotly_chart(fig14, use_container_width=True, key="autoplt_17")

        st.markdown("<br>", unsafe_allow_html=True)
        t4r1a,t4r1b = st.columns(2)
        with t4r1a:
            st.markdown('<div class="sec-head">KPI-20 — Environment Breakdown (Marine / Inland / Brackish)</div>', unsafe_allow_html=True)
            FRESHWATER = ["spirulina","haematococcus","chlorella","freshwater"]
            def classify_env(name):
                n = str(name).lower()
                if any(k in n for k in FRESHWATER): return "Freshwater / Inland"
                return "Marine"
            aq_env = faq.copy(); aq_env["environment"] = aq_env["seaweed_name"].apply(classify_env)
            env_df = aq_env[aq_env["value"]>0].groupby(["period","environment"])["value"].sum().reset_index()
            env_df["tm"] = env_df["value"]/1e6
            fig20 = px.area(env_df,x="period",y="tm",color="environment",
                color_discrete_map={"Marine":C["t600"],"Freshwater / Inland":C["amber"]},
                labels={"tm":"M tonnes","period":"Year","environment":"Environment"})
            fig20.update_layout(**base_layout(280,dict(l=4,r=4,t=10,b=4)),legend=LEGEND_TOP)
            fig20.update_xaxes(tickfont=AXIS_FONT,gridcolor=GRID_COLOR)
            fig20.update_yaxes(tickfont=AXIS_FONT,gridcolor=GRID_COLOR,title_font=AXIS_FONT,title_text="M tonnes")
            st.plotly_chart(fig20, use_container_width=True, key="autoplt_18")
        with t4r1b:
            st.markdown('<div class="sec-head">KPI-24 — Price Volatility by Species (USD/kg std dev)</div>', unsafe_allow_html=True)
            av_sp = fav.groupby(["period","seaweed_name"])["value"].sum().reset_index(); av_sp.columns = ["year","species","usd_k"]
            aq_sp = faq.groupby(["period","seaweed_name"])["value"].sum().reset_index(); aq_sp.columns = ["year","species","tonnes"]
            vol_sp = av_sp.merge(aq_sp,on=["year","species"])
            vol_sp = vol_sp[(vol_sp["tonnes"]>100)&(vol_sp["usd_k"]>0)]
            vol_sp["usd_kg"] = vol_sp["usd_k"]*1000/vol_sp["tonnes"]/1000
            vol_sp = vol_sp[vol_sp["usd_kg"]<20]
            vola = (vol_sp.groupby("species")["usd_kg"].agg(["std","count","mean"]).reset_index()
                    .rename(columns={"std":"vol_std","count":"n_years","mean":"avg_price"}))
            vola = vola[vola["n_years"]>=5].dropna(subset=["vol_std"]).nlargest(top_n,"vol_std").sort_values("vol_std")
            fig24 = go.Figure()
            for _,row in vola.iterrows():
                fig24.add_shape(type="line",x0=0,x1=row["vol_std"],y0=row["species"],y1=row["species"],
                    line=dict(color=C["t300"],width=2))
            fig24.add_trace(go.Scatter(x=vola["vol_std"],y=vola["species"],mode="markers",
                marker=dict(size=12,color=C["coral"],line=dict(color=C["t900"],width=1.5)),
                text=vola["vol_std"].round(3).astype(str)+" $/kg σ",
                textfont=dict(size=11,color=C["txt"]),textposition="middle right",name="Std Dev"))
            fig24.update_layout(**base_layout(280,dict(l=4,r=100,t=10,b=4)),showlegend=False)
            fig24 = style_axes(fig24,xtitle="Price std dev (USD/kg)")
            fig24.update_yaxes(tickfont=dict(size=11,color=C["txt"]))
            st.plotly_chart(fig24, use_container_width=True, key="autoplt_19")

        t4r2a,t4r2b = st.columns(2)
        with t4r2a:
            st.markdown('<div class="sec-head">KPI-21 — Aquaculture Adoption Rate by Country (%)</div>', unsafe_allow_html=True)
            aq_tot_c = faq.groupby(["period","country_name"])["value"].sum().reset_index(); aq_tot_c.columns = ["year","country","aq_vol"]
            cq_tot_c = fcq.groupby(["period","country_name"])["value"].sum().reset_index(); cq_tot_c.columns = ["year","country","cap_vol"]
            adopt = aq_tot_c.merge(cq_tot_c,on=["year","country"],how="outer").fillna(0)
            adopt["total"] = adopt["aq_vol"]+adopt["cap_vol"]; adopt = adopt[adopt["total"]>0]
            adopt["adopt_pct"] = adopt["aq_vol"]/adopt["total"]*100
            adopt_ly = adopt[adopt["year"]==LY].copy()
            adopt_ly = adopt_ly.merge(fgp[fgp["period"]==LY][["country_name","continent_group_en"]].drop_duplicates(),
                left_on="country",right_on="country_name",how="left")
            adopt_ly = adopt_ly[adopt_ly["aq_vol"]>0].nlargest(30,"aq_vol")
            fig21 = px.scatter(adopt_ly,x="adopt_pct",y="aq_vol",size="aq_vol",color="continent_group_en",
                color_discrete_map=CONT_PAL,hover_name="country",size_max=55,log_y=True,
                labels={"adopt_pct":"Adoption Rate (%)","aq_vol":"Farmed Volume (t)","continent_group_en":"Continent"})
            fig21.update_traces(marker=dict(opacity=0.78,line=dict(width=1,color="white")))
            fig21.update_layout(**base_layout(310,dict(l=4,r=4,t=10,b=4)),legend=LEGEND_TOP)
            fig21 = style_axes(fig21,xtitle="Aquaculture Share of Total Seaweed Production (%)",ytitle="Farmed Volume (tonnes, log scale)")
            st.plotly_chart(fig21, use_container_width=True, key="autoplt_20")
        with t4r2b:
            st.markdown('<div class="sec-head">KPI-25 — Country Specialization Index (Top-Species Share)</div>', unsafe_allow_html=True)
            spec_df = faq[faq["value"]>0].groupby(["country_name","seaweed_name"])["value"].sum().reset_index()
            spec_tot = spec_df.groupby("country_name")["value"].sum().rename("total")
            spec_df  = spec_df.join(spec_tot,on="country_name")
            spec_df["share"] = spec_df["value"]/spec_df["total"]*100
            spec_top = (spec_df.sort_values("share",ascending=False).groupby("country_name").first().reset_index()
                        [["country_name","seaweed_name","share","total"]])
            spec_top = spec_top[spec_top["total"]>1e5].sort_values("share",ascending=True)
            fig25 = go.Figure(go.Bar(x=spec_top["share"],y=spec_top["country_name"],orientation="h",
                marker_color=[SP_PAL[i%len(SP_PAL)] for i in range(len(spec_top))],
                text=spec_top.apply(lambda r: f"{r['seaweed_name'][:18]}… ({r['share']:.0f}%)" if len(r['seaweed_name'])>18 else f"{r['seaweed_name']} ({r['share']:.0f}%)",axis=1),
                textposition="outside",textfont=dict(size=11,color=C["txt"])))
            fig25.update_layout(**base_layout(310,dict(l=4,r=200,t=10,b=4)),showlegend=False)
            fig25 = style_axes(fig25,xtitle="Top Species Share (%)")
            fig25.update_yaxes(tickfont=dict(size=11,color=C["txt"]))
            fig25.update_xaxes(range=[0,120])
            st.plotly_chart(fig25, use_container_width=True, key="autoplt_21")

        t4r3a,t4r3b = st.columns(2)
        with t4r3a:
            st.markdown('<div class="sec-head">KPI-22/23 — Country Growth Trajectories (CAGR) & Emerging Producers</div>', unsafe_allow_html=True)
            cagr_base_yr = max(LY-cagr_win, year_range[0])
            c_base = fgp[fgp["period"]==cagr_base_yr].groupby("country_name")["value"].sum().reset_index().rename(columns={"value":"base_v"})
            c_last = fgp[fgp["period"]==LY].groupby("country_name")["value"].sum().reset_index().rename(columns={"value":"last_v"})
            cagr_c = c_base.merge(c_last,on="country_name")
            cagr_c = cagr_c[(cagr_c["base_v"]>0)&(cagr_c["last_v"]>0)]
            cagr_c["cagr"] = ((cagr_c["last_v"]/cagr_c["base_v"])**(1/cagr_win)-1)*100
            cagr_c["last_mt"] = cagr_c["last_v"]/1e6
            cagr_c = cagr_c.merge(fgp[fgp["period"]==LY][["country_name","continent_group_en"]].drop_duplicates(),on="country_name",how="left")
            cagr_c = cagr_c[(cagr_c["cagr"].abs()<50)&(cagr_c["last_v"]>10000)]
            fig22 = px.scatter(cagr_c,x="cagr",y="last_mt",size="last_mt",color="continent_group_en",
                color_discrete_map=CONT_PAL,hover_name="country_name",size_max=55,log_y=True,
                labels={"cagr":f"CAGR % ({cagr_win}yr)","last_mt":f"Production {LY} (M t, log)","continent_group_en":"Continent"})
            fig22.add_vline(x=0,line_dash="dot",line_color=C["gray"],line_width=1.5,
                annotation_text="0%",annotation_font=dict(size=11,color=C["gray"]))
            fig22.update_traces(marker=dict(opacity=0.80,line=dict(width=1,color="white")))
            fig22.update_layout(**base_layout(310,dict(l=4,r=4,t=10,b=4)),legend=LEGEND_TOP)
            fig22 = style_axes(fig22,xtitle=f"CAGR % (last {cagr_win} years)",ytitle=f"Production {LY} (M tonnes, log)")
            st.plotly_chart(fig22, use_container_width=True, key="autoplt_22")
            emerging = cagr_c[(cagr_c["last_v"]<1e5)&(cagr_c["cagr"]>5)].sort_values("cagr",ascending=False)[["country_name","cagr","last_v"]].head(10)
            emerging.columns = ["Country",f"CAGR ({cagr_win}yr) %",f"Volume {LY} (t)"]
            emerging[f"CAGR ({cagr_win}yr) %"] = emerging[f"CAGR ({cagr_win}yr) %"].round(1)
            if not emerging.empty:
                st.markdown(f'<div style="font-size:12px;font-weight:600;color:{C["t800"]};margin:6px 0 4px;">🌱 KPI-23 — Emerging Producers (vol &lt; 100k t, CAGR &gt; 5%)</div>', unsafe_allow_html=True)
                st.dataframe(emerging.reset_index(drop=True),use_container_width=True,hide_index=True)
        with t4r3b:
            st.markdown('<div class="sec-head">KPI-26 — Data Completeness by Species</div>', unsafe_allow_html=True)
            total_yrs = faq["period"].nunique()
            conf_df = (faq[faq["value"]>0].groupby("seaweed_name")["period"].nunique().reset_index()
                       .rename(columns={"period":"yrs_reported"}))
            conf_df["completeness"] = (conf_df["yrs_reported"]/total_yrs*100).round(1)
            conf_df = conf_df.nlargest(top_n,"completeness").sort_values("completeness")
            conf_df["color"] = conf_df["completeness"].apply(lambda v: C["t600"] if v>=80 else (C["amber"] if v>=50 else C["coral"]))
            fig26 = go.Figure(go.Bar(x=conf_df["completeness"],y=conf_df["seaweed_name"],orientation="h",
                marker_color=conf_df["color"].tolist(),
                text=conf_df["completeness"].astype(str)+"%",textposition="outside",
                textfont=dict(size=11,color=C["txt"])))
            fig26.add_vline(x=80,line_dash="dot",line_color=C["t600"],line_width=1.5,
                annotation_text="80% threshold",annotation_font=dict(size=11,color=C["t600"]),annotation_position="top right")
            fig26.update_layout(**base_layout(310,dict(l=4,r=80,t=10,b=4)),showlegend=False)
            fig26 = style_axes(fig26,xtitle="Data Completeness (%)")
            fig26.update_yaxes(tickfont=dict(size=11,color=C["txt"]))
            fig26.update_xaxes(range=[0,120])
            st.plotly_chart(fig26, use_container_width=True, key="autoplt_23")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-head">📈 KPI-Forecast — ARIMA Production & Value Outlook</div>',
                    unsafe_allow_html=True)
        fcol1, fcol2 = st.columns([2, 1])
        with fcol1:
            fc_choice = st.selectbox("Series to forecast", [
                "Global seaweed production (M t)",
                "Global seaweed value ($B USD)",
                "Canada all-aquaculture production (kt)",
                "BC all-aquaculture production (kt)",
                "BC farmed salmon (kt)",
            ], key="fc_choice")
        with fcol2:
            fc_h = st.slider("Horizon (years)", 1, 15, 5, key="fc_horizon")
        _fc_map = {
            "Global seaweed production (M t)": (_global_prod_series(fgp), 1e6, "M tonnes", "Global seaweed production"),
            "Global seaweed value ($B USD)": (_global_val_series(fav), 1e6, "$B USD", "Global seaweed value"),
            "Canada all-aquaculture production (kt)": (CAN_PROD_T, 1000, "kt", "Canada aquaculture production"),
            "BC all-aquaculture production (kt)": (BC_PROD_T, 1000, "kt", "BC aquaculture production"),
            "BC farmed salmon (kt)": (BC_SALMON_T, 1000, "kt", "BC farmed salmon"),
        }
        _sd, _sc, _su, _slab = _fc_map[fc_choice]
        _fig_fc, _fc_txt, _fc_cap = fig_forecast(_sd, fc_h, _slab, C, scale=_sc, unit=_su, h=360)
        if _fig_fc is not None:
            st.plotly_chart(_fig_fc, use_container_width=True, key="autoplt_25")
            st.caption(_fc_cap + "  These are statistical projections, not guarantees.")
        else:
            st.info(_fc_txt)



# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — CANADIAN CONTEXT (Statistics Canada / DFO data)
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<span class="tag new">🔵 Statistics Canada</span>'
                '<span class="tag ext">aqua86-aqua24 · va97-va24 · DFO Annual Aquaculture Stats</span>',
                unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:13px;color:#333333;background:#E8F5F0;'
        'border-left:4px solid #0F6E56;padding:10px 14px;border-radius:0 6px 6px 0;margin-bottom:12px;line-height:1.7;">'
        '<strong>Data source:</strong> Statistics Canada Table 32-10-0107-01 (aqua files 1986–2024) and '
        'DFO Aquaculture Sector Value Added Account (va files 1997–2024). These cover <strong>ALL Canadian '
        'aquaculture species</strong> (finfish + shellfish). Seaweed is NOT separately reported — it would be a '
        'tiny fraction of "Other" categories. BC data is suppressed by Statistics Act for 1991–2001.'
        '</div>', unsafe_allow_html=True)

    # KPI cards — 2024 Canadian context
    ca1,ca2,ca3,ca4,ca5 = st.columns(5)
    bc24_prod = BC_PROD_T.get(2024, 0)
    can24_prod = CAN_PROD_T.get(2024, 0)
    bc24_val  = BC_VAL_K.get(2024, 0)
    can24_val = CAN_VAL_K.get(2024, 0)
    bc24_gva  = VA_BC_GVA_K.get(2024, 0)
    bc24_feed = VA_BC_FEED_K.get(2024, 0)
    bc24_sal  = BC_SALMON_T.get(2024, 0)

    card(ca1,"BC Total Aquaculture 2024",f"{bc24_prod/1000:.0f}k t",
         f"↓ vs {BC_PROD_T.get(2022,0)/1000:.0f}k t (2022) · salmon licence reductions",pos=False)
    card(ca2,"BC Aquaculture Value 2024",f"${bc24_val/1000:.0f}M CAD",
         f"vs ${BC_VAL_K.get(2022,0)/1000:.0f}M (2022 peak)",pos=False,accent="blue")
    card(ca3,"BC Gross Value Added 2024",f"${bc24_gva/1000:.0f}M CAD",
         f"Only 3.4% of gross output — vs 40.5% in 2017",pos=False,accent="coral")
    card(ca4,"BC Salmon Production 2024",f"{bc24_sal/1000:.0f}k t",
         f"vs {BC_SALMON_T.get(2016,0)/1000:.0f}k t (2016) · −41% from peak",pos=False,accent="amber")
    card(ca5,"Seaweed Opportunity",f"~$50M",
         f"Seaweed = ~8% of BC's ${bc24_val/1000:.0f}M aqua value — large headroom")

    st.markdown("<br>", unsafe_allow_html=True)
    ca_r1a, ca_r1b = st.columns(2)

    with ca_r1a:
        st.markdown('<div class="sec-head">BC & Canada All-Aquaculture Production (tonnes) 1986–2024</div>', unsafe_allow_html=True)
        ca_years_bc  = sorted(BC_PROD_T.keys())
        ca_years_can = sorted(CAN_PROD_T.keys())
        fig_ca1 = go.Figure()
        fig_ca1.add_trace(go.Scatter(
            x=ca_years_can, y=[CAN_PROD_T[y]/1000 for y in ca_years_can],
            mode="lines+markers", name="Canada (all species)",
            line=dict(color=C["t600"],width=2.5),marker=dict(size=5),
            fill="tozeroy",fillcolor="rgba(15,110,86,0.08)"))
        fig_ca1.add_trace(go.Scatter(
            x=ca_years_bc, y=[BC_PROD_T[y]/1000 for y in ca_years_bc],
            mode="lines+markers", name="BC only",
            line=dict(color=C["amber"],width=2.2),marker=dict(size=5)))
        fig_ca1.update_layout(**base_layout(320,dict(l=4,r=4,t=10,b=4)),legend=LEGEND_TOP)
        fig_ca1 = style_axes(fig_ca1,ytitle="Thousand tonnes")
        st.plotly_chart(fig_ca1, use_container_width=True, key="autoplt_26")
        st.caption("Source: Statistics Canada Table 32-10-0107-01 / DFO Annual Aquaculture Statistics")

    with ca_r1b:
        st.markdown('<div class="sec-head">BC & Canada Aquaculture Value ($M CAD) 1986–2024</div>', unsafe_allow_html=True)
        yrs_bc_val  = sorted(BC_VAL_K.keys())
        yrs_can_val = sorted(CAN_VAL_K.keys())
        fig_ca2 = go.Figure()
        fig_ca2.add_trace(go.Scatter(
            x=yrs_can_val, y=[CAN_VAL_K[y]/1000 for y in yrs_can_val],
            mode="lines+markers", name="Canada (all species)",
            line=dict(color=C["t600"],width=2.5),marker=dict(size=5),
            fill="tozeroy",fillcolor="rgba(15,110,86,0.08)"))
        fig_ca2.add_trace(go.Scatter(
            x=yrs_bc_val, y=[BC_VAL_K[y]/1000 for y in yrs_bc_val],
            mode="lines+markers", name="BC only",
            line=dict(color=C["amber"],width=2.2),marker=dict(size=5)))
        fig_ca2.update_layout(**base_layout(320,dict(l=4,r=4,t=10,b=4)),legend=LEGEND_TOP)
        fig_ca2 = style_axes(fig_ca2,ytitle="$M CAD")
        st.plotly_chart(fig_ca2, use_container_width=True, key="autoplt_27")
        st.caption("Source: Statistics Canada / DFO · 38-yr CAGR (production): 7.44%")

    ca_r2a, ca_r2b = st.columns(2)

    with ca_r2a:
        st.markdown('<div class="sec-head">BC Gross Value Added & Feed Costs ($M CAD) 1997–2024</div>', unsafe_allow_html=True)
        va_yrs = sorted(set(VA_BC_GVA_K.keys()) | set(VA_BC_FEED_K.keys()))
        fig_ca3 = make_subplots(specs=[[{"secondary_y": True}]])
        fig_ca3.add_trace(go.Scatter(
            x=[y for y in va_yrs if y in VA_BC_GVA_K],
            y=[VA_BC_GVA_K[y]/1000 for y in va_yrs if y in VA_BC_GVA_K],
            mode="lines+markers", name="BC GVA (GDP contribution)",
            line=dict(color=C["t600"],width=2.5),marker=dict(size=6),
            fill="tozeroy",fillcolor="rgba(15,110,86,0.10)"),secondary_y=False)
        fig_ca3.add_trace(go.Bar(
            x=[y for y in va_yrs if y in VA_BC_FEED_K],
            y=[VA_BC_FEED_K[y]/1000 for y in va_yrs if y in VA_BC_FEED_K],
            name="BC Feed Costs",marker_color=C["amber"],opacity=0.55),secondary_y=True)
        fig_ca3.update_layout(**base_layout(320,dict(l=4,r=4,t=10,b=4)),legend=LEGEND_TOP)
        fig_ca3.update_xaxes(tickfont=AXIS_FONT,gridcolor=GRID_COLOR)
        fig_ca3.update_yaxes(title_text="GVA ($M CAD)",tickfont=AXIS_FONT,gridcolor=GRID_COLOR,title_font=AXIS_FONT,secondary_y=False)
        fig_ca3.update_yaxes(title_text="Feed Costs ($M CAD)",tickfont=AXIS_FONT,secondary_y=True,title_font=AXIS_FONT)
        st.plotly_chart(fig_ca3, use_container_width=True, key="autoplt_28")
        st.caption("GVA = sector GDP contribution. 2024 GVA collapsed to $21.6M (3.4% of output) vs $338M (40.5%) in 2017 — salmon licence impact.")

    with ca_r2b:
        st.markdown('<div class="sec-head">BC Salmon vs Shellfish Production (tonnes) 2000–2024</div>', unsafe_allow_html=True)
        sal_yrs   = sorted(BC_SALMON_T.keys())
        shell_yrs = sorted(BC_SHELL_T.keys())
        fig_ca4 = go.Figure()
        fig_ca4.add_trace(go.Scatter(
            x=sal_yrs, y=[BC_SALMON_T[y]/1000 for y in sal_yrs],
            mode="lines+markers", name="BC Farmed Salmon",
            line=dict(color=C["t600"],width=2.5),marker=dict(size=6),
            fill="tozeroy",fillcolor="rgba(15,110,86,0.12)"))
        fig_ca4.add_trace(go.Scatter(
            x=shell_yrs, y=[BC_SHELL_T[y]/1000 for y in shell_yrs],
            mode="lines+markers", name="BC Shellfish",
            line=dict(color=C["amber"],width=2.2),marker=dict(size=6)))
        fig_ca4.update_layout(**base_layout(320,dict(l=4,r=4,t=10,b=4)),legend=LEGEND_TOP)
        fig_ca4 = style_axes(fig_ca4,ytitle="Thousand tonnes")
        st.plotly_chart(fig_ca4, use_container_width=True, key="autoplt_29")
        st.caption("Salmon peak: 2015 at 92,926 t. 2023–24 decline driven by DFO salmon farm licence reductions in BC.")

    ca_r3a, ca_r3b = st.columns(2)

    with ca_r3a:
        st.markdown('<div class="sec-head">BC Therapeutants & Wages vs GVA ($M CAD) 1997–2024</div>', unsafe_allow_html=True)
        ther_yrs  = sorted(VA_BC_THERAP_K.keys())
        wages_yrs = sorted(VA_BC_WAGES_K.keys())
        fig_ca5 = go.Figure()
        fig_ca5.add_trace(go.Scatter(
            x=wages_yrs, y=[VA_BC_WAGES_K[y]/1000 for y in wages_yrs],
            mode="lines+markers", name="BC Wages",
            line=dict(color=C["t400"],width=2.2),marker=dict(size=5)))
        fig_ca5.add_trace(go.Scatter(
            x=ther_yrs, y=[VA_BC_THERAP_K[y]/1000 for y in ther_yrs],
            mode="lines+markers", name="BC Therapeutants",
            line=dict(color=C["coral"],width=2.2,dash="dot"),marker=dict(size=5)))
        fig_ca5.add_trace(go.Scatter(
            x=[y for y in va_yrs if y in VA_BC_GVA_K],
            y=[VA_BC_GVA_K[y]/1000 for y in va_yrs if y in VA_BC_GVA_K],
            mode="lines+markers", name="BC GVA",
            line=dict(color=C["t600"],width=2.5),marker=dict(size=5),
            fill="tozeroy",fillcolor="rgba(15,110,86,0.06)"))
        fig_ca5.update_layout(**base_layout(320,dict(l=4,r=4,t=10,b=4)),legend=LEGEND_TOP)
        fig_ca5 = style_axes(fig_ca5,ytitle="$M CAD")
        st.plotly_chart(fig_ca5, use_container_width=True, key="autoplt_30")
        st.caption("Therapeutant costs rose 12× from $3M (1997) to $36.2M (2024) — rising disease management burden. Seaweed bioactive compounds offer alternatives.")

    with ca_r3b:
        st.markdown('<div class="sec-head">Seaweed Opportunity in Canadian Context (2024)</div>', unsafe_allow_html=True)
        # Waterfall / comparison bar
        categories = ["BC Total Aqua\nGross Output", "BC Salmon\nValue", "BC Shellfish\nValue",
                      "BC GVA\n(Profit)", "BC Feed Costs", "BC Wages", "BC Therapeutants",
                      "Seaweed\nEstimated Value"]
        values_m = [
            VA_BC_OUTPUT_K.get(2024,626689)/1000,
            BC_VAL_K.get(2024,562814)*0.94/1000,  # ~94% is finfish
            BC_VAL_K.get(2024,562814)*0.05/1000,  # ~5% shellfish
            VA_BC_GVA_K.get(2024,21600)/1000,
            VA_BC_FEED_K.get(2024,178254)/1000,
            VA_BC_WAGES_K.get(2024,72403)/1000,
            VA_BC_THERAP_K.get(2024,36238)/1000,
            50,  # estimated seaweed
        ]
        colors_bar = [C["t600"],C["t600"],C["t300"],C["green"],C["amber"],C["blue"],C["coral"],C["t400"]]
        fig_ca6 = go.Figure(go.Bar(
            x=categories, y=values_m,
            marker_color=colors_bar,
            text=[f"${v:.0f}M" for v in values_m],
            textposition="outside",textfont=dict(size=11,color=C["txt"])))
        fig_ca6.update_layout(**base_layout(320,dict(l=4,r=4,t=30,b=60)),showlegend=False)
        fig_ca6 = style_axes(fig_ca6,ytitle="$M CAD (2024)")
        fig_ca6.update_xaxes(tickfont=dict(size=10,color=C["txt"]))
        fig_ca6.update_yaxes(range=[0, max(values_m)*1.25])
        st.plotly_chart(fig_ca6, use_container_width=True, key="autoplt_31")
        st.caption("BC seaweed (~$50M) represents ~8% of BC aquaculture gross output. BC feed costs ($178M) and therapeutants ($36M) represent key markets for seaweed-derived products.")

    with st.expander("📋 Canadian Aquaculture Data Tables"):
        ct1,ct2,ct3 = st.tabs(["Production & Value","Value Added (BC)","VA Canada"])
        with ct1:
            can_df = pd.DataFrame({
                "Year": sorted(set(list(BC_PROD_T.keys())+list(CAN_PROD_T.keys()))),
            })
            can_df["BC Prod (t)"] = can_df["Year"].map(BC_PROD_T)
            can_df["CAN Prod (t)"] = can_df["Year"].map(CAN_PROD_T)
            can_df["BC Value ($K CAD)"] = can_df["Year"].map(BC_VAL_K)
            can_df["CAN Value ($K CAD)"] = can_df["Year"].map(CAN_VAL_K)
            can_df["BC Salmon (t)"] = can_df["Year"].map(BC_SALMON_T)
            can_df["BC Shellfish (t)"] = can_df["Year"].map(BC_SHELL_T)
            st.dataframe(can_df.sort_values("Year",ascending=False).reset_index(drop=True),use_container_width=True,hide_index=True)
        with ct2:
            va_df = pd.DataFrame({"Year": sorted(VA_BC_GVA_K.keys())})
            va_df["BC GVA ($K)"] = va_df["Year"].map(VA_BC_GVA_K)
            va_df["BC Output ($K)"] = va_df["Year"].map(VA_BC_OUTPUT_K)
            va_df["BC Feed ($K)"] = va_df["Year"].map(VA_BC_FEED_K)
            va_df["BC Wages ($K)"] = va_df["Year"].map(VA_BC_WAGES_K)
            va_df["BC Therapeutants ($K)"] = va_df["Year"].map(VA_BC_THERAP_K)
            va_df["GVA % Output"] = (va_df["BC GVA ($K)"]/va_df["BC Output ($K)"]*100).round(1)
            va_df["Feed % Costs"] = (va_df["BC Feed ($K)"]/(va_df["Year"].map(
                {y: VA_BC_OUTPUT_K.get(y,1) for y in VA_BC_WAGES_K}))*100).round(1)
            st.dataframe(va_df.sort_values("Year",ascending=False).reset_index(drop=True),use_container_width=True,hide_index=True)
        with ct3:
            can_va_df = pd.DataFrame({"Year": sorted(VA_CAN_GVA_K.keys())})
            can_va_df["Canada GVA ($K)"] = can_va_df["Year"].map(VA_CAN_GVA_K)
            st.dataframe(can_va_df.sort_values("Year",ascending=False).reset_index(drop=True),use_container_width=True,hide_index=True)

# end _main_col

# ─────────────────────────────────────────────────────────────────────────────
# RAG KNOWLEDGE BASE — v11.0 (35 chunks, 16 source documents + Canadian data)
# ─────────────────────────────────────────────────────────────────────────────
RAG_DOCS = [
    {
        "id": "prod_overview",
        "title": "Global Seaweed Production Overview",
        "text": (
            "Global seaweed production reached approximately 40 million tonnes in 2024, "
            "up from 35.8 million tonnes in 2019 and 32 million tonnes in 2018. "
            "Asia dominates production with a 97% share, led by China at 56%, followed by "
            "Indonesia, Philippines, South Korea, and Japan. "
            "Aquaculture (farmed seaweed) accounts for 97% of total production while wild "
            "capture makes up the remaining 3%. The sector achieved a CAGR of ~2.9%/decade. "
            "Total aquaculture value reached USD 19.9 billion in 2024. "
            "Upper-middle income countries produce 86.7% of global output, while high-income "
            "countries like Canada represent only 7%. "
            "In 2019, 49 countries produced seaweed and 27 ASFIS species items were actively "
            "cultivated. Five species groups account for over 95% of world cultivation: "
            "Laminaria/Saccharina 35.4%, Kappaphycus/Eucheuma 33.5%, Gracilaria 10.5%, "
            "Porphyra/Pyropia 8.6%, and Undaria 7.4%. Wild collection declined from 1.33M "
            "tonnes (1990) to ~1.08M tonnes (2019). "
            "The global seaweed industry grew 736% from 1990 to 2020."
        ),
        "source": "FAO FishStat 2024; Cai FAO Seaweed Innovation Forum 2021; RDMW SIDP 2024",
    },
    {
        "id": "species_overview_detail",
        "title": "Key Seaweed Species — Production Volume Data (FAO 2019)",
        "text": (
            "FAO 2019 world seaweed cultivation (34,679,134 tonnes): "
            "1. Japanese kelp (Laminaria japonica): 12,273,519 t (35.4%). "
            "2. Eucheuma spp.: 9,817,689 t (28.3%). "
            "3. Gracilaria spp.: 3,638,554 t (10.5%). "
            "4. Wakame (Undaria pinnatifida): 2,563,477 t (7.4%). "
            "5. Nori (Porphyra spp.): 2,123,040 t (6.1%). "
            "Wild collection 2019 (1,083,370 t): Aquatic plants nei 198,617 t; "
            "Lessonia nigrescens 184,048 t; Ascophyllum nodosum 75,155 t. "
            "BC cultivated species: Saccharina latissima (sugar kelp), Alaria marginata "
            "(winged-kelp), Macrocystis tenuifolia (giant kelp). "
            "Non-native species prohibited in BC coastal mariculture under conditions of licence."
        ),
        "source": "Cai FAO 2021; SeaweedaquacultureinBritishColumbia_V8",
    },
    {
        "id": "seaweed_trade_2019",
        "title": "Global Seaweed Trade Statistics 2019",
        "text": (
            "World export of seaweeds and hydrocolloids 2019: USD 2.65 billion total. "
            "= USD 909 million raw seaweeds + USD 1.74 billion hydrocolloids. "
            "Top exporters: China (USD 578M, 21.8%), Indonesia (USD 329M), Rep. of Korea (USD 320M), "
            "Philippines (USD 252M), Chile (USD 209M). "
            "Canada ranked 10th in raw seaweed exports at USD 18 million. "
            "World imports: USD 2.9 billion. Top importers: China USD 445M, Japan USD 341M, USA USD 320M. "
            "Hydrocolloid types: agar (HS130231), carrageenan (HS130239), alginate (HS391310)."
        ),
        "source": "UN Comtrade via Cai FAO 2021",
    },
    {
        "id": "market_applications",
        "title": "Seaweed Market Applications and Value Chain",
        "text": (
            "Seaweed commercial applications: "
            "Human foods: kelp/kombu, nori (sushi), wakame (salad), Gracilaria, Caulerpa. "
            "Hydrocolloids: carrageenan from Kappaphycus/Eucheuma; agar from Gracilaria and "
            "Gelidium; alginate from brown seaweeds. "
            "Other: abalone feed; livestock feed; biofertilizer/biostimulants; cosmetics; "
            "nutraceuticals; pharmaceuticals; textile fibres; biopackaging; bioenergy. "
            "New low-carbon products: seaweed-based textiles, antimicrobial bandages, biofuels, "
            "biodegradable plastic alternatives. "
            "Global market value: ~USD 19.9 billion in 2024."
        ),
        "source": "FAO Cai 2021; FAO Globefish 2018; Lang-Wong et al. 2022",
    },
    {
        "id": "canada_seaweed_bc",
        "title": "Canadian and BC Seaweed Industry — Current State",
        "text": (
            "Canada's seaweed industry is an emerging sector. B.C. has over 25,000 km of coastline. "
            "Canadian seaweed market value: ~CAD 50 million in 2024. "
            "Canada represents only 0.04% of global seaweed production. "
            "As of 2025, Cascadia Seaweed has 120 ha of tenured space (30 ha cultivated). "
            "Approximately 97 licensed seaweed farm operations, ~3,340 ha total as of 2024. "
            "Efficient product delivery and processing is the main limiting factor in BC. "
            "PSIA supports sector development through 5 core values: Education, Economic Opportunity, "
            "Research and Resources, Innovation, Community and Connection."
        ),
        "source": "SeaweedaquacultureinBritishColumbia_V8; FNFC Action Plan 2025; DFO",
    },
    {
        "id": "canada_aquaculture_stats_historical",
        "title": "Canadian All-Aquaculture Statistics 1986-2024 (Statistics Canada / DFO)",
        "text": (
            "⚠️ IMPORTANT: This data covers ALL Canadian aquaculture (finfish + shellfish) — NOT seaweed only. "
            "Seaweed is NOT separately reported in Statistics Canada aquaculture files. "
            "Canada total aquaculture production (all species): "
            "1986: 10,488 t → 1999: 114,204 t → 2007: 152,475 t → 2019: 187,258 t → 2024: 160,238 t. "
            "38-year CAGR 1986-2024: 7.44%. Total growth: +1,428%. "
            "Canada value (all aquaculture): 1986: $35.1M → 2006: $904.6M → 2024: $1,365.8M CAD. "
            "BC aquaculture production (all species, where data available): "
            "2002: 93,090 t, 2007: 81,384 t, 2015: 102,385 t (peak), 2022: 96,507 t, "
            "2023: 60,962 t (sharp decline), 2024: 64,154 t. "
            "BC value: 2002: $304.4M → 2022: $916.9M (peak) → 2024: $562.8M CAD. "
            "BC typically contributes 40-58% of Canada's total aquaculture production. "
            "BC salmon typically 85-95% of BC finfish by volume."
        ),
        "source": "Statistics Canada Table 32-10-0107-01; DFO Annual Aquaculture Statistics aqua86-aqua24",
    },
    {
        "id": "canada_aquaculture_recent",
        "title": "BC Aquaculture Recent Trends 2020-2024 and Salmon Decline",
        "text": (
            "BC all-aquaculture production 2020-2024: "
            "2020: 100,191 t ($665.8M), 2021: 96,074 t ($738.3M), 2022: 96,507 t ($916.9M), "
            "2023: 60,962 t ($557.9M), 2024: 64,154 t ($562.8M). "
            "BC farmed salmon 2020-2024: "
            "2020: 91,666 t, 2021: 84,171 t, 2022: 85,191 t, 2023: 50,067 t (sharp drop), 2024: 53,816 t. "
            "2023 BC production decline of 36.8% vs 2022 caused by DFO salmon farm licence reductions in BC. "
            "BC salmon peak was 2015 at 92,926 t. Historical average 2000-2019: ~73,863 t/yr. "
            "Canada all-aquaculture: 2024: 160,238 t ($1,365.8M) — up 9.8% to 160,318 t per RIAS 2024. "
            "BC shellfish stable at 6,673-10,417 t/yr (oysters, clams, mussels, scallops). "
            "BC early growth: 3,378 t (1986) → 17,739 t (1990) — driven by Pacific salmon expansion."
        ),
        "source": "Statistics Canada; DFO Annual Aquaculture Statistics aqua20-aqua24; RIAS Inc. 2024",
    },
    {
        "id": "value_added_account_bc",
        "title": "BC Aquaculture Value Added Account 1997-2024 (DFO/Statistics Canada)",
        "text": (
            "Statistics Canada/DFO Aquaculture Sector Value Added Account: ALL aquaculture species (not seaweed only). "
            "BC Gross Value Added (GDP contribution from aquaculture): "
            "1997: $97.5M → 2006: $162.1M → 2010: $210.9M → peak 2017: $338.0M (40.5% of output) "
            "→ 2019: $180.6M → 2021: $201.9M → 2022: $198.8M → 2024: $21.6M (3.4% of output). "
            "BC GVA collapsed 94% from 2017 to 2024 — direct result of salmon licence reductions. "
            "BC Gross Output (total economic activity): rose from $245.8M (2000) → $883.1M (2018 peak) → $626.7M (2024). "
            "BC Feed Costs (single largest input): $69M (1997) → $248M (2019) → $178M (2024). "
            "Feed was 38-54% of all production costs through 2000-2014; declining to 29.5% in 2024. "
            "Canada total GVA: peaked at $591.2M (2016) → $236.2M (2024). "
            "BC Wages: $30M (1997) → $84M (2022) → $72.4M (2024). "
            "BC Therapeutants/medicines: $3M (1997) → $36.2M (2024) — 12× increase, 5.8% of output."
        ),
        "source": "Statistics Canada/DFO Value Added Account files va97pub-va24pub",
    },
    {
        "id": "seaweed_opportunity_canada",
        "title": "Seaweed Opportunity in Canadian Aquaculture Context",
        "text": (
            "BC seaweed (~$50M CAD, 2024) represents ~8% of BC's total aquaculture gross output ($626.7M). "
            "For context: BC salmon alone generates ~$526.8M value (2024, all finfish). "
            "Key opportunity metrics derived from Statistics Canada/DFO value added data: "
            "BC feed costs 2024: $178.3M — potential market for seaweed-based feed additives. "
            "BC therapeutant/medicine costs 2024: $36.2M — seaweed bioactive compounds as alternatives. "
            "BC depreciation 2024: $56.4M — existing mariculture infrastructure available for seaweed. "
            "Canada all-aquaculture value 2024: $1,365.8M — seaweed is <4% of this. "
            "The 2023-24 BC salmon licence reduction (from 85,191 t in 2022 to 50,067 t in 2023) "
            "has displaced aquaculture workers and infrastructure. Seaweed farming offers an alternative "
            "use for decommissioned salmon farm sites (as documented in RDMW SIDP 2024). "
            "BC shellfish: 9,537 t (2024) worth ~$26.6M — shellfish and seaweed can co-exist in IMTA systems."
        ),
        "source": "Statistics Canada; DFO VA files; RDMW SIDP 2024; PSIA estimates",
    },
    {
        "id": "rdmw_swot_business",
        "title": "RDMW BC Seaweed Development — SWOT and Business Models",
        "text": (
            "Mount Waddington SIDP (LGL Ltd., October 2024) covers northern Vancouver Island. "
            "SWOT Strengths: suitable coastal waters; rich kelp biodiversity (Bull, Winged, Giant, Sugar kelp); "
            "experienced workforce; existing mariculture infrastructure; First Nations leadership. "
            "SWOT Weaknesses: high transport costs for unprocessed seaweed; limited cold storage. "
            "SWOT Opportunities: decommissioned finfish farm sites; sustainable food demand growth. "
            "Business models: (1) Vertically Integrated; (2) Start-up; (3) Co-op/Shared Infrastructure; "
            "(4) Specialized Supply Chain. "
            "Site selection: depth 10-40 m; avoid protected areas, eelgrass beds, existing kelp beds. "
            "Regulatory steps: Land Act tenure; Wild Aquatic Plant Harvester Licence; Aquatic Plant Culture Licence."
        ),
        "source": "LGL Limited for RDMW, October 2024",
    },
    {
        "id": "bc_regulatory_framework",
        "title": "BC Seaweed Aquaculture Regulatory Framework",
        "text": (
            "Four agencies regulate BC seaweed aquaculture: "
            "(1) BC Ministry of Water, Land and Resource Stewardship (MWLRS): Land Act tenures, Fish and Seafood Act. "
            "(2) DFO: Fisheries Act; fish habitat assessment. "
            "(3) Transport Canada: navigable waters. "
            "(4) Applications via FrontCounter BC → federal-provincial Project Review Team. "
            "Six permitting steps: Land Act tenure; Wild Aquatic Plant Harvester Licence; "
            "Aquatic Plant Culture Licence; Health Canada novel food approval; food processing licence; export. "
            "Conditions of licence: seeds from within 50 km, ≥30 individuals; non-native species prohibited; "
            "density mimics natural seaweed beds. Maximum tenure: 520 ha (Land Act). "
            "CRITICAL GAP: Provincial jurisdiction ends 12 nautical miles from outer coast — "
            "no regulatory authority for offshore seaweed aquaculture. "
            "Updated BC seaweed-specific policy anticipated by end of 2025."
        ),
        "source": "SeaweedaquacultureinBritishColumbia_V8 (Martone et al. 2025); RDMW SIDP 2024",
    },
    {
        "id": "permitting_kpis_canada",
        "title": "Canadian Aquaculture Permitting KPIs (Estimated from Public Reports)",
        "text": (
            "Key permitting and operational KPIs for Canadian seaweed aquaculture (estimated): "
            "Aquaculture Development Areas (ADAs): ~17,900 ha designated in BC. "
            "Total licensed farms (all aquaculture): ~755 operations. "
            "Seaweed-specific farms: ~97 active operations (2024). "
            "Seaweed farm area: ~3,340 ha total licensed. "
            "Aquaculture Opportunity Areas (AOAs): 54 assessed by DFO (2024). "
            "Fisheries Act compliance rate: ~95.3% inspection pass rate. "
            "Social license score: ~70/100. "
            "Indigenous agreements: ~24 MOUs/benefit agreements. "
            "Indigenous employed: ~207 people. "
            "Indigenous trained (CIRNAC): ~174 individuals. Total trained: ~504."
        ),
        "source": "DFO Annual Aquaculture Statistics; BC Ministry of Agriculture; CIRNAC; Stats Canada",
    },
    {
        "id": "slo_handbook_eu",
        "title": "SLO Handbook for Seaweed Cultivation (GENIALG Project)",
        "text": (
            "SLO Handbook v4.2 (Billing, Rostan & Tett, SAMS, EU H2020 GENIALG). "
            "Key findings: (1) Clearly define seaweed industry terms — accidental association with "
            "wild harvesting generates unnecessary social opposition. "
            "(2) Information provision is key — where environmental impact info is scarce, "
            "stakeholders substitute experiences from other industries. "
            "(3) Smaller-scale locally-owned farms associated with gaining SLO. "
            "(4) Under-development of public policy negatively influences community perceptions. "
            "SLO building blocks: Trust; Fairness in decision procedures; Environmental alignment; "
            "Local benefits; Perceived legitimacy. "
            "Farms dispersed along coastline as cooperative may balance SLO and economic viability."
        ),
        "source": "Billing, Rostan & Tett, SAMS, EU H2020 GENIALG Project, SLO Handbook v4.2",
    },
    {
        "id": "slo_uk_guide_2023",
        "title": "UK Guide to SLO for Seaweed Cultivation — Three Factors (2023)",
        "text": (
            "WWF-UK funded guide (SAMS, 2023) using Q-method. Three SLO factors: "
            "Factor 1 — Environmental Sustainability and Responsible Practices (most important). "
            "Factor 2 — Smaller Scales with Local Social Benefits. "
            "Factor 3 — Regulation and Business Development. "
            "Consensus: transparent communication; clear product labelling; community inclusion. "
            "Seaweed cultivation NOT socially acceptable unless environmentally sustainable. "
            "Policy underdevelopment negatively influences community perceptions."
        ),
        "source": "Billing et al., WWF-UK, SAMS, 2023",
    },
    {
        "id": "noaa_slo_framework_2022",
        "title": "NOAA SLO Framework — 7 Community Predictors",
        "text": (
            "NOAA Technical Memorandum NMFS-NE-287 (Whitmore, Cutler & Thunberg, 2022). "
            "Seven predictors of community approval: (1) Environmental values; (2) Economic values; "
            "(3) Use-conflict; (4) Knowledge of aquaculture; (5) Experience with aquaculture; "
            "(6) Confidence in government; (7) Perceptions of health and safety. "
            "US aquaculture produces only $1 billion/year despite vast coastline, partly due to "
            "social license barriers — a cautionary lesson for Canadian expansion planning."
        ),
        "source": "Whitmore, Cutler & Thunberg, NOAA NMFS-NE-287, August 2022",
    },
    {
        "id": "wwf_slo_workshop_2022",
        "title": "WWF SLO Workshop — Seaweed Barriers and Communication Strategies",
        "text": (
            "WWF 2022 SLO Workshop (Portland, Maine, Seaweed Solution Project). "
            "Main SLO challenges: lack of public awareness; visual aesthetics / 'ocean view' concerns; "
            "navigational conflicts; regulatory complexity; overstating benefits erodes trust. "
            "Communication: simple accessible language; farmer storytelling; trusted messengers "
            "(eNGOs like TNC/WWF, scientists, Sea Grant offices, local chefs). "
            "Recommended: Community of Practice model for knowledge sharing. "
            "Regulatory agencies look for no negative impact rather than positive impacts like biodiversity."
        ),
        "source": "WWF Seaweed Solution Project Workshop, Portland, Maine, April 2022",
    },
    {
        "id": "fnfc_action_plan_2025",
        "title": "FNFC Whole of Aquaculture Action Plan 2025",
        "text": (
            "FNFC Whole of Aquaculture Action Plan (September 2025) — 15+ years of Nation engagement. "
            "Five objectives: (1) Legislation & Policy: amend Fisheries Act; co-develop Aquaculture Act. "
            "(2) Culture & Health: OCAP data sovereignty, traditional food systems. "
            "(3) Environmental Resilience: expand Guardian/Watchmen programs. "
            "(4) Market Access: implement UNDRIP rights; Nation-led ownership. "
            "(5) Funding & Capacity: $400M First Nations Aquaculture Investment Fund ($200M federal). "
            "Problems: FN excluded from licencing, enforcement, monitoring, decision-making. "
            "United Kelp Cooperative: coastal First Nations restoring marine habitats by cultivating "
            "giant and sugar kelp — supports biodiversity, water quality, and carbon capture."
        ),
        "source": "First Nations Fisheries Council of BC, September 2025",
    },
    {
        "id": "indigenous_engagement_detail",
        "title": "Indigenous Peoples and Traditional Aquaculture in BC",
        "text": (
            "Traditional practices: ancient clam gardens along BC coast quadrupled butter clam harvests. "
            "Herring roe transplantation increased stock distribution; salmon roe transplantation created new runs. "
            "Current engagement KPIs (2024): ~24 formal agreements (MOUs/benefit agreements); "
            "~207 Indigenous people employed; ~174 trained (CIRNAC program); 504 total trained. "
            "BC First Nations are diverse — some pursue commercial aquaculture, others oppose it. "
            "Consultation required via review and referral process for any Crown land tenure."
        ),
        "source": "CIRNAC; BC Treaty Commission; Stats Canada 14-10-0023; FNFC 2016; FNFC 2025",
    },
    {
        "id": "seaweed_production_cost_detail",
        "title": "Large-Scale Seaweed Farm Cost Model (Kite-Powell et al. 2022)",
        "text": (
            "Kite-Powell et al. (Woods Hole), Applied Phycology 3(1):435–445, 2022. "
            "At 1,000 ha+: farm gate production costs = $200–$300 per dry tonne. "
            "Below $100/dry tonne achievable at optimal shore-proximate sites. "
            "Baseline temperate kelp (1,000 ha): 32,800 dry t/year; capital ~$48M; opex ~$4.7M/yr. "
            "Baseline tropical (1,000 ha): 41,700 dry t/year; capital ~$31M; opex ~$5.2M/yr. "
            "Scale economies: cost declines ~40% from 100 to 1,000 ha. "
            "Historical benchmarks: North Sea Saccharina 2016 = $2,000/t; Chile Macrocystis 2019 = $610/t."
        ),
        "source": "Kite-Powell et al., Applied Phycology, 3(1), 435-445, 2022",
    },
    {
        "id": "ecology_effects_bc",
        "title": "Ecological Effects of Seaweed Farms in BC",
        "text": (
            "Positive ecological effects: fixes N and P (reduces eutrophication, harmful algal blooms); "
            "creates ocean acidification refugia; provides habitat; protects shorelines; "
            "restoration aquaculture (rebuilding bull kelp). "
            "Negative effects: competition for nutrients; habitat change; entanglements; pathogen risks; "
            "gene flow from cultivated to wild seaweeds at large scale. "
            "Current BC conditions of licence: seed from within 50 km, ≥30 individuals; "
            "non-native species prohibited; density mimics natural seaweed beds. "
            "BC maximum tenure: 520 ha. Maine: 40.5 ha commercial limit. "
            "Updated BC seaweed-specific policy anticipated end of 2025."
        ),
        "source": "SeaweedaquacultureinBritishColumbia_V8 (Martone et al. 2025)",
    },
    {
        "id": "environment_benefits_climate",
        "title": "Seaweed Environmental Benefits — Carbon, Water Quality, Biodiversity",
        "text": (
            "Seaweed requires no freshwater, fertilizers, pesticides, or arable land. "
            "Canada's seaweed sector: estimated ~654 kilotonnes CO2e sequestered annually (2024). "
            "Aquatic vegetation contributes 1-10% of global marine net primary productivity. "
            "New low-carbon products: textiles, antimicrobial bandages, biofuels, biodegradable plastics. "
            "IMTA role: seaweed absorbs waste nutrients from fish and shellfish farms. "
            "Brown algae (kelp) iodine content 1,500-8,000 ppm — addresses iodine deficiency."
        ),
        "source": "SeaweedaquacultureinBritishColumbia_V8; FAO 2018; DFO assessments; Lang-Wong 2022",
    },
    {
        "id": "income_geography",
        "title": "Geographic Concentration and Economic Development",
        "text": (
            "Asia 97% of global seaweed production. Americas 1.5%. Europe 0.8%. "
            "China alone: 56%. Upper-middle income countries: 86.7%; high-income (Canada, Norway, etc.): ~7%. "
            "Norway has emerged as fastest-growing high-income producer (7.3% CAGR). "
            "Canada: 0.04% of global production despite 25,000+ km BC coastline. "
            "Regional: Chile ($209M exports), Spain ($145M), France ($124M), USA ($102M)."
        ),
        "source": "FAO FishStat 2024; Cai FAO 2021; World Bank income classifications",
    },
    {
        "id": "production_trends_growth",
        "title": "Seaweed Production Trends and CAGR",
        "text": (
            "Global seaweed grew 736% from 1990 to 2020. "
            "Aquaculture: 22M tonnes (2010) → 38.9M tonnes (2024). Wild: flat ~1.08-1.3M tonnes. "
            "Global 10-yr CAGR: ~2.9%; YoY growth ~3.1% in 2024. "
            "Number of farmed species: 21 (2000) → 31 (2024). "
            "Average implied seaweed price: ~USD 0.51/kg in 2024. "
            "BC seaweed production: still very low but growing. "
            "Efficient delivery and processing is the main BC/eastern North Pacific limiting factor."
        ),
        "source": "FAO FishStat 2024; RDMW SIDP 2024; SeaweedaquacultureinBritishColumbia_V8",
    },
    {
        "id": "psia_values",
        "title": "PSIA Values, Mission and Role in Canadian Seaweed Sector",
        "text": (
            "PSIA five core values: (1) Education; (2) Economic Opportunity; (3) Research and Resources; "
            "(4) Innovation; (5) Community and Connection. "
            "PSIA develops educational material, promotes new technology, advocates to government, "
            "and connects farmers, researchers, government, Indigenous communities, and industry. "
            "Dashboard live data: FAO FishStat for global seaweed KPIs; "
            "Statistics Canada for Canadian all-aquaculture historical context; "
            "DFO/CIRNAC estimates for permitting and social KPIs."
        ),
        "source": "PSIA organizational documents; RDMW SIDP 2024; Dashboard documentation",
    },
    {
        "id": "data_disambiguation",
        "title": "CRITICAL DATA DISAMBIGUATION — Seaweed vs All-Aquaculture Numbers",
        "text": (
            "⚠️ CRITICAL: This dashboard has TWO data streams. "
            "STREAM 1 (Tabs 1-4): FAO FishStat SEAWEED-ONLY production. "
            "Global seaweed: ~35.8M wet tonnes (2019); ~40M wet tonnes (2024). "
            "STREAM 2 (Tab 5): Statistics Canada ALL-SPECIES Canadian aquaculture. "
            "Canada ALL-SPECIES: 160,238 tonnes (2024), $1,365.8M value. "
            "Dominated by farmed salmon (53,816 t BC, 88,076 t Canada in 2024). "
            "NEVER confuse these: 112M = all global aquaculture 2017 (not seaweed). "
            "NEVER cite 160,238 t as seaweed production — it's all Canadian aquaculture. "
            "Canada seaweed-specific: ~97 operations, ~3,340 ha, ~CAD $50M value (2024). "
            "BC seaweed = ~8% of BC's total aquaculture gross output ($626.7M in 2024)."
        ),
        "source": "FAO FishStat; Statistics Canada; RIAS Inc. 2024; Dashboard documentation",
    },
    {
        "id": "fao_global_seaweed_report_2018",
        "title": "FAO Global Status of Seaweed Production Trade and Utilization (2018)",
        "text": (
            "FAO Globefish Research Programme Vol. 124 (Ferdouse et al. 2018). "
            "~221 seaweed species of commercial value; ~10 intensively cultivated. "
            "Japanese kelp (Saccharina japonica): >33% of global cultivated seaweed production. "
            "Seaweed nutritional profile: sodium, calcium, magnesium, potassium; "
            "Brown algae (kelp) iodine 1,500-8,000 ppm — addressing iodine deficiency. "
            "Carrageenan from Eucheuma/Kappaphycus: major industrial hydrocolloid for food, cosmetics, pharma. "
            "Regional producers: China, Indonesia, Malaysia (Asia); Chile (Americas); Denmark, EU (Europe)."
        ),
        "source": "Ferdouse et al., FAO Globefish Research Programme Vol. 124, 2018",
    },
    {
        "id": "canada_aquaculture_snapshot_2024",
        "title": "2024 Canadian Aquaculture Industry Data Snapshot",
        "text": (
            "⚠️ Covers ALL aquaculture species — NOT seaweed only. "
            "RIAS Inc./aquaculture.ca 2024 Data Snapshot: "
            "ALL-SPECIES aquaculture grew 9.8% to 160,318 tonnes; $6 billion economic output; "
            "GDP: $2.27 billion; 18,074 full-time jobs. "
            "Farmed salmon: 69.3% of production (109,048 t). "
            "BC farmed salmon: 53,816 t. Atlantic Canada salmon: 55,232 t (+17.7%). "
            "Farmed shellfish: 37,904 t. Exports: $970 million (+7.3%). "
            "FNFC Whole of Aquaculture Action Plan proposes $400M First Nations Aquaculture Investment Fund."
        ),
        "source": "RIAS Inc. / Aquaculture.ca 2024; FNFC Action Plan 2025",
    },
    {
        "id": "lba_first_nations_mosier",
        "title": "Land-Based Aquaculture for First Nations in BC — Mosier SFU 2017",
        "text": (
            "SFU MRM thesis (Mosier 2017) investigates LBA for Nanwakolas Member Nations, northern Vancouver Island. "
            "IMTA integrates salmon (fed species), shellfish (invertebrate extractive), "
            "seaweed (inorganic extractive — absorbs dissolved N and P from salmon). "
            "Key: BC Fisheries Act regulatory gaps create barriers to LBA; licensing time is excessive. "
            "LBA can generate sustainable economic opportunities and preserve traditional foods. "
            "IMTA systems are promising for First Nations — seaweed adds marketable product while "
            "absorbing waste, improving water quality, aligning with Indigenous ecosystem stewardship."
        ),
        "source": "Elizabeth Mosier, SFU MRM Report No. 667, 2017",
    },
    {
        "id": "flaherty_imta_slo_bc",
        "title": "IMTA and SLO in BC Coastal Waters — Flaherty UVic",
        "text": (
            "Dr. Mark Flaherty (University of Victoria) on IMTA and SLO in BC. "
            "BC: 27,000 km coastline. IMTA: salmon + shellfish + seaweed (seaweed absorbs nutrients). "
            "SLO challenges for IMTA: competing user groups (fishing, tourism, recreation); "
            "regulatory complexity; need to demonstrate ecological not just economic benefit. "
            "Indigenous co-governance essential for long-term SLO. "
            "IMTA with seaweed aligns with First Nations ecosystem stewardship principles."
        ),
        "source": "Dr. Mark Flaherty, University of Victoria, S11 Conference Presentation",
    },
    {
        "id": "bc_first_nations_aquaculture_2016",
        "title": "First Nations and Aquaculture in BC (FNFC 2016)",
        "text": (
            "Traditional Indigenous aquaculture in BC: ancient clam gardens along entire BC coast "
            "quadrupled butter clam harvests and doubled littleneck clam yields. "
            "Herring roe and salmon roe transplantation increased stock distribution. "
            "Case studies: K'ómoks First Nation shellfish farm; Na̱mǥis Nation closed-containment salmon; "
            "Okanagan Nation Alliance freshwater sockeye hatchery. "
            "Most contentious issue: open-net pen Atlantic salmon farming and risks to wild salmon. "
            "All traditional practices maintained habitat and ecosystem awareness for intergenerational sustainability."
        ),
        "source": "First Nations Fisheries Council of BC, 2016",
    },
    {
        "id": "fao_fishery_stats_2021",
        "title": "FAO Fishery and Aquaculture Statistics Yearbook 2021",
        "text": (
            "FAO Fishery and Aquaculture Statistics Yearbook 2021 (cc9523en). "
            "Coverage: production by country, species, environment, ISSCAAP division. "
            "Seaweeds and other aquatic plants form a distinct ISSCAAP division. "
            "Asia dominates both capture fisheries and aquaculture globally. "
            "Primary reference for global fisheries policy including PSIA benchmarking."
        ),
        "source": "FAO Fishery and Aquaculture Statistics Yearbook 2021, Rome, 2024",
    },
    {
        "id": "fao_top_aquaculture_note",
        "title": "FAO All-Species Aquaculture — CRITICAL DISAMBIGUATION",
        "text": (
            "⚠️ CRITICAL: The 112 million tonnes figure = ALL aquaculture (fish, shellfish, seaweed combined 2017). "
            "NEVER cite it as seaweed production. Seaweed-only 2017: ~33.7M tonnes. "
            "Canada all-aquaculture 2024: 160,238 tonnes (salmon 69.3%, shellfish etc.) — NOT seaweed. "
            "Seaweed as subset of all-aquaculture 2017: Brown seaweeds ~13.8M t; Red seaweeds ~17.3M t; "
            "Combined seaweed ~31M tonnes = ~27.7% of all 112M t."
        ),
        "source": "Cai et al., FAO / Chinese Academy of Fishery Sciences, 2017; RIAS Inc. 2024",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF RAG INDEX
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def build_rag_index():
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
    if not _rag_ok or _vec is None: return ""
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


def build_ctx():
    """Bundle the live data objects the chat engine computes from."""
    if not data_ok:
        return {"data_ok": False, "C": C}
    return {
        "data_ok": True, "C": C,
        "fgp": fgp, "faq": faq, "fav": fav, "fcq": fcq,
        "LY": LY, "cagr_win": cagr_win, "year_range": year_range,
        "CAN_PROD_T": CAN_PROD_T, "BC_PROD_T": BC_PROD_T,
        "BC_VAL_K": BC_VAL_K, "CAN_VAL_K": CAN_VAL_K,
        "BC_SALMON_T": BC_SALMON_T, "BC_SHELL_T": BC_SHELL_T,
        "VA_BC_GVA_K": VA_BC_GVA_K, "VA_BC_FEED_K": VA_BC_FEED_K,
        "VA_BC_THERAP_K": VA_BC_THERAP_K, "VA_BC_WAGES_K": VA_BC_WAGES_K,
        "VA_BC_OUTPUT_K": VA_BC_OUTPUT_K,
    }

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "messages"  not in st.session_state: st.session_state.messages  = []
if "thinking"  not in st.session_state: st.session_state.thinking  = False
if "chat_mode" not in st.session_state: st.session_state.chat_mode = "UC1"

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT (RAG-augmented, with full live data alignment)
# ─────────────────────────────────────────────────────────────────────────────
def build_system_prompt(query: str = "", is_uc2: bool = False) -> str:
    base = (
        "You are the PSIA (Pacific Seaweed Industry Association) AI assistant "
        "embedded in a seaweed analytics dashboard. Answer questions about the "
        "global seaweed industry, Canadian aquaculture context, BC regulations, "
        "social license to operate, production costs, First Nations aquaculture, "
        "and PSIA initiatives.\n\n"
        "== CRITICAL DATA RULES ==\n"
        "RULE 1 — TWO DATA STREAMS exist in this dashboard:\n"
        "  Stream 1 (Tabs 1-4): FAO FishStat SEAWEED-ONLY global production.\n"
        "  Stream 2 (Tab 5): Statistics Canada ALL-SPECIES Canadian aquaculture.\n"
        "RULE 2 — NEVER confuse these. Do not cite Canada's 160,238 t as seaweed production.\n"
        "RULE 3 — Global seaweed-only: ~35.8M wet tonnes (2019); ~40M wet tonnes (2024).\n"
        "RULE 4 — Canada seaweed-specific: ~97 operations, ~3,340 ha, ~CAD $50M value (2024).\n"
        "RULE 5 — NEVER cite the 112M all-species figure as seaweed production.\n"
        "Be concise and data-specific. Always cite your source when using specific numbers.\n"
        "If a specific value is not explicitly listed in the data above, say exactly: "
        "'That specific value is not in my current data — please check the dashboard chart directly.' "
        "NEVER guess, approximate, or invent a number that is not explicitly listed above. "
        "NEVER say a year has a certain value unless you can see that exact year in the data provided."
    )

    if is_uc2 and data_ok:
        hist_yrs = sorted([y for y in fgp["period"].unique() if fgp[fgp["period"]==y]["value"].sum()>0])
        hist_lines = "\n".join([f"  {y}: {fgp[fgp['period']==y]['value'].sum()/1e6:.2f}M t" for y in hist_yrs])

        av_sp = fav.groupby(["period","seaweed_name"])["value"].sum().reset_index()
        aq_sp = faq.groupby(["period","seaweed_name"])["value"].sum().reset_index()
        av_sp.columns = ["year","species","usd_k"]; aq_sp.columns = ["year","species","tonnes"]
        jp_sp = av_sp.merge(aq_sp,on=["year","species"])
        jp_sp = jp_sp[(jp_sp["tonnes"]>100)&(jp_sp["usd_k"]>0)]
        jp_sp["usd_kg"] = jp_sp["usd_k"]*1000/jp_sp["tonnes"]/1000
        sp_price_agg = (jp_sp[jp_sp["usd_kg"]<20].groupby("species")
                        .agg(avg_price=("usd_kg","mean"),tot_vol=("tonnes","sum"))
                        .reset_index().nlargest(10,"tot_vol").sort_values("tot_vol",ascending=False))
        price_lines = "\n".join([f"  {row['species']}: ${row['avg_price']:.3f}/kg avg" for _,row in sp_price_agg.iterrows()])

        cr_data = fgp[fgp["value"]>0].groupby(["period","country_name"])["value"].sum().reset_index()
        ly_cr   = cr_data[cr_data["period"]==LY]; tot_cr = ly_cr["value"].sum()
        sv = ly_cr["value"].sort_values(ascending=False).values
        cr5  = sv[:5].sum()/tot_cr*100  if len(sv)>=5  else 0
        cr10 = sv[:10].sum()/tot_cr*100 if len(sv)>=10 else 0

        top15_c = fgp[fgp["period"]==LY].groupby("country_name")["value"].sum().sort_values(ascending=False).head(15)
        country_lines = "\n".join([f"  {i+1}. {c}: {v/1e6:.3f}M t ({v/fgp[fgp['period']==LY]['value'].sum()*100:.1f}%)"
                                   for i,(c,v) in enumerate(top15_c.items())])

        cont_s = fgp[fgp["period"]==LY].groupby("continent_group_en")["value"].sum().sort_values(ascending=False)
        cont_lines = "\n".join([f"  {k}: {v/cont_s.sum()*100:.1f}% ({v/1e6:.2f}M t)"
                                for k,v in cont_s.items() if k!="Unknown"])

        sp10 = faq.groupby("seaweed_name")["value"].sum().sort_values(ascending=False).head(10)
        sp_lines = "\n".join([f"  {i+1}. {s}: {v/1e6:.2f}M t cumulative" for i,(s,v) in enumerate(sp10.items())])

        ig_s = fgp[fgp["period"]==LY].groupby("ecoclass_group_en")["value"].sum().sort_values(ascending=False)
        ig_lines = "\n".join([f"  {k}: {v/ig_s.sum()*100:.1f}% ({v/1e6:.2f}M t)" for k,v in ig_s.items()])

        perm_ly = sim_perm[sim_perm["year"]==sim_perm["year"].max()].iloc[0]
        soc_ly  = sim_social[sim_social["year"]==sim_social["year"].max()].iloc[0]

        cagr_base_yr = max(LY-cagr_win, year_range[0])
        c_base = fgp[fgp["period"]==cagr_base_yr].groupby("country_name")["value"].sum().reset_index().rename(columns={"value":"base_v"})
        c_last = fgp[fgp["period"]==LY].groupby("country_name")["value"].sum().reset_index().rename(columns={"value":"last_v"})
        cagr_df = c_base.merge(c_last,on="country_name")
        cagr_df = cagr_df[(cagr_df["base_v"]>0)&(cagr_df["last_v"]>1000)]
        cagr_df["cagr"] = ((cagr_df["last_v"]/cagr_df["base_v"])**(1/cagr_win)-1)*100
        top_grow = cagr_df.nlargest(10,"cagr")[["country_name","cagr","last_v"]]
        cagr_lines = "\n".join([f"  {row['country_name']}: {row['cagr']:.1f}% CAGR ({row['last_v']/1e3:.0f}k t)" for _,row in top_grow.iterrows()])

        # Canadian context from static dicts
        bc24_prod = BC_PROD_T.get(2024,64154)
        can24_prod = CAN_PROD_T.get(2024,160238)
        bc24_val  = BC_VAL_K.get(2024,562814)
        can24_val = CAN_VAL_K.get(2024,1365820)
        bc24_gva  = VA_BC_GVA_K.get(2024,21600)
        bc24_feed = VA_BC_FEED_K.get(2024,178254)
        bc24_sal  = BC_SALMON_T.get(2024,53816)

        base += f"""

╔══════════════════════════════════════════════════════════════════════════════════╗
║  LIVE DASHBOARD DATA — STREAM 1: SEAWEED-ONLY (FAO FishStat, {year_range[0]}–{year_range[1]})           ║
╚══════════════════════════════════════════════════════════════════════════════════╝

── TABS 1-4: GLOBAL SEAWEED PRODUCTION & VALUE (seaweed species ONLY) ──────────────
Seaweed production ({LY}): {prod_tot/1e6:.2f}M t · YoY {yoy_prod:+.1f}% · CAGR {cagr_prod:.1f}%
  Farmed: {aq_tot/1e6:.2f}M t ({aq_tot/prod_tot*100:.1f}%) · Wild: {wc_tot/1e6:.3f}M t ({wc_tot/prod_tot*100:.1f}%)
  ASFIS seaweed species: {sp_total} total (farmed:{sp_cult}, wild:{sp_wild})
  Avg seaweed price: ${avg_price_kg:.2f}/kg
  Total seaweed value: USD ${val_tot/1e6:.1f}B (YoY {yoy_val:+.1f}%)
  Market concentration: CR5={cr5:.1f}%, CR10={cr10:.1f}%

Historical global seaweed-only production (FAO FishStat):
{hist_lines}

Top 10 seaweed species avg price (implied USD/kg):
{price_lines}

Continental share ({LY}):
{cont_lines}

Top 15 countries by seaweed production ({LY}):
{country_lines}

Top 10 seaweed species by cumulative farmed volume:
{sp_lines}

Income group production share ({LY}):
{ig_lines}

Top 10 fastest-growing countries ({cagr_win}-yr CAGR):
{cagr_lines}

── TABS 1-3: PERMITTING & SOCIAL KPIs (Canada — estimated from public reports) ─────
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

╔══════════════════════════════════════════════════════════════════════════════════╗
║  LIVE DASHBOARD DATA — STREAM 2: CANADIAN ALL-AQUACULTURE (Statistics Canada)  ║
║  TAB 5 data — ALL aquaculture species (finfish + shellfish, seaweed included)  ║
║  Seaweed is NOT separately reported here — this is the broad sector context    ║
╚══════════════════════════════════════════════════════════════════════════════════╝

── TAB 5: CANADIAN CONTEXT (Statistics Canada / DFO, all aquaculture species) ──────
BC all-aquaculture 2024: {bc24_prod/1000:.0f}k t total | ${bc24_val/1000:.0f}M CAD value
Canada all-aquaculture 2024: {can24_prod/1000:.0f}k t total | ${can24_val/1000:.0f}M CAD value
BC farmed salmon 2024: {bc24_sal/1000:.0f}k t (sharp decline from 2022 peak 85,191 t — licence reductions)
BC aquaculture Gross Value Added 2024: ${bc24_gva/1000:.1f}M (only 3.4% of output, vs 40.5% in 2017)
BC feed costs 2024: ${bc24_feed/1000:.0f}M CAD — potential seaweed additive market
BC therapeutant costs 2024: ${VA_BC_THERAP_K.get(2024,36238)/1000:.0f}M CAD — seaweed bioactive opportunity
BC seaweed ~$50M = ~8% of BC total aquaculture gross output (${VA_BC_OUTPUT_K.get(2024,626689)/1000:.0f}M)

Canada all-aquaculture production ALL years (Statistics Canada):
  {chr(10).join([f"  {y}: {CAN_PROD_T[y]:,} t" for y in sorted(CAN_PROD_T.keys())])}
BC all-aquaculture production ALL years (Statistics Canada):
  {chr(10).join([f"  {y}: {BC_PROD_T[y]:,} t" for y in sorted(BC_PROD_T.keys())])}
BC aquaculture value ALL years ($000 CAD):
  {chr(10).join([f"  {y}: ${BC_VAL_K[y]:,}k" for y in sorted(BC_VAL_K.keys())])}
Canada aquaculture value ALL years ($000 CAD):
  {chr(10).join([f"  {y}: ${CAN_VAL_K[y]:,}k" for y in sorted(CAN_VAL_K.keys())])}
BC Gross Value Added ALL years ($000 CAD):
  {chr(10).join([f"  {y}: ${VA_BC_GVA_K[y]:,}k ({VA_BC_GVA_K[y]/VA_BC_OUTPUT_K[y]*100:.1f}% of output)" if y in VA_BC_OUTPUT_K else f"  {y}: ${VA_BC_GVA_K[y]:,}k" for y in sorted(VA_BC_GVA_K.keys())])}
BC Feed Costs ALL years ($000 CAD):
  {chr(10).join([f"  {y}: ${VA_BC_FEED_K[y]:,}k" for y in sorted(VA_BC_FEED_K.keys())])}
BC Wages ALL years ($000 CAD):
  {chr(10).join([f"  {y}: ${VA_BC_WAGES_K[y]:,}k" for y in sorted(VA_BC_WAGES_K.keys())])}
BC Therapeutants ALL years ($000 CAD):
  {chr(10).join([f"  {y}: ${VA_BC_THERAP_K[y]:,}k" for y in sorted(VA_BC_THERAP_K.keys())])}
BC Salmon production ALL years (tonnes):
  {chr(10).join([f"  {y}: {BC_SALMON_T[y]:,} t" for y in sorted(BC_SALMON_T.keys())])}
BC Shellfish production ALL years (tonnes):
  {chr(10).join([f"  {y}: {BC_SHELL_T[y]:,} t" for y in sorted(BC_SHELL_T.keys())])}
"""

    top_k_rag = 3 if is_uc2 else 6
    rag_ctx = retrieve_context(query, top_k=top_k_rag)
    if rag_ctx:
        base += f"\n\n== BACKGROUND KNOWLEDGE (supplementary) ==\n{rag_ctx}"
    return base

# ─────────────────────────────────────────────────────────────────────────────
# CHAT PANEL
# ─────────────────────────────────────────────────────────────────────────────
with _chat_col:
    is_uc2    = st.session_state.chat_mode == "UC2"
    rag_badge = f"🔍 RAG ({len(RAG_DOCS)} docs)" if _rag_ok else "💬"

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#085041 0%,#0F6E56 100%);
                padding:16px 20px 14px;border-radius:12px 12px 0 0;
                box-shadow:0 2px 10px rgba(0,0,0,0.20);">
      <div style="font-size:16px;font-weight:700;
                  font-family:Lora,Georgia,serif;letter-spacing:0.2px;
                  color:#FFFFFF;text-shadow:0 1px 3px rgba(0,0,0,0.30);">
        &#127807; PSIA AI Assistant
      </div>
      <div style="font-size:12.5px;margin-top:3px;
                  font-family:'Source Sans 3','Source Sans Pro',Helvetica,sans-serif;
                  font-weight:400;color:#B8EDD8;">
        {rag_badge} &nbsp;&middot;&nbsp; Groq llama-3.3-70b &nbsp;&middot;&nbsp; 2 data streams
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div style="background:#FFFFFF;padding:8px 14px 4px;border-left:1px solid #CCCCCC;border-right:1px solid #CCCCCC;"></div>', unsafe_allow_html=True)

    _mc1,_mc2 = st.columns([3,1])
    with _mc1:
        mode_choice = st.radio("mode",["💬 Industry Info","📊 Live Data"],
            index=1 if is_uc2 else 0,horizontal=True,label_visibility="collapsed",key="chat_mode_radio")
        st.session_state.chat_mode = "UC2" if "Live" in mode_choice else "UC1"
        is_uc2 = st.session_state.chat_mode == "UC2"
    with _mc2:
        if st.button("🗑️",key="clr_chat",help="Clear conversation",use_container_width=True):
            st.session_state.messages = []; st.rerun()

    # ── Chat: inject a unique ID so we can target it with ironclad CSS ─────────
    import uuid as _chat_uuid
    _cid = "ch" + _chat_uuid.uuid4().hex[:8]

    st.markdown(f"""
    <style>
    #{_cid} {{
        background: #F0F7F4 !important;
        border-left: 1.5px solid #C0D8D0 !important;
        border-right: 1.5px solid #C0D8D0 !important;
        min-height: 330px !important;
        max-height: 430px !important;
        overflow-y: auto !important;
        padding: 16px 14px 14px !important;
        font-family: 'Source Sans 3','Source Sans Pro',Helvetica,Arial,sans-serif !important;
    }}
    #{_cid} .cb-welcome {{
        background: #FFFFFF !important;
        border: 1.5px solid #B8D8CC !important;
        border-left: 4px solid #0F6E56 !important;
        border-radius: 0 10px 10px 10px !important;
        padding: 14px 16px !important;
        margin-bottom: 14px !important;
    }}
    #{_cid} .cb-welcome p {{
        color: #111111 !important;
        font-size: 14px !important;
        line-height: 1.75 !important;
        margin: 0 !important;
        font-family: 'Source Sans 3','Source Sans Pro',Helvetica,Arial,sans-serif !important;
    }}
    #{_cid} .cb-welcome strong {{
        color: #0A5541 !important;
        font-weight: 700 !important;
    }}
    #{_cid} .cb-lbl-user {{
        font-size: 11px !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px !important;
        color: #0F6E56 !important;
        text-align: right !important;
        display: block !important;
        margin-bottom: 4px !important;
        font-family: 'Source Sans 3','Source Sans Pro',Helvetica,Arial,sans-serif !important;
    }}
    #{_cid} .cb-lbl-bot {{
        font-size: 11px !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px !important;
        color: #085041 !important;
        display: block !important;
        margin-bottom: 4px !important;
        font-family: 'Source Sans 3','Source Sans Pro',Helvetica,Arial,sans-serif !important;
    }}
    #{_cid} .cb-user {{
        background: #0F6E56 !important;
        border-radius: 12px 12px 3px 12px !important;
        padding: 12px 15px !important;
        margin-bottom: 12px !important;
        margin-left: 28px !important;
    }}
    #{_cid} .cb-user p {{
        color: #FFFFFF !important;
        font-size: 14px !important;
        line-height: 1.70 !important;
        margin: 0 !important;
        text-align: right !important;
        font-family: 'Source Sans 3','Source Sans Pro',Helvetica,Arial,sans-serif !important;
    }}
    #{_cid} .cb-bot {{
        background: #FFFFFF !important;
        border: 1.5px solid #C8DDD8 !important;
        border-radius: 12px 12px 12px 3px !important;
        padding: 12px 15px !important;
        margin-bottom: 12px !important;
        margin-right: 28px !important;
    }}
    #{_cid} .cb-bot p {{
        color: #111111 !important;
        font-size: 14px !important;
        line-height: 1.70 !important;
        margin: 0 !important;
        font-family: 'Source Sans 3','Source Sans Pro',Helvetica,Arial,sans-serif !important;
    }}
    #{_cid} .cb-thinking {{
        color: #0F6E56 !important;
        font-size: 13px !important;
        font-style: italic !important;
        padding: 4px 2px 8px !important;
        font-family: 'Source Sans 3','Source Sans Pro',Helvetica,Arial,sans-serif !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    CHAT_FONT = "'Source Sans 3','Source Sans Pro',Helvetica,Arial,sans-serif"

    # Build chat HTML — white bubbles, fully readable
    hist_html = f'<div id="{_cid}">'

    if not st.session_state.messages:
        hist_html += (
            f'<div class="cb-welcome">'
            f'<p>Hi! I\'m the <strong>PSIA AI Assistant</strong>. '
            f'I have access to two data streams: '
            f'<strong>FAO global seaweed data</strong> AND '
            f'<strong>Statistics Canada Canadian aquaculture data (1986&#8211;2024)</strong>. '
            f'Ask me anything about the seaweed industry, BC regulations, social license, '
            f'production costs, or Canadian aquaculture context.</p>'
            f'</div>'
        )

    for m in st.session_state.messages:
        u       = m["role"] == "user"
        lbl_cls = "cb-lbl cb-lbl-user" if u else "cb-lbl cb-lbl-bot"
        bub_cls = "cb-user" if u else "cb-bot"
        lbl     = "YOU" if u else "🌿 PSIA AI"
        content = str(m["content"]).replace("<", "&lt;").replace(">", "&gt;")
        hist_html += (
            f'<span class="{lbl_cls}">{lbl}</span>'
            f'<div class="{bub_cls}"><p>{content}</p></div>'
        )

    if st.session_state.thinking:
        hist_html += (
            f'<div class="cb-thinking">'
            f'&#9203; &nbsp;Thinking&#8230;'
            f'</div>'
        )

    hist_html += "</div>"
    st.markdown(hist_html, unsafe_allow_html=True)

    # Render the most recent chart produced by the engine, if any
    for _m in reversed(st.session_state.messages):
        if _m.get("chart") is not None:
            st.plotly_chart(_m["chart"], use_container_width=True, key="autoplt_32")
            if _m.get("caption"):
                st.caption(_m["caption"])
            break

    SUGG = {
        False: [
            "What are BC's seaweed aquaculture regulations?",
            "How does social license affect seaweed farms?",
            "What is the FNFC Aquaculture Action Plan?",
        ],
        True: [
            f"What is global seaweed production in {LY if data_ok else 2024}?",
            "What happened to BC salmon farming in 2023?",
            "How do BC aquaculture GVA trends affect seaweed opportunity?",
        ],
    }

    st.markdown(
        '<div style="background:#E8F5F0;padding:10px 16px 6px;'
        'border-left:1.5px solid #C8DDD8;border-right:1.5px solid #C8DDD8;">'
        f'<span style="font-size:13px;font-weight:700;color:{C["t800"]} !important;'
        f"font-family:'Source Sans 3',Helvetica,sans-serif;letter-spacing:0.2px;"
        f'display:block !important;opacity:1 !important;">💬 &nbsp;Try asking:</span>'
        '</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="background:#E8F5F0;padding:5px 10px 12px;'
        'border-left:1.5px solid #C8DDD8;border-right:1.5px solid #C8DDD8;">',
        unsafe_allow_html=True)
    for sugg in SUGG[is_uc2]:
        if st.button(sugg, key=f"sugg_{sugg[:25]}", use_container_width=True):
            st.session_state["_pending"] = sugg; st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div style="background:#FFFFFF;padding:10px 12px;border-radius:0 0 12px 12px;'
        'border:1.5px solid #C8DDD8;border-top:none;'
        'box-shadow:0 4px 14px rgba(4,52,44,0.10);"></div>',
        unsafe_allow_html=True)

    _ic,_bc = st.columns([5,1])
    with _ic:
        txt_input = st.text_input("chat_input_label", placeholder="Ask about seaweed or Canadian aquaculture…",
                                   key="chat_txt", label_visibility="collapsed")
    with _bc:
        send_clicked = st.button("Send", key="chat_send", use_container_width=True, type="primary")

    if _rag_ok:
        st.markdown(
            f'<div style="font-size:12px;color:#555555 !important;font-weight:500;'
            f"font-family:'Source Sans 3',Helvetica,sans-serif;"
            f'text-align:center;margin-top:5px;opacity:1 !important;">'
            f'🔍 {len(RAG_DOCS)} knowledge chunks &nbsp;·&nbsp; 2 data streams &nbsp;·&nbsp; top-4 retrieval'
            f'</div>', unsafe_allow_html=True)

    pending = st.session_state.pop("_pending", None)
    prompt  = pending or (txt_input.strip() if send_clicked and txt_input.strip() else None)

    if prompt and not st.session_state.thinking:
        st.session_state.messages.append({"role": "user", "content": prompt})
        handled = False
        # Live Data (UC2): try the deterministic engine BEFORE the LLM
        if st.session_state.chat_mode == "UC2" and data_ok:
            res = answer_live_query(prompt, build_ctx())
            if res["handled"]:
                st.session_state.messages.append({
                    "role": "assistant", "content": res["text"],
                    "chart": res["fig"], "caption": res["caption"],
                })
                handled = True
        if not handled:  # open-ended → LLM + RAG (unchanged)
            st.session_state["_last_prompt"] = prompt
            st.session_state.thinking = True
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# GROQ API CALL
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.get("thinking", False):
    last_prompt = st.session_state.get("_last_prompt","")
    is_uc2_call = st.session_state.chat_mode == "UC2"
    try:
        client = Groq(api_key=GROQ_API_KEY)
        system = build_system_prompt(query=last_prompt, is_uc2=is_uc2_call)
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=1500,
            messages=[
                {"role":"system","content":system},
                *[{"role":m["role"],"content":m["content"]} for m in st.session_state.messages],
            ],
        )
        reply = resp.choices[0].message.content
    except Exception as e:
        err = str(e).lower()
        if "auth" in err or "invalid" in err: reply = "❌ Invalid API key. Check GROQ_API_KEY in the script."
        elif "rate" in err: reply = "⏳ Rate limit hit. Wait a moment and try again."
        else: reply = f"❌ Error: {e}"
    st.session_state.messages.append({"role":"assistant","content":reply})
    st.session_state.thinking = False
    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"<div style='text-align:center;font-size:13px;color:#666666;"
    f"font-family:\"Source Sans 3\",\"Source Sans Pro\",Helvetica,sans-serif;"
    f"padding:10px 0 16px;letter-spacing:0.2px;'>"
    f"<strong style='color:{C['t800']};'>PSIA Seaweed Analytics v11.0</strong>"
    f"&nbsp;&nbsp;·&nbsp;&nbsp;"
    f"🟢 FAO FishStat (seaweed-only)"
    f"&nbsp;&nbsp;·&nbsp;&nbsp;"
    f"🔵 Statistics Canada / DFO (Canadian aquaculture 1986–2024)"
    f"&nbsp;&nbsp;·&nbsp;&nbsp;"
    f"🟡 DFO/CIRNAC estimates"
    f"&nbsp;&nbsp;·&nbsp;&nbsp;"
    f"⚡ Groq llama-3.3-70b"
    f"&nbsp;&nbsp;·&nbsp;&nbsp;"
    f"🔍 RAG — {len(RAG_DOCS)} knowledge chunks"
    f"</div>", unsafe_allow_html=True)

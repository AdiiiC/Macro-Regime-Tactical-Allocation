"""
Custom UI Components for the Macro Regime Dashboard.
KPI cards, ticker tape, regime badges, sparklines, gauges.
"""

import numpy as np
import pandas as pd
import streamlit as st
from typing import Optional, List, Dict


def inject_custom_css():
    """Inject custom CSS theme into the Streamlit app."""
    import os
    css_path = os.path.join(os.path.dirname(__file__), "..", "styles", "theme.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_header():
    """Render branded header bar."""
    st.markdown("""
    <div class="header-bar">
        <div>
            <h1>◈ Macro Regime Engine</h1>
            <div class="subtitle">Tactical Asset Allocation System</div>
        </div>
        <div style="text-align: right; color: #6b7d93; font-size: 0.8rem;">
            Hidden Markov Model · 9 Asset Classes · Walk-Forward
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_ticker_tape(indicators: Dict[str, Dict]):
    """
    Render a market ticker tape at the top.

    Args:
        indicators: {name: {value: float, change: float, unit: str}}
    """
    items_html = ""
    for name, data in indicators.items():
        value = data.get("value", 0)
        change = data.get("change", 0)
        unit = data.get("unit", "")

        if change > 0:
            arrow = "▲"
            css_class = "up"
        elif change < 0:
            arrow = "▼"
            css_class = "down"
        else:
            arrow = "─"
            css_class = ""

        items_html += f"""
        <span class="ticker-item">
            <span class="label">{name}</span>
            <span class="value">{value:.2f}{unit}</span>
            <span class="{css_class}"> {arrow} {abs(change):.2f}</span>
        </span>
        """

    st.markdown(f'<div class="ticker-tape">{items_html}</div>', unsafe_allow_html=True)


def render_kpi_card(
    label: str,
    value: str,
    delta: Optional[str] = None,
    delta_positive: bool = True,
    regime_class: str = "",
    sparkline_data: Optional[List[float]] = None,
):
    """Render a single KPI card with optional sparkline."""
    delta_html = ""
    if delta:
        delta_class = "positive" if delta_positive else "negative"
        arrow = "↑" if delta_positive else "↓"
        delta_html = f'<div class="kpi-delta {delta_class}">{arrow} {delta}</div>'

    sparkline_html = ""
    if sparkline_data and len(sparkline_data) > 2:
        sparkline_html = _generate_sparkline_svg(sparkline_data)

    regime_css = f"regime-{regime_class.lower()}" if regime_class else ""

    return f"""
    <div class="kpi-card {regime_css}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
        {sparkline_html}
    </div>
    """


def render_kpi_row(cards: List[str]):
    """Render a row of KPI cards."""
    cards_html = "".join(cards)
    st.markdown(
        f'<div class="kpi-container">{cards_html}</div>',
        unsafe_allow_html=True,
    )


def render_regime_badge(regime: str, confidence: float):
    """Render a styled regime badge."""
    css_class = regime.lower()
    icon_map = {
        "expansion": "📈",
        "slowdown": "⚡",
        "recession": "📉",
        "recovery": "🌱",
    }
    icon = icon_map.get(css_class, "◉")

    st.markdown(f"""
    <div class="regime-badge {css_class}">
        {icon} {regime} <span style="opacity:0.7; margin-left:0.5rem; font-size:0.8rem;">
        ({confidence:.0%} confidence)</span>
    </div>
    """, unsafe_allow_html=True)


def render_confidence_gauge(confidence: float):
    """Render a circular confidence gauge."""
    # SVG circular gauge
    radius = 32
    circumference = 2 * np.pi * radius
    offset = circumference * (1 - confidence)

    if confidence >= 0.8:
        color = "#2ecc71"
    elif confidence >= 0.6:
        color = "#f39c12"
    else:
        color = "#e74c3c"

    svg = f"""
    <div style="text-align: center;">
        <svg width="80" height="80" viewBox="0 0 80 80">
            <circle cx="40" cy="40" r="{radius}" fill="none"
                    stroke="#2a3a4e" stroke-width="6"/>
            <circle cx="40" cy="40" r="{radius}" fill="none"
                    stroke="{color}" stroke-width="6"
                    stroke-dasharray="{circumference}"
                    stroke-dashoffset="{offset}"
                    stroke-linecap="round"
                    transform="rotate(-90 40 40)"/>
            <text x="40" y="38" text-anchor="middle" fill="#ffffff"
                  font-size="14" font-weight="700">{confidence:.0%}</text>
            <text x="40" y="52" text-anchor="middle" fill="#6b7d93"
                  font-size="8">CONFIDENCE</text>
        </svg>
    </div>
    """
    st.markdown(svg, unsafe_allow_html=True)


def render_sidebar_freshness(last_updated: str, model_status: str = "Active"):
    """Render data freshness indicator in sidebar."""
    status_color = "#2ecc71" if model_status == "Active" else "#f39c12"
    st.markdown(f"""
    <div class="sidebar-freshness">
        <span class="label">Data Status</span>
        <span class="value" style="color: {status_color};">● {model_status}</span>
        <br/>
        <span class="label" style="margin-top: 0.3rem;">Last Updated</span>
        <span class="value">{last_updated}</span>
    </div>
    """, unsafe_allow_html=True)


def render_section_header(title: str, icon: str = ""):
    """Render a styled section header."""
    st.markdown(f"""
    <div class="section-header">
        <span style="font-size: 1.3rem;">{icon}</span>
        <h2>{title}</h2>
    </div>
    """, unsafe_allow_html=True)


def render_whatif_panel():
    """Render What-If scenario panel header."""
    st.markdown("""
    <div class="whatif-panel">
        <h3>⚡ What-If Scenario Analysis</h3>
    </div>
    """, unsafe_allow_html=True)


def _generate_sparkline_svg(data: List[float], width: int = 100, height: int = 30) -> str:
    """Generate an inline SVG sparkline."""
    if not data or len(data) < 2:
        return ""

    data = np.array(data)
    min_val, max_val = data.min(), data.max()
    if max_val == min_val:
        normalized = np.full_like(data, 0.5)
    else:
        normalized = (data - min_val) / (max_val - min_val)

    points = []
    for i, val in enumerate(normalized):
        x = (i / (len(data) - 1)) * width
        y = height - (val * height)
        points.append(f"{x:.1f},{y:.1f}")

    polyline = " ".join(points)

    # Determine color based on trend
    color = "#2ecc71" if data[-1] > data[0] else "#e74c3c"

    return f"""
    <div class="kpi-sparkline">
        <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">
            <polyline points="{polyline}" fill="none"
                      stroke="{color}" stroke-width="1.5" stroke-linecap="round"/>
        </svg>
    </div>
    """


def styled_dataframe(df: pd.DataFrame, highlight_cols: Optional[List[str]] = None, height: Optional[int] = None):
    """Render a styled DataFrame with heatmap coloring."""
    styler = df.style

    if highlight_cols:
        for col in highlight_cols:
            if col in df.columns:
                styler = styler.background_gradient(
                    subset=[col], cmap="RdYlGn", vmin=-0.1, vmax=0.1
                )

    st.dataframe(styler, use_container_width=True, height=height)

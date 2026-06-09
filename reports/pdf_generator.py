"""
PDF Report Generator.
Creates professional investment memo with regime analysis, allocation, and metrics.
"""

import io
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, HRFlowable,
)
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.barcharts import VerticalBarChart


class PDFReportGenerator:
    """Generates professional PDF investment memos."""

    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Define custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name="ReportTitle",
            parent=self.styles["Title"],
            fontSize=24,
            spaceAfter=20,
            textColor=colors.HexColor("#1a237e"),
        ))
        self.styles.add(ParagraphStyle(
            name="SectionHeader",
            parent=self.styles["Heading1"],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.HexColor("#283593"),
        ))
        self.styles.add(ParagraphStyle(
            name="SubHeader",
            parent=self.styles["Heading2"],
            fontSize=13,
            spaceAfter=8,
            textColor=colors.HexColor("#3949ab"),
        ))
        self.styles.add(ParagraphStyle(
            name="BodyText",
            parent=self.styles["Normal"],
            fontSize=10,
            spaceAfter=6,
            leading=14,
        ))
        self.styles.add(ParagraphStyle(
            name="MetricValue",
            parent=self.styles["Normal"],
            fontSize=20,
            alignment=1,  # center
            textColor=colors.HexColor("#1b5e20"),
        ))
        self.styles.add(ParagraphStyle(
            name="Disclaimer",
            parent=self.styles["Normal"],
            fontSize=7,
            textColor=colors.grey,
            spaceBefore=20,
        ))

    def generate_report(
        self,
        current_regime: str,
        regime_confidence: float,
        allocation: pd.Series,
        backtest_metrics: Dict,
        regime_history: pd.Series,
        var_results: Optional[Dict] = None,
        leading_indicators: Optional[pd.DataFrame] = None,
        output_path: str = "investment_memo.pdf",
    ) -> str:
        """
        Generate complete investment memo PDF.

        Returns:
            Path to generated PDF file.
        """
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=2 * cm,
            leftMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
        )

        elements = []

        # ─── Header ───────────────────────────────────────────────────
        elements.append(Paragraph(
            "Macro Regime Tactical Allocation",
            self.styles["ReportTitle"],
        ))
        elements.append(Paragraph(
            f"Investment Strategy Memo — {datetime.now().strftime('%B %d, %Y')}",
            self.styles["BodyText"],
        ))
        elements.append(HRFlowable(
            width="100%", thickness=2, color=colors.HexColor("#1a237e"),
        ))
        elements.append(Spacer(1, 20))

        # ─── Executive Summary ────────────────────────────────────────
        elements.append(Paragraph("Executive Summary", self.styles["SectionHeader"]))

        regime_color_map = {
            "Expansion": "#2ecc71",
            "Slowdown": "#f39c12",
            "Recession": "#e74c3c",
            "Recovery": "#3498db",
        }
        regime_color = regime_color_map.get(current_regime, "#333333")

        elements.append(Paragraph(
            f'Current Economic Regime: <font color="{regime_color}">'
            f'<b>{current_regime}</b></font> '
            f'(Confidence: {regime_confidence:.1%})',
            self.styles["BodyText"],
        ))

        rationale_map = {
            "Expansion": "Strong growth with moderate inflation favors risk assets. Overweight equities and cyclicals.",
            "Slowdown": "Decelerating growth warrants defensive positioning. Rotate toward bonds and quality.",
            "Recession": "Capital preservation paramount. Maximum allocation to safe havens and cash.",
            "Recovery": "Early cycle recovery favors cyclical risk-on positioning.",
        }
        elements.append(Paragraph(
            f"<i>{rationale_map.get(current_regime, 'Unknown regime state.')}</i>",
            self.styles["BodyText"],
        ))
        elements.append(Spacer(1, 15))

        # ─── Current Allocation ───────────────────────────────────────
        elements.append(Paragraph("Target Portfolio Allocation", self.styles["SectionHeader"]))

        alloc_data = [["Asset Class", "Target Weight", "Category"]]
        for asset, weight in allocation.items():
            if weight > 0.001:
                category = self._categorize_asset(asset)
                alloc_data.append([asset, f"{weight:.1%}", category])

        alloc_table = Table(alloc_data, colWidths=[4 * cm, 3 * cm, 4 * cm])
        alloc_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#283593")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("ALIGN", (1, 0), (1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        elements.append(alloc_table)
        elements.append(Spacer(1, 20))

        # ─── Backtest Performance ─────────────────────────────────────
        elements.append(Paragraph("Historical Performance", self.styles["SectionHeader"]))

        metrics_data = [["Metric", "Strategy", "Benchmark (60/40)"]]
        key_metrics = [
            "Ann. Return", "Ann. Volatility", "Sharpe Ratio",
            "Max Drawdown", "Sortino Ratio", "Calmar Ratio", "Information Ratio",
        ]
        for metric in key_metrics:
            strat_key = f"{metric} (Strategy)"
            bench_key = f"{metric} (Benchmark)"
            strat_val = backtest_metrics.get(strat_key, "—")
            bench_val = backtest_metrics.get(bench_key, "—")
            if bench_val == "—" and metric in ["Sortino Ratio", "Calmar Ratio", "Information Ratio"]:
                bench_val = "—"
                strat_val = backtest_metrics.get(metric, "—")
            metrics_data.append([metric, strat_val, bench_val])

        metrics_table = Table(metrics_data, colWidths=[5 * cm, 3.5 * cm, 3.5 * cm])
        metrics_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1b5e20")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#e8f5e9")]),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        elements.append(metrics_table)
        elements.append(Spacer(1, 20))

        # ─── Risk Metrics (VaR) ───────────────────────────────────────
        if var_results:
            elements.append(Paragraph("Risk Assessment", self.styles["SectionHeader"]))

            risk_data = [["Risk Measure", "Value"]]
            risk_data.append(["VaR (95%)", f"{var_results.get('VaR_95%', 0):.2%}"])
            risk_data.append(["CVaR (95%)", f"{var_results.get('CVaR_95%', 0):.2%}"])
            risk_data.append(["VaR (99%)", f"{var_results.get('VaR_99%', 0):.2%}"])
            risk_data.append(["CVaR (99%)", f"{var_results.get('CVaR_99%', 0):.2%}"])
            risk_data.append(["Prob. of Loss", f"{var_results.get('prob_negative', 0):.1%}"])
            risk_data.append(["Expected Return", f"{var_results.get('mean_return', 0):.2%}"])

            risk_table = Table(risk_data, colWidths=[5 * cm, 4 * cm])
            risk_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#b71c1c")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (1, 0), (1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            elements.append(risk_table)
            elements.append(Spacer(1, 20))

        # ─── Regime History Summary ───────────────────────────────────
        elements.append(Paragraph("Regime Distribution", self.styles["SectionHeader"]))

        regime_counts = regime_history.value_counts()
        regime_data = [["Regime", "Months", "% of History"]]
        for regime, count in regime_counts.items():
            pct = count / len(regime_history)
            regime_data.append([regime, str(count), f"{pct:.1%}"])

        regime_table = Table(regime_data, colWidths=[4 * cm, 3 * cm, 3 * cm])
        regime_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4a148c")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        elements.append(regime_table)
        elements.append(Spacer(1, 30))

        # ─── Disclaimer ───────────────────────────────────────────────
        elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
        elements.append(Paragraph(
            "DISCLAIMER: This report is generated by a quantitative model for educational and research purposes only. "
            "Past performance is not indicative of future results. This does not constitute investment advice. "
            "The model uses historical data and statistical techniques that may not capture future market dynamics. "
            "All investments carry risk of loss.",
            self.styles["Disclaimer"],
        ))

        # Build PDF
        doc.build(elements)
        return output_path

    @staticmethod
    def _categorize_asset(asset: str) -> str:
        """Categorize asset class for reporting."""
        categories = {
            "US_Equity": "Growth",
            "Intl_Equity": "Growth",
            "EM_Equity": "Growth",
            "US_Bonds": "Defensive",
            "TIPS": "Inflation Hedge",
            "Gold": "Safe Haven",
            "Commodities": "Real Assets",
            "Real_Estate": "Real Assets",
            "Cash": "Liquidity",
        }
        return categories.get(asset, "Other")

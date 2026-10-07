"""
Milestone 10: Multi-Format Report Exporter & Presentation Formatters (JSON, Markdown, HTML)
"""
from typing import Dict, Any
from app.services.analytical_report.models import AnalyticalReport


class ReportFormatter:
    """
    Translates canonical AnalyticalReport DTO into presentation formats.
    Maintains clean separation between presentation formatting and analytical computation.
    """

    @staticmethod
    def to_json_dict(report: AnalyticalReport) -> Dict[str, Any]:
        """Returns standard JSON dictionary representation."""
        return report.model_dump(mode="json")

    @staticmethod
    def to_markdown(report: AnalyticalReport) -> str:
        """Renders report as a structured research Markdown document."""
        m = report.metadata
        b = report.baseline
        c = report.modality_coverage
        d = report.deviation_summary

        p_start = report.period.period_start.strftime("%Y-%m-%d %H:%M:%S UTC") if report.period.period_start else "Session Beginning"
        p_end = report.period.period_end.strftime("%Y-%m-%d %H:%M:%S UTC") if report.period.period_end else "Latest Observation"

        lines = [
            f"# Longitudinal Analytical Report — Participant `{m.user_id}`",
            f"**Report ID**: `{m.report_id}` | **Generated**: {m.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')} | **Version**: `{m.feature_vector_version}`",
            "",
            "> **MANDATORY RESEARCH DISCLAIMER**:",
            f"> {report.disclaimer}",
            "",
            "---",
            "",
            "## 1. Executive Summary & Baseline Status",
            "",
            f"- **Observation Period**: {p_start} to {p_end}",
            f"- **Analyzed Observations**: {m.observation_count} turns",
            f"- **Baseline Status**: **{b.status.upper()}** (N={b.sample_count} prior observations, Established={b.is_baseline_established})",
            f"- **Overall Statistical Classification**: **{d.statistical_variation_category}**",
            f"  *{d.category_description}*",
            "",
            "---",
            "",
            "## 2. Modality Completeness & Coverage",
            "",
            "| Modality | Measured Turns | Total Turns | Coverage Percentage |",
            "|---|---|---|---|",
            f"| Communication Dynamics | {c.communication.measured_count} | {c.communication.total_count} | {c.communication.percentage}% |",
            f"| Acoustic Signal Measurements | {c.acoustic.measured_count} | {c.acoustic.total_count} | {c.acoustic.percentage}% |",
            f"| NLP & Linguistic Metrics | {c.nlp.measured_count} | {c.nlp.total_count} | {c.nlp.percentage}% |",
            "",
            "---",
            "",
            "## 3. Aggregate Statistical Variation Metrics",
            "",
            "| Metric | Mean | Median | Minimum | Maximum | Sample Std |",
            "|---|---|---|---|---|---|",
            f"| **Mean Absolute |Z|** | {d.mean_absolute_z.mean:.3f}σ | {d.mean_absolute_z.median:.3f}σ | {d.mean_absolute_z.minimum:.3f}σ | {d.mean_absolute_z.maximum:.3f}σ | {d.mean_absolute_z.std:.3f} |",
            f"| **RMS Z** | {d.rms_z.mean:.3f}σ | {d.rms_z.median:.3f}σ | {d.rms_z.minimum:.3f}σ | {d.rms_z.maximum:.3f}σ | {d.rms_z.std:.3f} |",
            f"| **Standardized Euclidean Distance** | {d.standardized_euclidean_distance.mean:.3f} | {d.standardized_euclidean_distance.median:.3f} | {d.standardized_euclidean_distance.minimum:.3f} | {d.standardized_euclidean_distance.maximum:.3f} | {d.standardized_euclidean_distance.std:.3f} |",
            "",
            "---",
            "",
            "## 4. Top Behavioral Feature Shifts (|Z| Descending)",
            "",
            "| Index | Feature Name | Modality | Mean |Z| | Max |Z| | Latest Signed Z | Baseline Reference (μ ± σ) |",
            "|---|---|---|---|---|---|---|",
        ]

        for s in report.feature_shifts:
            sign_str = f"+{s.latest_signed_z:.2f}" if s.latest_signed_z > 0 else f"{s.latest_signed_z:.2f}"
            lines.append(
                f"| `{s.feature_index:03d}` | **{s.feature_name}** | {s.modality} | {s.mean_absolute_z:.3f}σ | {s.max_absolute_z:.3f}σ | {sign_str}σ | {s.baseline_mean:.2f} ± {s.baseline_std:.2f} |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## 5. Longitudinal Temporal Trends (OLS & Moving Averages)",
            "",
            f"**Distribution**: {report.trend_summary.increasing_count} Increasing, {report.trend_summary.decreasing_count} Decreasing, {report.trend_summary.stable_count} Stable",
            "",
            "| Feature Name | Modality | Trend Direction | Normalized Slope (/hr) | Latest SMA-3 | Latest EMA-3 | Rate of Change |",
            "|---|---|---|---|---|---|---|",
        ])

        for t in report.trend_summary.top_trends:
            dir_icon = "↑ Increasing" if t.direction == "increasing" else "↓ Decreasing" if t.direction == "decreasing" else "→ Stable"
            sma_str = f"{t.latest_sma3:.3f}" if t.latest_sma3 is not None else "N/A"
            ema_str = f"{t.latest_ema3:.3f}" if t.latest_ema3 is not None else "N/A"
            roc_str = f"{t.latest_roc:.3f}" if t.latest_roc is not None else "N/A"
            lines.append(
                f"| **{t.feature_name}** | {t.modality} | {dir_icon} | {t.normalized_slope:+.4f} | {sma_str} | {ema_str} | {roc_str} |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## 6. Research Limitations & Methodology Notes",
            ""
        ])
        for limit in report.limitations:
            lines.append(f"- {limit}")

        lines.extend([
            "",
            "---",
            "*Report assembled deterministically by Multimodal Behavioral Analysis Platform — Milestone 10 Engine.*"
        ])

        return "\n".join(lines)

    @staticmethod
    def to_html(report: AnalyticalReport) -> str:
        """Renders report as a clean, styled HTML page."""
        md_content = ReportFormatter.to_markdown(report)
        # Convert simple markdown headers and tables to clean semantic HTML
        import html
        escaped_md = html.escape(md_content)
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Analytical Report - {report.metadata.user_id}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #1e293b; background: #f8fafc; margin: 0; padding: 40px 20px; }}
        .container {{ max-width: 900px; margin: 0 auto; background: #ffffff; padding: 40px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); border: 1px solid #e2e8f0; }}
        h1 {{ color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 12px; margin-top: 0; font-size: 24px; }}
        h2 {{ color: #1e293b; margin-top: 30px; font-size: 18px; border-bottom: 1px solid #f1f5f9; padding-bottom: 8px; }}
        .disclaimer {{ background: #eff6ff; border-left: 4px solid #3b82f6; padding: 14px 18px; border-radius: 6px; margin: 20px 0; color: #1e40af; font-size: 13px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 13px; }}
        th, td {{ border: 1px solid #e2e8f0; padding: 8px 12px; text-align: left; }}
        th {{ background: #f8fafc; font-weight: 600; color: #475569; }}
        tr:nth-child(even) {{ background: #fcfdfe; }}
        .badge {{ display: inline-block; padding: 3px 8px; border-radius: 9999px; font-size: 11px; font-weight: 600; }}
        .badge-stable {{ background: #dcfce7; color: #166534; }}
        .badge-noticeable {{ background: #fef3c7; color: #92400e; }}
        .badge-significant {{ background: #fee2e2; color: #991b1b; }}
        pre {{ white-space: pre-wrap; font-family: monospace; background: #f1f5f9; padding: 16px; border-radius: 8px; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Longitudinal Behavioral Analytical Report</h1>
        <p><strong>Participant:</strong> {report.metadata.user_id} | <strong>Report ID:</strong> {report.metadata.report_id} | <strong>Generated:</strong> {report.metadata.generated_at.isoformat()}</p>
        
        <div class="disclaimer">
            <strong>MANDATORY RESEARCH DISCLAIMER:</strong><br>
            {report.disclaimer}
        </div>

        <pre>{escaped_md}</pre>
    </div>
</body>
</html>"""

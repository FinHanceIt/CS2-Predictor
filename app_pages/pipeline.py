"""Pipeline page: control buttons for refresh / predict / track."""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
DATA_PARQUET = ROOT / "data" / "clean" / "cs2_matches.parquet"
ENSEMBLE_PATH = ROOT / "reference" / "ensemble.json"

DEFAULT_TIMEOUT_REFRESH = 600   # seconds
DEFAULT_TIMEOUT_FAST = 300      # predict-no-refresh / track


def _mtime(p: Path) -> datetime | None:
    if not p.exists():
        return None
    return datetime.fromtimestamp(p.stat().st_mtime)


@st.cache_data(ttl=30)
def _latest_report_date() -> str | None:
    if not REPORTS_DIR.exists():
        return None
    dates = []
    for p in REPORTS_DIR.glob("*_cs2"):
        if p.is_dir():
            stem = p.name.replace("_cs2", "")
            if len(stem) == 8 and stem.isdigit():
                dates.append(stem)
    if not dates:
        return None
    return sorted(dates, reverse=True)[0]


@st.cache_data(ttl=30)
def _ensemble_info() -> dict | None:
    if not ENSEMBLE_PATH.exists():
        return None
    try:
        return json.loads(ENSEMBLE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def _run_command(cmd: list[str], timeout: int) -> tuple[int, str, str]:
    """Run a subprocess from project root. Returns (returncode, stdout, stderr)."""
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired as e:
        return -1, e.stdout or "", (e.stderr or "") + f"\n[TimeoutExpired after {timeout}s]"
    except FileNotFoundError as e:
        return -1, "", f"Command not found: {e}"
    except Exception as e:
        return -1, "", f"Unexpected error: {e!r}"


def _show_result(rc: int, stdout: str, stderr: str, key_prefix: str):
    if rc == 0:
        st.success("Done! Switch to the **Dashboard** tab to see predictions.")
    else:
        st.error(f"Command exited with code {rc}.")
    with st.expander("Command output", expanded=(rc != 0)):
        if stdout:
            st.markdown("**stdout**")
            st.code(stdout, language="text")
        if stderr:
            st.markdown("**stderr**")
            st.code(stderr, language="text")
        if not stdout and not stderr:
            st.caption("(no output captured)")


def render():
    st.title("Pipeline")
    st.caption("Trigger the CS2-Predictor scripts and view their logs.")

    # ---- Status panel ----
    st.subheader("Status")
    s1, s2, s3 = st.columns(3)

    data_mt = _mtime(DATA_PARQUET)
    s1.metric(
        "Cached match data",
        data_mt.strftime("%Y-%m-%d %H:%M") if data_mt else "missing",
        help=str(DATA_PARQUET),
    )

    latest = _latest_report_date()
    s2.metric(
        "Latest predictions",
        latest if latest else "none",
        help="Most recent reports/<YYYYMMDD>_cs2 directory",
    )

    ens = _ensemble_info()
    if ens:
        weights = ens.get("weights", {})
        wstr = " / ".join(f"{k}:{v:.2f}" for k, v in weights.items()) if weights else "loaded"
        s3.metric("Ensemble", wstr, help=f"intercept = {ens.get('intercept', 0):+.3f}")
    else:
        s3.metric("Ensemble", "not loaded", help="reference/ensemble.json missing")

    st.markdown("---")

    # ---- Settings ----
    st.subheader("Settings")
    c1, c2 = st.columns(2)
    with c1:
        days_ahead = st.number_input(
            "--days-ahead", min_value=1, max_value=14, value=2, step=1,
            help="How many days in the future to fetch upcoming matches.",
        )
    with c2:
        tier_floor = st.selectbox(
            "--tier-floor", options=["s", "a", "b", "c", "d"], index=3,
            help="Minimum tournament tier to include (lower = more matches).",
        )

    st.markdown("---")

    # ---- Buttons ----
    st.subheader("Actions")
    py = sys.executable or "python"

    b1, b2, b3 = st.columns(3)
    refresh_clicked = b1.button("Refresh data + predict", type="primary",
                                 help="Re-scrape bo3.gg history then predict. Can take several minutes.")
    predict_clicked = b2.button("Predict (no refresh)",
                                 help="Use cached match data, just re-run the models.")
    track_clicked = b3.button("Track results",
                               help="Pull finished match outcomes and update accuracy stats.")

    # Slot for output (so it appears below the buttons regardless of which was clicked)
    output_slot = st.container()

    def _common_args() -> list[str]:
        return ["--days-ahead", str(int(days_ahead)), "--tier-floor", tier_floor]

    if refresh_clicked:
        cmd = [py, "scripts/predict_today.py", "--refresh-data", *_common_args()]
        with output_slot:
            st.write(f"Running: `{' '.join(cmd)}`")
            with st.spinner("Refreshing data and running predictions... (up to 10 minutes)"):
                rc, out, err = _run_command(cmd, timeout=DEFAULT_TIMEOUT_REFRESH)
            _show_result(rc, out, err, "refresh")
            # Bust caches so the dashboard sees new files
            _latest_report_date.clear()
            _ensemble_info.clear()

    elif predict_clicked:
        cmd = [py, "scripts/predict_today.py", *_common_args()]
        with output_slot:
            st.write(f"Running: `{' '.join(cmd)}`")
            with st.spinner("Running predictions on cached data..."):
                rc, out, err = _run_command(cmd, timeout=DEFAULT_TIMEOUT_FAST)
            _show_result(rc, out, err, "predict")
            _latest_report_date.clear()
            _ensemble_info.clear()

    elif track_clicked:
        cmd = [py, "scripts/track_results.py"]
        with output_slot:
            st.write(f"Running: `{' '.join(cmd)}`")
            with st.spinner("Tracking results..."):
                rc, out, err = _run_command(cmd, timeout=DEFAULT_TIMEOUT_FAST)
            _show_result(rc, out, err, "track")
            _latest_report_date.clear()

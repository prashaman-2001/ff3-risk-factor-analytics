# app.py
# ------------------------------------------------------------
# FF3 Risk + Factor Analytics (Daily) using LOCAL FILES
#
# Uses:
#   - Local Ken French daily FF3 file:
#       /Users/prashamanmainali/Documents/Streamlit app /DATA/F-F_Research_Data_Factors_daily.csv
#   - Local S&P 500 tickers list:
#       /Users/prashamanmainali/Documents/Streamlit app /DATA/SP500.csv
#   - yfinance for daily adjusted prices
#
# Run (IMPORTANT):
#   streamlit run app.py
# ------------------------------------------------------------

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
import yfinance as yf

# ----------------------------
# Local file paths (YOUR MAC PATHS)
# ----------------------------
FF3_DAILY_PATH = "/Users/prashamanmainali/Documents/Streamlit app /DATA/F-F_Research_Data_Factors_daily.csv"
SP500_PATH = "/Users/prashamanmainali/Documents/Streamlit app /DATA/SP500.csv"

# ----------------------------
# Analysis window controls
# ----------------------------
END_DATE = "2025-12-31"  # cap analysis at end of 2025

# ----------------------------
# App config
# ----------------------------
st.set_page_config(page_title="FF3 (Daily) — Risk + Factor Analytics", page_icon="📈", layout="wide")

DEFAULT_TICKERS = ["KO", "AAPL", "ABBV"]  # NVDA removed, AAPL added
DEFAULT_BENCH = "SPY"

TRADING_DAYS = 252
ROLLING_WINDOW_DEFAULT = 252  # ~1 trading year
MIN_OBS_FOR_REG = 252  # require at least ~1 year daily data for regression


# ----------------------------
# Formatting helpers
# ----------------------------
def fmt_pct(x: float, nd: int = 2) -> str:
    if pd.isna(x):
        return "—"
    return f"{x * 100:,.{nd}f}%"


def fmt_num(x: float, nd: int = 3) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:,.{nd}f}"


def ann_return_from_daily(r: pd.Series) -> float:
    if len(r) == 0:
        return np.nan
    return (1.0 + r).prod() ** (TRADING_DAYS / len(r)) - 1.0


def ann_vol_from_daily(r: pd.Series) -> float:
    if len(r) < 2:
        return np.nan
    return float(r.std(ddof=1) * math.sqrt(TRADING_DAYS))


def max_drawdown_from_returns(r: pd.Series) -> float:
    if len(r) == 0:
        return np.nan
    equity = (1.0 + r).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def sharpe_from_daily_excess(excess_r: pd.Series) -> float:
    if len(excess_r) < 2:
        return np.nan
    mu = excess_r.mean()
    sd = excess_r.std(ddof=1)
    if sd == 0 or pd.isna(sd):
        return np.nan
    return float((mu / sd) * math.sqrt(TRADING_DAYS))


def sortino_from_daily_excess(excess_r: pd.Series) -> float:
    if len(excess_r) < 2:
        return np.nan
    downside = excess_r[excess_r < 0]
    dd = downside.std(ddof=1)
    if dd == 0 or pd.isna(dd):
        return np.nan
    return float((excess_r.mean() / dd) * math.sqrt(TRADING_DAYS))


def hist_var_cvar(r: pd.Series, alpha: float = 0.05) -> Tuple[float, float]:
    if len(r) == 0:
        return np.nan, np.nan
    r = r.dropna()
    if len(r) == 0:
        return np.nan, np.nan
    var = float(np.quantile(r, alpha))
    tail = r[r <= var]
    cvar = float(tail.mean()) if len(tail) else np.nan
    return var, cvar


# ----------------------------
# Loaders (LOCAL)
# ----------------------------
@st.cache_data(show_spinner=False)
def load_sp500_tickers_from_file(path: str) -> List[str]:
    """
    Accepts columns: Symbol / Ticker / or first column fallback.
    Normalizes to Yahoo format (BRK.B -> BRK-B).
    """
    df = pd.read_csv(path)

    col = None
    for c in ["Symbol", "Ticker", "symbol", "ticker"]:
        if c in df.columns:
            col = c
            break
    if col is None:
        col = df.columns[0]

    tickers = (
        df[col]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(".", "-", regex=False)
        .replace({"NAN": np.nan})
        .dropna()
        .unique()
        .tolist()
    )

    return sorted(tickers)


@st.cache_data(show_spinner=False)
def load_ff3_daily_from_kf_csv(path: str) -> pd.DataFrame:
    """
    Loads DAILY FF3 from Ken French CSV that contains headers/notes + a table.

    Expected daily table header is either:
      Date,Mkt-RF,SMB,HML,RF
    or
      ,Mkt-RF,SMB,HML,RF

    Returns DataFrame:
      index: datetime daily
      columns: ['Mkt-RF','SMB','HML','RF'] in DECIMALS (not percent)
    """
    with open(path, "r", encoding="latin1") as f:
        lines = f.read().splitlines()

    header_idx = None
    for i, line in enumerate(lines):
        s = line.strip().replace(" ", "")
        if s in ("Date,Mkt-RF,SMB,HML,RF", ",Mkt-RF,SMB,HML,RF"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not find FF3 daily header row in the file.")

    start = header_idx + 1

    # End at first blank line after the daily table
    end = None
    for i in range(start, len(lines)):
        if lines[i].strip() == "":
            end = i
            break
    if end is None:
        end = len(lines)

    rows = lines[start:end]
    rows = [r for r in rows if r.count(",") >= 4]

    df = pd.DataFrame([r.split(",")[:5] for r in rows], columns=["Date", "Mkt-RF", "SMB", "HML", "RF"])
    df["Date"] = pd.to_datetime(df["Date"].str.strip(), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["Date"])

    for c in ["Mkt-RF", "SMB", "HML", "RF"]:
        df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0

    df = df.dropna().set_index("Date").sort_index()
    return df[["Mkt-RF", "SMB", "HML", "RF"]]


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_prices_daily_adjclose(tickers: Tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    """
    Downloads daily adjusted close using yfinance auto_adjust=True.
    """
    raw = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        px = raw["Close"].copy()
    else:
        px = raw[["Close"]].rename(columns={"Close": tickers[0]})

    px = px.dropna(how="all")
    px.columns = [str(c).upper() for c in px.columns]
    return px


def prices_to_daily_returns(px: pd.DataFrame) -> pd.DataFrame:
    return px.pct_change().dropna(how="all")


# ----------------------------
# Regression engine
# ----------------------------
@dataclass
class FF3Result:
    ticker: str
    n_obs: int
    alpha_daily: float
    alpha_annual: float
    beta_mkt: float
    beta_smb: float
    beta_hml: float
    r2: float
    adj_r2: float
    t_alpha: float
    t_mkt: float
    t_smb: float
    t_hml: float
    p_alpha: float
    p_mkt: float
    p_smb: float
    p_hml: float


def run_ff3_regression_daily(
    asset_ret_d: pd.Series, ff3_d: pd.DataFrame
) -> Tuple[Optional[FF3Result], Optional[sm.regression.linear_model.RegressionResultsWrapper], pd.DataFrame]:
    """
    Model:
      Ri - RF = alpha + b_m(Mkt-RF) + b_s(SMB) + b_h(HML) + eps
    All are DAILY returns in decimals.
    """
    df = pd.concat([asset_ret_d.rename("Ri"), ff3_d], axis=1).dropna()
    if df.shape[0] < MIN_OBS_FOR_REG:
        return None, None, df

    y = df["Ri"] - df["RF"]
    X = sm.add_constant(df[["Mkt-RF", "SMB", "HML"]])

    model = sm.OLS(y, X).fit()

    a = float(model.params.get("const", np.nan))
    bm = float(model.params.get("Mkt-RF", np.nan))
    bs = float(model.params.get("SMB", np.nan))
    bh = float(model.params.get("HML", np.nan))

    alpha_ann = (1.0 + a) ** TRADING_DAYS - 1.0 if not pd.isna(a) else np.nan

    res = FF3Result(
        ticker=str(asset_ret_d.name),
        n_obs=int(df.shape[0]),
        alpha_daily=a,
        alpha_annual=alpha_ann,
        beta_mkt=bm,
        beta_smb=bs,
        beta_hml=bh,
        r2=float(model.rsquared),
        adj_r2=float(model.rsquared_adj),
        t_alpha=float(model.tvalues.get("const", np.nan)),
        t_mkt=float(model.tvalues.get("Mkt-RF", np.nan)),
        t_smb=float(model.tvalues.get("SMB", np.nan)),
        t_hml=float(model.tvalues.get("HML", np.nan)),
        p_alpha=float(model.pvalues.get("const", np.nan)),
        p_mkt=float(model.pvalues.get("Mkt-RF", np.nan)),
        p_smb=float(model.pvalues.get("SMB", np.nan)),
        p_hml=float(model.pvalues.get("HML", np.nan)),
    )
    return res, model, df


def rolling_ff3_daily(asset_ret_d: pd.Series, ff3_d: pd.DataFrame, window: int) -> pd.DataFrame:
    merged = pd.concat([asset_ret_d.rename("Ri"), ff3_d], axis=1).dropna()
    if merged.shape[0] < window:
        return pd.DataFrame()

    out = []
    idx = merged.index
    for i in range(window - 1, len(merged)):
        sub = merged.iloc[i - window + 1 : i + 1]
        y = sub["Ri"] - sub["RF"]
        X = sm.add_constant(sub[["Mkt-RF", "SMB", "HML"]])
        try:
            m = sm.OLS(y, X).fit()
            out.append(
                {
                    "Date": idx[i],
                    "alpha": float(m.params.get("const", np.nan)),
                    "beta_mkt": float(m.params.get("Mkt-RF", np.nan)),
                    "beta_smb": float(m.params.get("SMB", np.nan)),
                    "beta_hml": float(m.params.get("HML", np.nan)),
                    "r2": float(m.rsquared),
                }
            )
        except Exception:
            out.append({"Date": idx[i], "alpha": np.nan, "beta_mkt": np.nan, "beta_smb": np.nan, "beta_hml": np.nan, "r2": np.nan})

    df_out = pd.DataFrame(out).set_index("Date").sort_index()
    return df_out


# ----------------------------
# Risk metrics
# ----------------------------
def risk_metrics_daily(r: pd.Series, rf: pd.Series) -> Dict[str, float]:
    ex = r - rf
    var5, cvar5 = hist_var_cvar(r, 0.05)
    return {
        "Ann Return": ann_return_from_daily(r),
        "Ann Vol": ann_vol_from_daily(r),
        "Max Drawdown": max_drawdown_from_returns(r),
        "Sharpe": sharpe_from_daily_excess(ex),
        "Sortino": sortino_from_daily_excess(ex),
        "VaR 5% (d)": var5,
        "CVaR 5% (d)": cvar5,
        "Obs (days)": float(len(r)),
    }


# ----------------------------
# UI
# ----------------------------
st.title("Fama–French 3-Factor (Daily) — Risk + Factor Analytics")
st.caption(
    "Daily FF3 regressions using a local Ken French factor file and a local S&P 500 ticker universe file. "
    "Prices are pulled via yfinance (auto-adjusted)."
)

with st.sidebar:
    st.header("Inputs")

    try:
        sp500 = load_sp500_tickers_from_file(SP500_PATH)
        st.success(f"Loaded {len(sp500):,} S&P 500 tickers from SP500.csv")
    except Exception as e:
        sp500 = []
        st.error(f"Failed to load SP500.csv: {e}")

    tickers = st.multiselect(
        "Select S&P 500 tickers",
        options=sp500,
        default=[t for t in DEFAULT_TICKERS if t in sp500] if sp500 else DEFAULT_TICKERS,
        help="These options come from your SP500.csv file.",
    )

    bench = st.text_input("Benchmark ticker (for context)", value=DEFAULT_BENCH).strip().upper()

    start_year = st.slider("Start year", min_value=1990, max_value=2025, value=2010)
    start_date = f"{start_year}-01-01"

    rolling_window = st.slider(
        "Rolling window (trading days)",
        min_value=126,
        max_value=756,
        value=ROLLING_WINDOW_DEFAULT,
        step=63,
        help="Number of trading days used in each rolling regression window.",
    )

    show_debug = st.toggle("Show debug tables", value=False)

if not tickers:
    st.info("Select at least one ticker from the sidebar.")
    st.stop()

# Determine tickers to download (include benchmark if provided)
all_tickers = sorted(list(set(tickers + ([bench] if bench else []))))

# Load factors (local)
try:
    with st.spinner("Loading daily FF3 factors from local file..."):
        ff3_d = load_ff3_daily_from_kf_csv(FF3_DAILY_PATH)
    st.caption(f"FF3 factor sample range: {ff3_d.index.min().date()} → {ff3_d.index.max().date()}")
except Exception as e:
    st.error(f"Could not load FF3 daily factors from:\n{FF3_DAILY_PATH}\n\nError: {e}")
    st.stop()

# Cap factors to the analysis end date
ff3_d = ff3_d.loc[ff3_d.index <= pd.to_datetime(END_DATE)]

# Load prices (yfinance)
with st.spinner("Downloading price data (yfinance)..."):
    px_d = load_prices_daily_adjclose(tuple(all_tickers), start=start_date, end=END_DATE)

if px_d.empty:
    st.error(
        "No price data returned. This is often Yahoo rate limiting.\n\n"
        "Try again later, reduce the number of tickers, or change the start year."
    )
    st.stop()

rets_d = prices_to_daily_returns(px_d)

# Align to factors
common = rets_d.index.intersection(ff3_d.index)
rets_d = rets_d.loc[common]
ff3_d = ff3_d.loc[common]

# Handle partial/missing downloads gracefully (rate limits, bad tickers, etc.)
available = [t for t in all_tickers if t in rets_d.columns]
missing = [t for t in all_tickers if t not in rets_d.columns]

if missing:
    st.warning(
        "Some tickers did not download (often due to Yahoo rate limits). They will be excluded: "
        + ", ".join(missing)
    )

if not available:
    st.error(
        "No price series were successfully downloaded. This is usually Yahoo rate limiting.\n\n"
        "Fixes: wait 5–30 minutes and rerun, reduce the number of tickers, or switch data source."
    )
    st.stop()

# Only keep tickers we actually have
all_tickers = available

# If benchmark missing, note it
if bench and bench not in all_tickers:
    st.info("Benchmark was not downloaded successfully and will be omitted from charts/tables.")

st.caption(f"Analysis window: {start_date} → {END_DATE}")

if len(common) < MIN_OBS_FOR_REG:
    st.warning(
        f"Only {len(common):,} overlapping trading days between prices and FF3 factors. "
        f"FF3 regression needs at least {MIN_OBS_FOR_REG} days. Try an earlier start year."
    )

tabs = st.tabs(["Overview (Risk)", "FF3 Regression", "Rolling Stability", "Cross-Ticker Comparison"])

# ----------------------------
# Tab 1: Overview (Risk)
# ----------------------------
with tabs[0]:
    st.subheader("Performance and risk overview (daily)")

    norm = (1.0 + rets_d[all_tickers]).cumprod()
    norm = 100.0 * (norm / norm.iloc[0])
    st.line_chart(norm, height=360)
    st.caption(
        "Shows the cumulative performance of each stock over time, normalized to the same starting value "
        "to enable direct comparison of relative growth and drawdowns."
    )

    rf_series = ff3_d["RF"]

    rows = []
    for t in all_tickers:
        if t not in rets_d.columns:
            continue
        r = rets_d[t].dropna()
        if r.empty:
            continue
        rf_aligned = rf_series.loc[r.index]
        m = risk_metrics_daily(r, rf_aligned)
        rows.append(
            {
                "Ticker": t,
                "Ann Return": m["Ann Return"],
                "Ann Vol": m["Ann Vol"],
                "Max Drawdown": m["Max Drawdown"],
                "Sharpe": m["Sharpe"],
                "Sortino": m["Sortino"],
                "VaR 5% (d)": m["VaR 5% (d)"],
                "CVaR 5% (d)": m["CVaR 5% (d)"],
                "Obs (days)": int(m["Obs (days)"]),
            }
        )

    risk_df = pd.DataFrame(rows).set_index("Ticker").sort_index()

    disp = risk_df.copy()
    for c in ["Ann Return", "Ann Vol", "Max Drawdown", "VaR 5% (d)", "CVaR 5% (d)"]:
        disp[c] = disp[c].map(lambda x: fmt_pct(x, 2))
    for c in ["Sharpe", "Sortino"]:
        disp[c] = disp[c].map(lambda x: fmt_num(x, 2))

    st.dataframe(disp, use_container_width=True)
    st.caption(
        "Summarizes key risk and return statistics for each stock, including volatility, drawdowns, and risk-adjusted performance measures."
    )

    if show_debug:
        st.write("Daily returns (head):")
        st.dataframe(rets_d.head(), use_container_width=True)
        st.write("FF3 daily factors (head):")
        st.dataframe(ff3_d.head(), use_container_width=True)

# ----------------------------
# Tab 2: FF3 Regression
# ----------------------------
with tabs[1]:
    st.subheader("FF3 regression (daily excess returns)")

    # Only allow tickers that actually exist in returns data
    ticker_options = [t for t in tickers if t in rets_d.columns]
    if not ticker_options:
        st.error("None of the selected tickers downloaded successfully. Try fewer tickers or rerun later.")
        st.stop()

    pick = st.selectbox("Choose a ticker to inspect", options=ticker_options, index=0)
    series = rets_d[pick].dropna()

    res, model, merged = run_ff3_regression_daily(series.rename(pick), ff3_d)

    if res is None or model is None:
        st.warning(
            f"Not enough overlapping daily data to run regression (need ~{MIN_OBS_FOR_REG} trading days). "
            "Try an earlier start year."
        )
    else:
        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Alpha (annualized)", fmt_pct(res.alpha_annual, 2))
            st.metric("Alpha (daily)", fmt_pct(res.alpha_daily, 4))
        with c2:
            st.metric("Beta (MKT-RF)", fmt_num(res.beta_mkt, 3))
            st.metric("Beta (SMB)", fmt_num(res.beta_smb, 3))
            st.metric("Beta (HML)", fmt_num(res.beta_hml, 3))
        with c3:
            st.metric("R²", fmt_num(res.r2, 3))
            st.metric("Obs (days)", f"{res.n_obs:,}")

        st.caption(
            "Alpha represents the portion of returns not explained by common risk factors, estimated in-sample after controlling for systematic exposures."
        )
        st.caption(
            "R² indicates how much of the stock’s return variability is explained by the Fama–French factor model."
        )

        coef_tbl = pd.DataFrame(
            {
                "coef": {
                    "alpha (const)": res.alpha_daily,
                    "Mkt-RF": res.beta_mkt,
                    "SMB": res.beta_smb,
                    "HML": res.beta_hml,
                },
                "t-stat": {
                    "alpha (const)": res.t_alpha,
                    "Mkt-RF": res.t_mkt,
                    "SMB": res.t_smb,
                    "HML": res.t_hml,
                },
                "p-value": {
                    "alpha (const)": res.p_alpha,
                    "Mkt-RF": res.p_mkt,
                    "SMB": res.p_smb,
                    "HML": res.p_hml,
                },
            }
        )

        st.markdown("### Coefficients")
        st.dataframe(coef_tbl, use_container_width=True)
        st.caption(
            "Displays the stock’s sensitivity to market, size, and value factors, highlighting the primary drivers of its return behavior."
        )

        pred = model.predict(sm.add_constant(merged[["Mkt-RF", "SMB", "HML"]]))
        actual = merged["Ri"] - merged["RF"]
        compare = pd.DataFrame({"Actual excess": actual, "Model fitted": pred}).dropna()

        st.markdown("### Actual vs fitted excess returns")
        st.line_chart(compare, height=300)
        st.caption(
            "Compares observed excess returns with model-predicted returns to assess how well the factor model explains day-to-day movements."
        )

        if show_debug:
            st.write("Merged regression frame (tail):")
            st.dataframe(merged.tail(10), use_container_width=True)

# ----------------------------
# Tab 3: Rolling Stability
# ----------------------------
with tabs[2]:
    st.subheader("Rolling FF3 stability (daily)")

    ticker_options = [t for t in tickers if t in rets_d.columns]
    pick2 = st.selectbox("Ticker for rolling analysis", options=ticker_options, index=0, key="rollpick")

    s2 = rets_d[pick2].dropna()
    roll = rolling_ff3_daily(s2.rename(pick2), ff3_d, window=rolling_window)

    if roll.empty:
        st.warning("Not enough data for rolling regression. Try an earlier start year or a shorter rolling window.")
    else:
        st.markdown("### Rolling factor exposures")
        st.line_chart(roll[["beta_mkt", "beta_smb", "beta_hml"]], height=320)
        st.caption("Shows how a stock’s factor exposures evolve over time using a rolling regression window.")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Rolling alpha (daily)")
            st.line_chart(roll[["alpha"]], height=260)
            st.caption(
                "Illustrates how unexplained returns change over time, helping distinguish persistent effects from short-term noise."
            )
        with c2:
            st.markdown("### Rolling R²")
            st.line_chart(roll[["r2"]], height=260)
            st.caption("Shows how the explanatory power of the factor model varies across different market environments.")

        st.caption(f"Rolling window = {rolling_window} trading days.")

        if show_debug:
            st.dataframe(roll.tail(20), use_container_width=True)

# ----------------------------
# Tab 4: Cross-ticker comparison
# ----------------------------
with tabs[3]:
    st.subheader("Cross-ticker FF3 comparison")

    results: List[FF3Result] = []
    for t in ticker_options:
        s = rets_d[t].dropna()
        r, m, _ = run_ff3_regression_daily(s.rename(t), ff3_d)
        if r is not None:
            results.append(r)

    if not results:
        st.warning("No regressions could be computed (insufficient data overlap). Try an earlier start year.")
    else:
        out = pd.DataFrame(
            [
                {
                    "Ticker": r.ticker,
                    "Alpha (ann)": r.alpha_annual,
                    "Beta MKT": r.beta_mkt,
                    "Beta SMB": r.beta_smb,
                    "Beta HML": r.beta_hml,
                    "R²": r.r2,
                    "Obs": r.n_obs,
                    "p(alpha)": r.p_alpha,
                    "p(MKT)": r.p_mkt,
                    "p(SMB)": r.p_smb,
                    "p(HML)": r.p_hml,
                }
                for r in results
            ]
        ).set_index("Ticker").sort_index()

        disp = out.copy()
        disp["Alpha (ann)"] = disp["Alpha (ann)"].map(lambda x: fmt_pct(x, 2))
        for c in ["Beta MKT", "Beta SMB", "Beta HML", "R²"]:
            disp[c] = disp[c].map(lambda x: fmt_num(x, 3))
        for c in ["p(alpha)", "p(MKT)", "p(SMB)", "p(HML)"]:
            disp[c] = disp[c].map(lambda x: fmt_num(x, 4))

        st.dataframe(disp, use_container_width=True)
        st.caption(
            "Compares factor exposures across selected stocks to identify similarities, differences, and potential risk concentration."
        )

        st.markdown("### Factor exposure heatmap")
        heat = out[["Beta MKT", "Beta SMB", "Beta HML"]].copy()
        st.dataframe(heat.style.background_gradient(axis=None), use_container_width=True)
        st.caption(
            "Provides a visual summary of factor loadings across stocks, making patterns and clustering of risk exposures easy to identify."
        )

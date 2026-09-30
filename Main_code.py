"""
Project 2: Volatility Modeling with GARCH(1,1)
See Project2_GARCH_Volatility_README.md for the full explanation, resume bullets and interview Q&A.
"""
import argparse
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(HERE, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)
TRADING_DAYS = 252

# ----------------------------------------------------------------------------
# Minimal Black-Scholes helpers (copied from Project 1 so this file is standalone)
# ----------------------------------------------------------------------------
def bs_price(S, K, T, r, sigma, option="call", q=0.0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option == "call":
        return S * np.exp(-q * T) * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    return K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * np.exp(-q * T) * stats.norm.cdf(-d1)


def greeks(S, K, T, r, sigma, option="call", q=0.0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf = stats.norm.pdf(d1)
    return {
        "delta": np.exp(-q * T) * stats.norm.cdf(d1),
        "gamma": np.exp(-q * T) * pdf / (S * sigma * np.sqrt(T)),
        "vega": S * np.exp(-q * T) * pdf * np.sqrt(T) / 100,
        "theta": (-S * pdf * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * stats.norm.cdf(d2)) / 365,
    }



# ----------------------------------------------------------------------------
# 1. DATA
# ----------------------------------------------------------------------------
def simulate_garch_prices(n=2500, seed=7):
    """Offline fallback: simulate a GARCH(1,1)-t process so the pipeline runs
    anywhere. NOT real data - use real Nifty data for your resume numbers."""
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 0.02, 0.09, 0.89
    r = np.zeros(n); h = np.zeros(n); h[0] = omega / (1 - alpha - beta)
    for t in range(1, n):
        h[t] = omega + alpha * r[t - 1] ** 2 + beta * h[t - 1]
        r[t] = np.sqrt(h[t]) * rng.standard_t(6) * np.sqrt(4 / 6)
    dates = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(17000 * np.exp(np.cumsum(r / 100)), index=dates, name="Close")


def load_prices(ticker, csv, synthetic, start="2015-01-01"):
    if synthetic:
        print("Using SYNTHETIC data (demo only).")
        return simulate_garch_prices(), False
    if csv:
        df = pd.read_csv(csv, parse_dates=["Date"], index_col="Date")
        return df["Close"].dropna().sort_index(), True
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
        s = df["Close"].squeeze().dropna()
        if len(s) < 500:
            raise RuntimeError("too little data")
        print(f"Downloaded {len(s)} days of {ticker} ({s.index[0].date()} -> {s.index[-1].date()})")
        return s, True
    except Exception as e:
        print(f"[warn] download failed ({e}). Falling back to SYNTHETIC data.")
        return simulate_garch_prices(), False


def load_vix(start="2015-01-01"):
    try:
        import yfinance as yf
        v = yf.download("^INDIAVIX", start=start, progress=False, auto_adjust=True)["Close"].squeeze().dropna()
        return v if len(v) > 100 else None
    except Exception:
        return None


# ----------------------------------------------------------------------------
# 2-3. STYLISED FACTS & ARCH EFFECTS
# ----------------------------------------------------------------------------
def stylised_facts(ret):
    print("\n" + "=" * 66 + "\nSTEP 1: STYLISED FACTS OF RETURNS (in % )\n" + "=" * 66)
    print(f"Mean daily return : {ret.mean():.4f}%")
    print(f"Daily std dev     : {ret.std():.4f}%  (annualised {ret.std()*np.sqrt(TRADING_DAYS):.2f}%)")
    print(f"Skewness          : {stats.skew(ret):.3f}")
    print(f"Excess kurtosis   : {stats.kurtosis(ret):.3f}   (0 for a normal -> fat tails if >> 0)")
    jb = stats.jarque_bera(ret)
    print(f"Jarque-Bera p-val : {jb.pvalue:.4g}   (<0.05 => NOT normal)")

    lb_r = acorr_ljungbox(ret, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
    lb_r2 = acorr_ljungbox(ret**2, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
    arch_lm = het_arch(ret - ret.mean(), nlags=10)[1]
    print(f"Ljung-Box(10) on returns     p = {lb_r:.4g}  (returns ~ unpredictable)")
    print(f"Ljung-Box(10) on returns^2   p = {lb_r2:.4g}  (<0.05 => volatility clusters)")
    print(f"Engle ARCH-LM(10)            p = {arch_lm:.4g}  (<0.05 => ARCH effects => GARCH is justified)")


# ----------------------------------------------------------------------------
# 4. FIT
# ----------------------------------------------------------------------------
def fit_garch(ret, dist="normal"):
    am = arch_model(ret, mean="Constant", vol="GARCH", p=1, q=1, dist=dist)
    return am.fit(disp="off")


def summarise(res, label):
    p = res.params
    omega, alpha, beta = p["omega"], p["alpha[1]"], p["beta[1]"]
    persistence = alpha + beta
    lr_var = omega / (1 - persistence) if persistence < 1 else np.nan
    half_life = np.log(0.5) / np.log(persistence) if 0 < persistence < 1 else np.nan
    print(f"\n--- {label} ---")
    print(res.summary().tables[2])
    print(f"omega={omega:.5f}  alpha={alpha:.4f}  beta={beta:.4f}")
    print(f"Persistence (alpha+beta) : {persistence:.4f}")
    print(f"Long-run daily vol       : {np.sqrt(lr_var):.4f}%  -> annualised {np.sqrt(lr_var*TRADING_DAYS):.2f}%")
    print(f"Half-life of a vol shock : {half_life:.1f} trading days")
    print(f"AIC = {res.aic:.1f}   BIC = {res.bic:.1f}   LogLik = {res.loglikelihood:.1f}")
    return dict(omega=omega, alpha=alpha, beta=beta, persistence=persistence, lr_var=lr_var)


def diagnostics(res):
    std_resid = res.std_resid.dropna()
    lb = acorr_ljungbox(std_resid**2, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
    print(f"\nResidual check: Ljung-Box(10) on standardised residuals^2 p = {lb:.4f} "
          f"-> {'no ARCH left: model adequate' if lb > 0.05 else 'some ARCH left: try GJR/EGARCH'}")


# ----------------------------------------------------------------------------
# 6. OUT-OF-SAMPLE
# ----------------------------------------------------------------------------
def out_of_sample(ret, train_frac=0.8):
    n = len(ret); split = int(n * train_frac)
    train = ret.iloc[:split]
    res_train = arch_model(train, mean="Constant", vol="GARCH", p=1, q=1, dist="normal").fit(disp="off")

    # Re-use TRAIN parameters on the full series -> genuine out-of-sample 1-day-ahead forecasts
    am_full = arch_model(ret, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
    res_fixed = am_full.fix(res_train.params)
    fc = res_fixed.forecast(start=ret.index[split], horizon=1, reindex=False)
    garch_var = fc.variance.iloc[:, 0]

    test = ret.iloc[split:]
    realised_proxy = (test - train.mean()) ** 2             # squared return = noisy variance proxy
    const_var = pd.Series(train.var(), index=test.index)    # baseline 1: constant vol
    # baseline 2: 21-day rolling historical variance, lagged 1 day (no look-ahead)
    roll_var = (ret ** 2).rolling(21).mean().shift(1).loc[test.index]

    garch_var = garch_var.reindex(test.index)
    def mse(f): return np.mean((f - realised_proxy) ** 2)
    def qlike(f): return np.mean(np.log(f) + realised_proxy / f)   # robust loss for vol forecasts

    table = pd.DataFrame({
        "MSE": [mse(garch_var), mse(roll_var), mse(const_var)],
        "QLIKE": [qlike(garch_var), qlike(roll_var), qlike(const_var)],
    }, index=["GARCH(1,1)", "Rolling 21d", "Constant vol"])
    print("\n" + "=" * 66 + f"\nSTEP 4: OUT-OF-SAMPLE ({len(test)} days, lower = better)\n" + "=" * 66)
    print(table.round(4))
    best = table["QLIKE"].idxmin()
    print(f"Best by QLIKE: {best}")

    plt.figure(figsize=(11, 4.5))
    plt.plot(test.index, np.sqrt(garch_var * TRADING_DAYS), label="GARCH forecast (ann. %)", lw=1.6)
    plt.plot(test.index, np.sqrt(roll_var * TRADING_DAYS), label="Rolling 21d vol (ann. %)", lw=1, alpha=0.8)
    plt.plot(test.index, np.sqrt((realised_proxy.rolling(5).mean()) * TRADING_DAYS),
             label="Realised vol, 5d (ann. %)", lw=0.8, color="grey")
    plt.axhline(np.sqrt(const_var.iloc[0] * TRADING_DAYS), color="r", ls="--", lw=0.8, label="Constant vol")
    plt.title("Out-of-sample 1-day-ahead volatility forecasts"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "oos_forecast.png"), dpi=130); plt.close()
    return table


# ----------------------------------------------------------------------------
# 7. MULTI-STEP FORECAST
# ----------------------------------------------------------------------------
def term_structure(res, horizon=60):
    fc = res.forecast(horizon=horizon, reindex=False)
    var_path = fc.variance.iloc[-1].values
    ann_vol_path = np.sqrt(var_path * TRADING_DAYS)
    # Average vol over the next h days is what matters for an h-day option
    avg_vol = np.sqrt(np.cumsum(var_path) / np.arange(1, horizon + 1) * TRADING_DAYS)
    plt.figure(figsize=(7, 4.5))
    plt.plot(range(1, horizon + 1), ann_vol_path, label="Daily forecast vol (ann. %)")
    plt.plot(range(1, horizon + 1), avg_vol, label="Average vol to horizon (ann. %)")
    lr = np.sqrt(res.params["omega"] / (1 - res.params["alpha[1]"] - res.params["beta[1]"]) * TRADING_DAYS)
    plt.axhline(lr, color="r", ls="--", label=f"Long-run vol {lr:.1f}%")
    plt.xlabel("Days ahead"); plt.title("GARCH vol term structure (mean reversion)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "term_structure.png"), dpi=130); plt.close()
    return avg_vol


# ----------------------------------------------------------------------------
# 8. TIE-IN WITH PROJECT 1
# ----------------------------------------------------------------------------
def tie_in_with_black_scholes(res, avg_vol, prices, vix, real_data):
    print("\n" + "=" * 66 + "\nSTEP 5: LINK TO BLACK-SCHOLES (Project 1)\n" + "=" * 66)
    horizon = 21                                   # ~1 month option
    garch_vol_21 = avg_vol[horizon - 1]            # in % annualised
    print(f"GARCH forecast avg vol over next {horizon} days: {garch_vol_21:.2f}% (annualised)")

    # compare with India VIX (market-implied 30-day vol for Nifty)
    if vix is not None and real_data:
        vix_now = float(vix.iloc[-1])
        print(f"India VIX (market implied 30d vol)         : {vix_now:.2f}%")
        gap = vix_now - garch_vol_21
        view = "options look EXPENSIVE vs GARCH (implied > forecast)" if gap > 0 else \
               "options look CHEAP vs GARCH (implied < forecast)"
        print(f"Implied - GARCH = {gap:+.2f} vol pts -> {view}")

        # plot GARCH conditional vol vs VIX over time
        cv = res.conditional_volatility * np.sqrt(TRADING_DAYS)
        both = pd.concat([cv.rename("GARCH cond. vol"), vix.rename("India VIX")], axis=1).dropna()
        if len(both) > 100:
            print(f"Correlation GARCH vol vs VIX: {both.corr().iloc[0, 1]:.3f}")
            both.plot(figsize=(11, 4.5), lw=1, title="GARCH conditional vol vs India VIX (annualised %)")
            plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, "garch_vs_vix.png"), dpi=130); plt.close()
    else:
        print("(India VIX not available / synthetic data -> skipping VIX comparison)")

    # price an ATM 1-month call with GARCH vol using Project 1's pricer
    try:
        pass  # bs_price and greeks are defined above in this file
        S = float(prices.iloc[-1]); K = round(S / 50) * 50; T = horizon / TRADING_DAYS; r = 0.065
        sig = garch_vol_21 / 100
        c = bs_price(S, K, T, r, sig, "call"); g = greeks(S, K, T, r, sig, "call")
        print(f"\nATM call (S={S:.0f}, K={K:.0f}, 21d) priced at GARCH vol {sig:.1%}: {c:.2f}")
        print(f"  Delta={g['delta']:.3f}  Gamma={g['gamma']:.5f}  Vega={g['vega']:.2f}  Theta/day={g['theta']:.2f}")
        if vix is not None and real_data:
            c_mkt = bs_price(S, K, T, r, float(vix.iloc[-1]) / 100, "call")
            print(f"Same option at India VIX vol: {c_mkt:.2f}  (difference {c_mkt - c:+.2f} points)")
    except ImportError:
        print("skipping option pricing tie-in.")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="^NSEI")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()

    prices, real = load_prices(a.ticker, a.csv, a.synthetic)
    ret = (np.log(prices).diff().dropna() * 100).rename("ret")      # % log returns

    # Plot returns
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(ret, lw=0.6); ax[0].set_title("Daily log returns (%) - note vol clustering")
    ax[1].hist(ret, bins=80, density=True, alpha=0.6, label="Returns")
    x = np.linspace(ret.min(), ret.max(), 300)
    ax[1].plot(x, stats.norm.pdf(x, ret.mean(), ret.std()), "r", label="Normal")
    ax[1].set_title("Return distribution: fat tails"); ax[1].legend()
    plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, "returns_overview.png"), dpi=130); plt.close()

    stylised_facts(ret)

    print("\n" + "=" * 66 + "\nSTEP 2: FIT GARCH(1,1)\n" + "=" * 66)
    res_n = fit_garch(ret, "normal"); summarise(res_n, "GARCH(1,1) - Normal errors")
    res_t = fit_garch(ret, "t");      summarise(res_t, "GARCH(1,1) - Student-t errors")
    better = "Student-t" if res_t.aic < res_n.aic else "Normal"
    print(f"\nLower AIC -> {better} errors fit better.")

    print("\n" + "=" * 66 + "\nSTEP 3: DIAGNOSTICS\n" + "=" * 66)
    diagnostics(res_t if better == "Student-t" else res_n)

    cond = res_t.conditional_volatility * np.sqrt(TRADING_DAYS)
    plt.figure(figsize=(11, 4.5)); plt.plot(cond, lw=0.9)
    plt.title("GARCH(1,1) conditional volatility (annualised %)"); plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "conditional_vol.png"), dpi=130); plt.close()

    out_of_sample(ret)
    avg_vol = term_structure(res_n)
    vix = load_vix() if real else None
    tie_in_with_black_scholes(res_n, avg_vol, prices, vix, real)
    print(f"\nPlots saved in: {PLOT_DIR}")


if __name__ == "__main__":
    main()
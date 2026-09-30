# Project 2 --- Volatility Modeling with GARCH(1,1)

A quantitative volatility-modeling project that forecasts **Nifty 50
volatility using GARCH(1,1)**, tests the model against simple
out-of-sample baselines, compares forecast volatility with **India
VIX**, and connects the forecast to **Black-Scholes option pricing**.

> **Quant Finance / Volatility Modeling Project**\
> **Tech:** Python, NumPy, Pandas, SciPy, Statsmodels, ARCH, Matplotlib,
> yfinance

------------------------------------------------------------------------

## 📁 Project Files

``` text
Project2_GARCH_Volatility/
│
├── Project2_GARCH_Volatility_code.py
├── Project2_GARCH_Volatility_README.md
└── plots/
    ├── returns_overview.png
    ├── conditional_vol.png
    ├── oos_forecast.png
    ├── term_structure.png
    └── garch_vs_vix.png
```

------------------------------------------------------------------------

# 🚀 How to Run

Install the required packages:

``` bash
pip install numpy pandas scipy matplotlib statsmodels arch yfinance
```

### Run with real NIFTY data

``` bash
python Project2_GARCH_Volatility_code.py
```

### Run the offline synthetic-data demo

``` bash
python Project2_GARCH_Volatility_code.py --synthetic
```

The real-data version uses NIFTY 50 data (`^NSEI`) obtained through
`yfinance`.

Generated plots are saved automatically inside the `plots/` folder.

------------------------------------------------------------------------

# 1. Problem & Motivation

Black-Scholes requires a volatility input, but market volatility is
**not constant**.

Volatility tends to cluster:

> Calm periods tend to follow calm periods, while turbulent periods tend
> to follow turbulent periods.

This project models that time-varying volatility using **GARCH(1,1)**.

For an options trader, an important comparison is:

``` text
Forecast Volatility
        vs
Implied Volatility
```

The difference can provide information about how the market is pricing
future volatility.

However, a volatility gap by itself is **not a free-lunch trading
signal** because implied volatility contains risk premia and other
market effects.

------------------------------------------------------------------------

# 2. Stylised Facts of Financial Returns

The project examines several well-known characteristics of financial
returns:

1.  Returns are close to unpredictable in their levels.
2.  Returns have **fat tails**.
3.  Squared returns are autocorrelated, indicating **volatility
    clustering**.
4.  Volatility tends to be **mean-reverting**.

These characteristics motivate the use of a conditional-volatility model
such as GARCH.

------------------------------------------------------------------------

# 3. GARCH(1,1) Theory

The return process is written as:

$$
r_t=\mu+\epsilon_t
$$

with:

$$
\epsilon_t=\sigma_t z_t
$$

where $z_t$ can follow a Normal or Student-t distribution.

The GARCH(1,1) variance equation is:

$$
\sigma_t^2
=
\omega
+
\alpha\epsilon_{t-1}^2
+
\beta\sigma_{t-1}^2
$$

### Parameter Interpretation

  Parameter        Meaning
  ---------------- -------------------------------------------------------
  $\omega$         Baseline variance component
  $\alpha$         Reaction to the previous period's shock
  $\beta$          Persistence / memory of previous conditional variance
  $\alpha+\beta$   Overall volatility persistence

A value of $\alpha+\beta$ close to 1 indicates that volatility shocks
decay slowly.

------------------------------------------------------------------------

# 4. Long-Run Variance & Half-Life

For a stationary GARCH(1,1) process:

$$
\alpha+\beta<1
$$

The long-run variance is:

$$
\sigma^2_{LR}
=
\frac{\omega}{1-\alpha-\beta}
$$

The half-life of a volatility shock is:

$$
Half\text{-}life
=
\frac{\ln(0.5)}
{\ln(\alpha+\beta)}
$$

A longer half-life means that a volatility shock takes longer to decay.

------------------------------------------------------------------------

# 5. Multi-Step Volatility Forecast

The $h$-step conditional variance forecast is:

$$
E[\sigma_{t+h}^2]
=
\sigma^2_{LR}
+
(\alpha+\beta)^h
\left(
\sigma_t^2-\sigma^2_{LR}
\right)
$$

As $h$ increases:

$$
(\alpha+\beta)^h\rightarrow0
$$

and the forecast gradually approaches the long-run variance.

This demonstrates the **mean-reverting nature of GARCH volatility**.

------------------------------------------------------------------------

# 6. Project Pipeline

``` text
NIFTY 50 Daily Prices
        ↓
Log Returns
        ↓
Stylised-Fact Tests
        ↓
ARCH Effects
        ↓
GARCH(1,1)
        ↓
Normal vs Student-t Errors
        ↓
Residual Diagnostics
        ↓
Out-of-Sample Forecast
        ↓
Compare with Baselines
        ↓
GARCH Term Structure
        ↓
Compare with India VIX
        ↓
Black-Scholes Option Pricing
```

------------------------------------------------------------------------

# 7. What the Code Does

  -----------------------------------------------------------------------
  Step                    What happens            Why
  ----------------------- ----------------------- -----------------------
  **1. Data**             Downloads NIFTY         Log returns are
                          (`^NSEI`) daily closes  additive through time
                          and computes log        
                          returns in %            

  **2. Stylised facts**   Calculates kurtosis,    Tests whether GARCH is
                          Jarque-Bera, Ljung-Box  justified
                          on returns and squared  
                          returns, and Engle      
                          ARCH-LM                 

  **3. Model fitting**    Fits GARCH(1,1) with    Selects the better
                          Normal and Student-t    error specification
                          errors and compares AIC 

  **4. Diagnostics**      Runs Ljung-Box on       Checks whether
                          squared standardized    volatility clustering
                          residuals               remains

  **5. Out-of-sample      Uses the first 80% for  Avoids look-ahead bias
  testing**               estimation and the      
                          final 20% for           
                          forecasting             

  **6. Baselines**        Compares against        Tests whether GARCH
                          constant volatility and adds predictive value
                          21-day rolling          
                          volatility              

  **7. Term structure**   Produces a 60-day       Shows mean reversion
                          volatility forecast     over the forecast
                          curve                   horizon

  **8. Options tie-in**   Compares GARCH          Connects volatility
                          volatility with India   modeling to options
                          VIX and prices an ATM   pricing
                          21-day call at the      
                          GARCH forecast          
                          volatility              
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 8. Statistical Diagnostics

Before fitting GARCH, the project tests whether the data exhibits
characteristics that justify conditional-volatility modeling.

### Tests used

-   Kurtosis
-   Jarque-Bera test
-   Ljung-Box test on returns
-   Ljung-Box test on squared returns
-   Engle ARCH-LM test

The key idea is:

``` text
Returns may show little serial correlation
                ↓
But squared returns can be autocorrelated
                ↓
Volatility is clustered
                ↓
GARCH becomes appropriate
```

------------------------------------------------------------------------

# 9. Normal vs Student-t Errors

Two conditional distributions are fitted:

### Normal

$$
z_t\sim N(0,1)
$$

### Student-t

A Student-t distribution allows for heavier tails than the Normal
distribution.

This matters because financial returns can experience unusually large
movements more frequently than a Normal distribution suggests.

The project compares the two specifications using **AIC**.

------------------------------------------------------------------------

# 10. Maximum-Likelihood Estimation

GARCH parameters are estimated using **maximum likelihood**.

For the Normal case, the log-likelihood is based on terms of the form:

$$
-\frac{1}{2}
\left[
\log(\sigma_t^2)
+
\frac{\epsilon_t^2}{\sigma_t^2}
\right]
$$

The estimation procedure searches for parameter values that maximize the
likelihood of observing the return series.

The `arch` Python library performs the numerical optimization.

------------------------------------------------------------------------

# 11. Residual Diagnostics

After fitting the model, standardized residuals are examined.

The project applies a **Ljung-Box test to squared standardized
residuals**.

The objective is to check whether significant conditional-volatility
structure remains after fitting the model.

Conceptually:

``` text
Raw returns
    ↓
Volatility clustering
    ↓
Fit GARCH
    ↓
Standardized residuals
    ↓
Test residual²
    ↓
Little remaining autocorrelation?
    ↓
Model captured much of the clustering
```

------------------------------------------------------------------------

# 12. Out-of-Sample Testing

A major focus of this project is avoiding **look-ahead bias**.

The dataset is split approximately:

``` text
80% Training
     ↓
Estimate GARCH parameters
     ↓
Freeze parameters
     ↓
20% Test
     ↓
Generate 1-day-ahead forecasts
```

The model is therefore evaluated on observations that were not used to
estimate its parameters.

The baselines also use only information available up to the forecasting
date.

------------------------------------------------------------------------

# 13. Baseline Models

GARCH is compared against two simple alternatives:

### Constant Volatility

Uses a constant historical volatility estimate.

### 21-Day Rolling Volatility

Uses the volatility calculated over the previous 21 trading days.

The purpose is important:

> A complex model should be compared against simple alternatives rather
> than assumed to be useful merely because it is sophisticated.

------------------------------------------------------------------------

# 14. Evaluation Metrics

The project uses:

### Mean Squared Error (MSE)

The forecast is compared with squared returns, which provide a noisy
proxy for realised variance.

### QLIKE

The project also uses:

$$
QLIKE
=
\log(\hat{\sigma}^2_t)
+
\frac{r_t^2}{\hat{\sigma}^2_t}
$$

Lower values indicate better volatility forecasts.

QLIKE is useful because the true conditional variance is not directly
observable.

------------------------------------------------------------------------

# 15. Volatility Term Structure

The project produces a 60-day volatility forecast curve.

Conceptually:

``` text
Current conditional variance
            ↓
     1-day forecast
            ↓
     2-day forecast
            ↓
          ...
            ↓
     60-day forecast
            ↓
Long-run variance
```

The forecast gradually approaches the long-run level because of the
mean-reverting structure of GARCH.

For options, the relevant quantity is the expected volatility over the
option's life rather than simply today's volatility.

------------------------------------------------------------------------

# 16. GARCH vs India VIX

The project compares model-forecast volatility with **India VIX**.

Conceptually:

``` text
GARCH
  ↓
Forecast volatility
       ↘
        Compare
       ↗
India VIX
  ↓
Market-implied volatility
```

A difference between the two can indicate that the market's implied
volatility and the model's forecast disagree.

However:

> A gap does not automatically imply arbitrage.

Forecast volatility is a physical-measure estimate, while implied
volatility incorporates the market price of volatility risk and other
factors.

------------------------------------------------------------------------

# 17. Black-Scholes Integration

The project connects directly to Project 1.

The workflow is:

``` text
GARCH Forecast
      ↓
Expected Volatility
      ↓
Black-Scholes σ
      ↓
ATM 21-Day Call Price
```

The same option can be examined using different volatility assumptions.

For example:

``` text
GARCH Forecast Vol
        ↓
Black-Scholes Price

India VIX / Implied Vol
        ↓
Black-Scholes Price
```

This creates a direct connection between:

**Time-Series Volatility Modeling → Options Pricing**

------------------------------------------------------------------------

# 18. Results

The values below are from the built-in **synthetic demonstration data**
and are included only to demonstrate that the pipeline works.

> **Important:** Replace these values with results from your actual
> NIFTY data before using them on your resume.

The synthetic demonstration indicates:

-   Significant ARCH effects, supporting the use of GARCH.
-   Persistence around 0.98.
-   Shock half-life around 35--40 days.
-   Student-t errors outperforming Normal errors on AIC.
-   Residual ARCH test with $p>0.05$.
-   GARCH outperforming the simple baselines on the demonstrated
    out-of-sample QLIKE comparison.

### Real-data Results

After running the project on real NIFTY data, replace the placeholders
in the resume bullets with:

``` text
Persistence = [XX]
Half-life = [XX] days
Test observations = [XX]
QLIKE improvement vs constant volatility = [XX]%
QLIKE improvement vs rolling volatility = [XX]%
```

Only report numbers produced by the real-data run.

------------------------------------------------------------------------

# 19. Limitations

### 1. Symmetric Volatility Response

Standard GARCH treats positive and negative shocks symmetrically.

Equity markets can react differently to negative and positive returns.

Possible extensions:

-   GJR-GARCH
-   EGARCH

### 2. Noisy Volatility Proxy

Squared daily returns are a noisy proxy for realised variance.

Intraday realised volatility can provide a richer target.

### 3. Regime Changes

GARCH parameters can change across market regimes.

Examples include:

-   Calm markets
-   Crisis periods
-   High-volatility regimes

Periodic refitting or regime-aware models can address this.

### 4. Physical vs Implied Volatility

GARCH forecasts are physical-measure estimates.

Option-implied volatility incorporates a volatility risk premium.

Therefore:

``` text
Implied Volatility > Forecast Volatility
```

does not automatically mean that options are overpriced.

------------------------------------------------------------------------

# 20. Possible Extensions

The project can be extended with:

-   GJR-GARCH
-   EGARCH
-   Heston stochastic volatility
-   Realised-volatility models
-   Volatility-surface construction
-   Option-chain calibration
-   VaR and Expected Shortfall
-   Volatility-risk-premium analysis
-   Delta-hedging P&L
-   Transaction costs and bid-ask spreads

------------------------------------------------------------------------

# 21. Generated Plots

Running the project produces:

  Plot                     Description
  ------------------------ -----------------------------------------
  `returns_overview.png`   NIFTY returns and distribution overview
  `conditional_vol.png`    Conditional volatility from GARCH
  `oos_forecast.png`       Out-of-sample volatility forecast
  `term_structure.png`     Multi-day volatility forecast
  `garch_vs_vix.png`       GARCH forecast volatility vs India VIX

The exact set of plots depends on whether the script is run using real
or synthetic data.

------------------------------------------------------------------------

# 22. Resume Bullets

### Volatility Modeling with GARCH(1,1)

**Python, ARCH, Statsmodels**

-   Modeled NIFTY 50 daily returns using **GARCH(1,1)** with Normal and
    Student-t errors; tested ARCH effects using Engle LM and Ljung-Box
    diagnostics and estimated volatility persistence and shock
    half-life.
-   Evaluated **1-day-ahead out-of-sample volatility forecasts** against
    constant-volatility and 21-day rolling baselines using MSE and
    QLIKE.
-   Compared GARCH forecast volatility with **India VIX** and linked
    forecast volatility to Black-Scholes pricing of ATM options to
    analyze differences between model-based and market-implied
    volatility.

> Replace persistence, half-life, test-period and performance values
> with results from your actual NIFTY run before submitting the resume.

------------------------------------------------------------------------

# 23. Interview Questions & Answers

## 1. What is volatility clustering?

Large price movements tend to be followed by large movements, while
small movements tend to be followed by small movements.

Returns themselves may have little serial correlation, but squared
returns often show autocorrelation.

That behavior is known as **volatility clustering**.

------------------------------------------------------------------------

## 2. Write the GARCH(1,1) equation and explain each term.

$$
\sigma_t^2
=
\omega
+
\alpha\epsilon_{t-1}^2
+
\beta\sigma_{t-1}^2
$$

-   $\omega$ = baseline variance.
-   $\alpha$ = response to the previous shock.
-   $\beta$ = persistence of previous variance.
-   $\alpha+\beta$ = overall persistence.

------------------------------------------------------------------------

## 3. What does $\alpha+\beta$ mean?

It measures volatility persistence.

If:

$$
\alpha+\beta<1
$$

the process is stationary and mean-reverting.

As $\alpha+\beta$ approaches 1, volatility shocks decay more slowly.

If:

$$
\alpha+\beta=1
$$

the model becomes an IGARCH-type case with no finite long-run variance
under the usual formulation.

------------------------------------------------------------------------

## 4. Why GARCH instead of constant or rolling volatility?

Constant volatility ignores volatility clustering.

Rolling volatility uses a fixed window and treats observations within
that window similarly.

GARCH provides:

-   Dynamic volatility.
-   Decaying memory.
-   A formal multi-step forecast.
-   Mean reversion toward long-run variance.

------------------------------------------------------------------------

## 5. How is GARCH estimated?

The parameters are estimated using **maximum likelihood**.

For Normal errors, the likelihood is based on the conditional variance
generated by:

$$
\sigma_t^2
=
\omega+\alpha\epsilon_{t-1}^2+\beta\sigma_{t-1}^2
$$

Numerical optimization is used to find the parameters that maximize the
likelihood.

------------------------------------------------------------------------

## 6. Why use Student-t errors?

Financial returns often have heavier tails than a Normal distribution.

A Student-t distribution allows the model to represent more extreme
observations.

The project compares Normal and Student-t specifications using AIC.

------------------------------------------------------------------------

## 7. How did you check whether the model was adequate?

The project uses Ljung-Box diagnostics on squared standardized
residuals.

If significant autocorrelation remains, the model may not have captured
all of the volatility clustering.

------------------------------------------------------------------------

## 8. How did you avoid look-ahead bias?

The data is divided into approximately:

``` text
80% training
20% testing
```

GARCH parameters are estimated only using the training period.

The parameters are then frozen while producing forecasts over the test
period.

This prevents future observations from influencing the model parameters
used for earlier forecasts.

------------------------------------------------------------------------

## 9. Why QLIKE?

The true conditional variance is unobservable.

Squared returns are used as a noisy proxy for realised variance.

QLIKE is a useful volatility-forecast evaluation metric and is less
sensitive to extreme observations than directly relying on squared-error
comparisons alone.

Lower QLIKE indicates a better forecast under the chosen evaluation
setup.

------------------------------------------------------------------------

## 10. What are the weaknesses of GARCH(1,1)?

Main weaknesses:

-   Symmetric response to positive and negative shocks.
-   Dependence on the chosen volatility proxy.
-   Parameters can change across regimes.
-   Standard GARCH may not capture jumps well.

Possible alternatives include:

-   GJR-GARCH
-   EGARCH
-   Stochastic-volatility models

------------------------------------------------------------------------

## 11. What does a gap between GARCH volatility and implied volatility mean?

GARCH produces a model-based forecast of future volatility.

Implied volatility is extracted from option prices and contains the
market's pricing of volatility risk.

Therefore, a difference between them can reflect:

-   Volatility risk premium.
-   Model misspecification.
-   Market expectations.
-   Supply and demand for options.
-   Differences between physical and risk-neutral measures.

A gap alone is not sufficient to conclude that an option is mispriced.

------------------------------------------------------------------------

## 12. How do Project 1 and Project 2 connect?

Project 2 forecasts volatility using GARCH.

Project 1 uses volatility as an input to Black-Scholes.

The connection is:

``` text
NIFTY Returns
     ↓
GARCH(1,1)
     ↓
Forecast Volatility
     ↓
Black-Scholes
     ↓
Option Price
     ↓
Compare with Market / Implied Volatility
```

This combines **time-series modeling with derivatives pricing**.

------------------------------------------------------------------------

# 24. Formula Sheet

### GARCH(1,1)

$$
\sigma_t^2
=
\omega
+
\alpha\epsilon_{t-1}^2
+
\beta\sigma_{t-1}^2
$$

### Long-Run Variance

$$
\sigma^2_{LR}
=
\frac{\omega}{1-\alpha-\beta}
$$

### Shock Half-Life

$$
Half\text{-}life
=
\frac{\ln(0.5)}
{\ln(\alpha+\beta)}
$$

### Multi-Step Forecast

$$
E[\sigma_{t+h}^2]
=
\sigma^2_{LR}
+
(\alpha+\beta)^h
(\sigma_t^2-\sigma^2_{LR})
$$

### Annualised Volatility

$$
\sigma_{annual}
=
\sigma_{daily}\sqrt{252}
$$

------------------------------------------------------------------------

# 25. Key Quant Concepts Demonstrated

``` text
Financial Time Series
        ↓
Log Returns
        ↓
Statistical Diagnostics
        ↓
ARCH Effects
        ↓
GARCH(1,1)
        ↓
Maximum Likelihood
        ↓
Volatility Forecasting
        ↓
Out-of-Sample Validation
        ↓
India VIX
        ↓
Black-Scholes
        ↓
Options Analysis
```

Together with Project 1, this project demonstrates a coherent
quantitative-finance workflow:

``` text
             MARKET DATA
                  │
        ┌─────────┴─────────┐
        ↓                   ↓
  VOLATILITY MODEL      OPTIONS MODEL
        │                   │
     GARCH(1,1)        Black-Scholes
        │                   │
        ↓                   ↓
 Forecast Vol            Greeks
        │               Implied Vol
        │               Monte Carlo
        │                   │
        └─────────┬─────────┘
                  ↓
          OPTIONS ANALYSIS
                  ↓
          HEDGING / RISK
```

------------------------------------------------------------------------

## Author

**Shreyash Gharat**

M.Sc. Mathematics and Scientific Computing\
National Institute of Technology, Warangal

GitHub: `github.com/shreyashgharat848`

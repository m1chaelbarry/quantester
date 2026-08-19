# What does Carver prescribe for 256 business days versus measured frequency?

Ticket: [What does Carver prescribe for 256 business days versus measured frequency?](../issues/03-carver-256.md).
This note is the **Carver fact**, not a Quantester product default. That default is [What is the canonical periods-per-year and cash day-count policy?](../issues/06-periods-per-year.md).

## Verdict

Carver presents **256 business days / \(\sqrt{256}=16\)** as a **convenience convention for daily business-day series**, so the vol/Sharpe scalar is exactly 16. He does **not** present it as a universal constant that applies to every bar clock. He does **not** prescribe replacing 256 with a measured count of the series (US daily equities \(\approx 252\), median \(\Delta t\), NYSE session hours).

When the return frequency is not daily business days, his own engine switches to **other hardcoded year-counts** (weeks, months, calendar days, hours), still not to an index-inferred \(N_T\). He knows 256 overstates the real calendar (~252 days / ~21 days per month) and keeps 256 anyway.

## Source availability

**Robert Carver, *Systematic Trading* (the book) is not in this workspace.** No PDF, extract, or notebook dump of the relevant pages was found. This note therefore cannot quote the book verbatim.

**First-party substitutes used instead** (Carver’s own blog, and the open-source engine plus book-companion examples he published):

1. Carver, “R squared and Sharpe Ratio,” *This Blog is Systematic*, 18 Nov 2025.  
   <https://qoppac.blogspot.com/2025/11/r-squared-and-sharpe-ratio.html>
2. Carver, “Wordle (TM) and the one simple hack you need to pass funded trader challenges,” *This Blog is Systematic*, 5 Nov 2025.  
   <https://qoppac.blogspot.com/2025/11/wordle-tm-and-one-simple-hack-you-need.html>
3. `pysystemtrade` `syscore/dateutils.py` (`BUSINESS_DAYS_IN_YEAR = 256.0`, `from_frequency_to_times_per_year`) and `systems/accounts/curves/account_curve.py` (`vol_scalar`, `returns_scalar`). Develop branch, pst-group.  
   <https://github.com/pst-group/pysystemtrade>
4. `systematictradingexamples/common.py` and `optimisation.py` — header: “As in chapters 3 and 4 of *Systematic Trading*.”  
   <https://github.com/robcarver17/systematictradingexamples>

**Not used as primary for Carver’s words:** [`3rd Cross Reference Synthesis.md`](../../../3rd%20Cross%20Reference%20Synthesis.md) §4.2 (and the per-book audit in [`3rd Cross Reference.md`](../../../3rd%20Cross%20Reference.md)). Those files assert a Carver “standard” of 256; they are the question, not the evidence.

**Secondary, labelled as such:** Mike Rawson’s 7 Circles reading notes of *Systematic Trading* (2020). They are a third-party paraphrase of the book, used only as a consistency check that the 256/16 convention and the “usually 252” aside sit in ST itself.

## What Carver actually does with 256

### Daily business-day year = 256 so the root is 16

Carver’s engine defines:

```
CALENDAR_DAYS_IN_YEAR = 365.25
BUSINESS_DAYS_IN_YEAR = 256.0
ROOT_BDAYS_INYEAR     = BUSINESS_DAYS_IN_YEAR ** 0.5   # 16.0
```

(`pysystemtrade` `syscore/dateutils.py`.) The doctest for `from_frequency_to_times_per_year(Frequency.BDay)` returns `256.0`.

The book-companion examples hard-code the same pair and apply it to daily Sharpe and vol:

```
DAYS_IN_YEAR = 256.0
ROOT_DAYS_IN_YEAR = DAYS_IN_YEAR ** 0.5
# annualised_vol  = daily_std * ROOT_DAYS_IN_YEAR
# sharpe          = ROOT_DAYS_IN_YEAR * mean / std   # "assumes daily returns"
```

(`systematictradingexamples/common.py`.) `optimisation.py` equalises daily vols with `default_vol / 16.0`.

On the blog he writes the identity in the other direction:

> “Anyway under LAM at an annual holding period an R squared of 0.01 equates to an IC/SR of 0.10. Under LAM we'd expect the same R squared to result in a \(\sqrt{256} = 16\), SR of 1.6 at a daily holding period.”
>
> “So here is one, assuming 256 business days in a year: The SR for N days holding period is equal to \(16 \times \sqrt{\text{R squared} / N}\).”

(R-squared post, Nov 2025.)

Vol targeting uses the same 16: daily cash vol target = annual cash vol target / `ROOT_BDAYS_INYEAR` (`systems/positionsizing.py` `get_daily_cash_vol_target`). Annualised instrument vol is `daily_vol * ROOT_BDAYS_INYEAR` (`systems/rawdata.py` `annualised_returns_volatility`; `systems/portfolio.py` `annualised_percentage_vol`). Daily SR is `ROOT_BDAYS_INYEAR * daily_return / daily_std` (`sysquant/returns.py` `annual_SR_from_daily_returns`). Cost drag is de-annualised by the same 16 (`daily_SR_cost = dict_of_SR_costs[column] / ROOT_BDAYS_INYEAR`).

That is the whole daily convention: **pick 256 so \(\sqrt{P}=16\)**, then use 16 everywhere a daily quantity is scaled to (or from) a year.

### He knows 256 is not the measured calendar

Nov 2025, modelling a 30-trading-day prop challenge:

> “256 business days a year, 22 business days a month (**it's actually more like 21**, but again this higher figure will make the prop firm look good)”

(Wordle / prop-challenge post; emphasis added.) That is Carver in his own voice: 256/22 is a **rounded-up modelling year**, not a count of exchange sessions.

The 7 Circles ST reading notes (secondary) report the same aside inside the book: annual SD is “around 16 times daily SD. This assumes 256 trading days in a year, **whereas there are usually 252**”; and for vol targeting, “If we assume 256 trading days in a year (there are 252 in the UK, I think, including two half-days) then we have a multiplier of 16.” ([ST 1 – Theory](https://the7circles.uk/systematic-trading-1-theory/); [ST 4 – Volatility targeting](https://the7circles.uk/systematic-trading-4-volatility-targeting-and-position-sizing/).) That matches the first-party code/blog; it is **not** an independent Carver quote.

## Not a universal constant

256 is the **business-day daily** bucket. `from_frequency_to_times_per_year` maps other clocks to **different** hardcoded year-counts (`syscore/dateutils.py`):

| Frequency | Year-count used | Vol scalar |
|---|---|---|
| `BDay` | `BUSINESS_DAYS_IN_YEAR` = **256** | 16 |
| `Day` (calendar, weekends in) | `CALENDAR_DAYS_IN_YEAR` = **365.25** | \(\sqrt{365.25}\) |
| `Week` | `365.25 / 7` | \(\sqrt{365.25/7}\) |
| `Month` | **12** | \(\sqrt{12}\) |
| `Year` | **1** | 1 |
| `Hour` | `24 * 256` = **6144** | \(\sqrt{6144}\) |
| 15-min / 5-min / second | fractions of `MINUTES_PER_YEAR` / `SECONDS_IN_YEAR` (calendar 365.25 × 24h) | \(\sqrt{\cdot}\) |

`accountCurve` then annualises with those lookups, not with 256 on every series:

- `returns_scalar = from_frequency_to_times_per_year(self.frequency)`
- `vol_scalar = times_per_year ** 0.5`
- `ann_mean = total / (len / returns_scalar)`
- `ann_std = period_std * vol_scalar`
- `sharpe = ann_mean / ann_std`

(`systems/accounts/curves/account_curve.py`.) Switching `.daily` / `.weekly` / `.monthly` / `.annual` changes the Frequency enum, which changes the scalar.

The 2015 companion `account_curve.new_freq` is even blunter: resampling to weekly/monthly “**will break certain things (eg Sharpe) so be careful**,” and `.sharpe()` “**assumes daily returns**” (`systematictradingexamples/common.py`). Hourly prices exist (`hourly_returns`), but **annualised vol is still computed from daily returns × 16**, not from hourly std × \(\sqrt{256}\) (`systems/rawdata.py`).

So 256 is **not** “one \(P\) for all assets and all clocks.” It is the daily business-day year. Applying 256 (or \(\sqrt{256}\)) to hourly, weekly, or 24/7 bars is **not** what Carver’s code does.

## Convenience for scalar 16, not a measured \(N_T\)

Three facts sit together:

1. The arithmetic identity \(\sqrt{256}=16\) is used as a closed-form daily/annual bridge (blog; `ROOT_BDAYS_INYEAR`; companion `/16.0`).
2. Carver states the real month is “more like 21” business days and still uses 22/256 (Wordle post).
3. Nothing in the first-party sources **replaces** 256 with a count taken from the DatetimeIndex (median \(\Delta t\), NYSE session length, or “how many bars were in this sample”).

That is a **convenience default that does not yield to a measured daily calendar**. The ~1.6% gap \(\sqrt{256}/\sqrt{252}\approx 1.0079\) is treated as not worth tracking. It is **not** a theorem that 256 is the true number of US/UK sessions.

## Does he yield to actual periods per year?

**Yes, at the level of frequency class. No, at the level of measuring the series.**

- **Yields to frequency class:** daily business-day vs calendar-day vs week vs month vs hour vs minute. Each class has a **pre-declared** \(P\). Account-curve Sharpe follows that \(P\).
- **Does not yield to measured \(N_T\):** he does not infer 252 from US equity holidays, 1638 from 6.5 NYSE hours, or 8760 from crypto. Hourly \(P\) in `pysystemtrade` is **24 × 256 = 6144** (24-hour clock on a 256-day year), which is neither NYSE hours nor 24/7 calendar hours (that would be \(24\times 365.25\)).

If the question is “should a daily US equity book use 252 because that is what the calendar actually is?”, Carver’s published practice is **no: keep 256/16**. If the question is “should hourly bars be annualised with the daily 256?”, Carver’s published practice is **no: either collapse to daily then ×16, or use the hourly lookup (24×256), not 256 itself**.

## Brief Chan-style contrast (context only)

Ernest Chan, on his own blog, annualises **daily** vol/Sharpe with \(\sqrt{252}\), not 256:

> “So to annualize a Sharpe ratio, you multiple a daily Sharpe ratio by square root of 252.”  
> (comment, 17 Dec 2014, <http://epchan.blogspot.com/2014/11/rent-dont-buy-data-our-experience-with.html>)
>
> “This is why if we measure daily returns, we need to multiply the daily volatility by \(\sqrt{252}\) to obtain the annualized volatility.”  
> (“Mean reversion, momentum, and volatility term structure,” 11 Apr 2016, <http://epchan.blogspot.com/2016/04/mean-reversion-momentum-and-volatility.html>)

On the same 2016 post he also refuses a single session-hour constant when bridging intraday to daily: overnight variance is neither 6.5 hours nor 24 hours; for SPY he needed “**1 trading day as equivalent to 10 trading hours**. Not 6.5 … and not 24. The precise number of equivalent trading hours … varies across different instruments.”

The synthesis’s \(N_T = 252 \times 6.5 = 1638\) for hourly NYSE is the usual **session-hours** restatement of “use the actual number of periods in a year” (widely copied, e.g. QuantStart’s Sharpe article). **Chan QT ch. 3 was not in this workspace**, so that 1638 figure is **not** notebook- or book-verified here. The first-party Chan point that is verified is: **daily \(P=252\)**, and **intraday \(P\) should match the clock you actually sampled**, which can differ from both 6.5 and 24.

Relative to that:

- Carver daily = **256** (convenience 16), Chan daily = **252** (trading-day count).
- Carver hourly lookup = **24×256**, Chan-style session hours ≈ **252×6.5** (unverified from the book; Chan’s own vol-bridge example used **~10h**, not 6.5).
- Neither author, in the first-party sources above, annualises by **median \(\Delta t\) of the index**. That third option in [the periods-per-year ticket](../issues/06-periods-per-year.md) is a product construction, not a Carver (or Chan-blog) prescription.

## What Quantester does (code, not Carver)

Repo evidence only. Defaults are **252** for performance/MC/synthetic vol, and **365** (not 365.25) for idle-cash yield. Callers can override some Sharpe/Calmar paths; several sites cannot.

**Hardcoded 252 (annualisation or GBM scaling):**

| Site | What 252 does |
|---|---|
| `quantester/analytics/performance.py` `TRADING_DAYS = 252` | Default `periods` for `annualized_sharpe` (\(\times\sqrt{P}\)) and `calmar_ratio` (`len(equity)/P` as years). Docstring writes \(\sqrt{252}\). |
| `quantester/analytics/dsr.py` | Default `periods_per_year=252.0` when de-annualising PSR/DSR. |
| `quantester/montecarlo/fast_track.py` `FastResult.sharpe` | Simple returns \(\times\sqrt{252}\) — **no `periods` argument**. |
| `quantester/visualization/static.py` `plot_rolling_metrics` | Default `periods=252` on simple `pct_change` Sharpe and vol. |
| `quantester/indicators/__init__.py` `rolling_volatility` | Default `periods=252` on log-return std. |
| `quantester/utils/synthetic.py` | GBM \(\mu/252\), \(\sigma/\sqrt{252}\). |

**365-day cash (calendar simple interest, not 252 and not Carver’s 256):**

`PortfolioManager._accrue_cash_yield` in `quantester/portfolio/portfolio.py`:

```
days = (timestamp - self._last_valuation_ts).total_seconds() / 86400.0
self.cash += self.cash * effective * days / 365.0
```

`effective = cash_yield_rate * idle_cash_fraction`. Tests in `tests/test_portfolio.py` assert the `/365.0` day-count. The module docstring cites Carver only for **including** RF on undeployed cash (*Systematic Trading* ch. 12), not for a 256-day metric year. Kaufman/Carver idle-cash is calendar time between valuations.

**252 used as a lookback, not as \(P\):** `quantester/strategy/pairs_trading.py` `ols_window: int = 252` is a regression window.

**Override exists, default still 252:** `annualized_sharpe(..., periods=...)` and `calmar_ratio(..., periods=...)`. Crypto examples pass 365 or `24*365` (`examples/tranche_pullback/run_ccxt.py`, `examples/donchian_breakout/_shared.py`). `NOTEBOOK_CONTEXT.md` comments `# periods=365 for crypto`. **Fast-track Sharpe cannot be overridden** without editing `fast_track.py`.

Quantester therefore already **mixes** a 252-period metric year with a 365-day cash year. It does **not** implement Carver’s 256/16 daily convention.

## What this does not decide

No product default. In particular this note does not choose among:

- default \(P=256\) (Carver daily convention),
- default \(P=252\) (current `TRADING_DAYS`, Chan daily blog),
- measured \(N_T\) / median \(\Delta t\),
- explicit `periods_per_year` plus a separate cash day-count.

Those belong to [What is the canonical periods-per-year and cash day-count policy?](../issues/06-periods-per-year.md). The Carver fact to carry into that ticket is: **256 is a daily-business-day convenience so the scalar is 16; it is not a universal constant and it is not an instruction to measure the calendar.**

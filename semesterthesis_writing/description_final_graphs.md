# Description of the final PAA graphs

This document describes the PNG version of each generated three-panel graph. The
PDF files contain the same information and are intentionally not analysed
separately.

The graphs use the same layout throughout:

- **Panel (a), cumulative performance and daily alpha:** The black line is the
  PAA portfolio. The blue line is SPY buy-and-hold, the green line is equal-
  weight buy-and-hold, the dashed pink line is the SAA equal-weight reference,
  and the grey dotted line is the 100% cash/EFFR benchmark. The vertical axis
  is cumulative return from the initial capital. The small bars below the
  curves show daily PAA return minus daily SPY return in basis points; green
  bars are positive and red bars are negative.
- **Panel (b), portfolio composition:** The stacked area always totals 100%.
  Grey is cash. The remaining colours identify Crude, Gold, SPY, and the
  international equity ETFs EWG, EWH, EWJ, EWQ, EWS, EWT, EWU, and EWY. The
  height of a colour is the portfolio weight after the daily rebalance.
- **Panel (c), trading behaviour:** The bars show daily turnover as a
  percentage of NAV. Green bars indicate that total risky/equity exposure
  increased, red bars indicate that it decreased into cash, and grey bars
  indicate reallocation with little net exposure change. The classification
  uses the script's current 0.5 percentage-point exposure-change threshold.
  The colour strip below the bars shows the signed daily change in equity
  exposure: red means a decrease, green means an increase, and pale colours
  mean a small change.

The reported turnover figures below are the sum of daily turnover values over
the period, while average exposure is the average pre-rebalance risky-asset
exposure. These are useful numerical complements to the visual composition
panel.

## Hierarchical PAA, run 00283, config 10019

### `best_model_excess_over_spy_abs__validation_val_00_3panel.png`

**File:** `src/agents/PPO_portfolio_allocator_weights/saved_models/00283_config_10019_26_10_02/val_and_test_inference/best_model_excess_over_spy_abs__validation_val_00_3panel.png`

This graph covers validation block `val_00`, from 2005-12-16 to 2007-02-14.
The PAA portfolio finishes at approximately **32.8%** cumulative return. It
ends above SPY buy-and-hold at **15.2%**, equal-weight buy-and-hold at
**24.6%**, and the SAA equal-weight reference at **17.9%**. Thus, in this
period the hierarchical PAA is the strongest of the displayed strategies and
maintains a substantial positive cumulative advantage over both market
benchmarks and the SAA reference. The daily-alpha bars are predominantly
positive during the long rising sections, although negative clusters occur
during drawdowns and sharp reallocations.

The composition panel shows a highly dynamic allocation. Crude is often the
largest individual position, but its weight repeatedly contracts while Gold,
SPY, and the international ETFs expand. The portfolio therefore changes not
only its overall risky exposure but also its preferred risky assets. Cash is a
small but persistent base layer, generally around the high-single-digit
percentage range. The major wide areas in the stack correspond to sustained
preferences for Crude or Gold; narrow, rapidly changing bands correspond to
shorter-lived ETF allocations.

Total turnover is approximately **44.6 times NAV**, with average risky exposure
of approximately **90.7%**. Panel (c) consequently contains many substantial
turnover bars. Green and red bars appear in clusters, showing periods in which
the model changes total market exposure, while grey bars identify sizeable
asset substitutions that leave total exposure broadly unchanged. The exposure
strip is mostly near neutral with alternating green and red episodes rather
than a permanent move to cash.

### `best_model_excess_over_spy_abs__validation_val_01_3panel.png`

**File:** `src/agents/PPO_portfolio_allocator_weights/saved_models/00283_config_10019_26_10_02/val_and_test_inference/best_model_excess_over_spy_abs__validation_val_01_3panel.png`

This graph covers validation block `val_01`, from 2012-01-12 to 2013-03-20.
The PAA finishes at approximately **18.8%**, slightly below SPY buy-and-hold
at **20.2%**, but above equal-weight buy-and-hold at **13.3%** and clearly
above the SAA equal-weight reference at **4.1%**. The comparison therefore
shows a mixed result: the hierarchical PAA does not beat SPY over the whole
block, but it beats both the equal-weight market portfolio and the SAA
reference. The daily-alpha bars alternate frequently, with positive and
negative observations both contributing materially to the final small
shortfall to SPY.

The portfolio composition begins with substantial risky exposure and then moves
through several distinct allocation regimes. Crude is dominant in some
segments, while Gold, SPY, and groups of international ETFs become prominent
in others. Cash remains a relatively thin but stable layer. The stack therefore
indicates active cross-asset selection rather than a simple cash-timing
strategy: large changes in the coloured risky bands often occur without a
large change in the grey cash band.

Turnover is approximately **37.0 times NAV**, and average risky exposure is
approximately **90.7%**. The turnover panel contains frequent grey
reallocation bars together with intermittent green and red bars. This means
that much of the activity is switching between assets, while the colour strip
shows shorter episodes of increasing or reducing total equity exposure. The
portfolio does not spend the period in a sustained defensive cash regime.

### `best_model_excess_over_spy_abs__validation_val_02_3panel.png`

**File:** `src/agents/PPO_portfolio_allocator_weights/saved_models/00283_config_10019_26_10_02/val_and_test_inference/best_model_excess_over_spy_abs__validation_val_02_3panel.png`

This graph covers validation block `val_02`, from 2018-02-27 to 2019-04-23.
The PAA ends at approximately **23.97%**, compared with **6.7%** for SPY
buy-and-hold, **-3.8%** for equal-weight buy-and-hold, and approximately
**0.0%** for SAA equal-weight. The black line is above all comparison lines
by the end of the block. The graph also shows that the final advantage is not
uniform: the PAA experiences a pronounced mid-period drawdown and then
recovers strongly, while SPY also suffers a sharp decline and equal-weight
buy-and-hold remains materially weaker.

The composition is especially regime-like. Crude occupies a very large share
in several long intervals, while Gold expands strongly during other
intervals. The ETF bands become wider when the portfolio reduces its Crude
concentration. Cash stays near a low single-digit/high single-digit base,
rather than becoming the dominant holding. Consequently, the model's
defensive behaviour is expressed mainly through changing risky-asset
composition and moderate exposure reductions, not through holding mostly cash.

Turnover is approximately **44.5 times NAV**, with average risky exposure of
approximately **91.4%**. The turnover bars include several high-intensity
green and red episodes around the major composition shifts, but many grey bars
are also present. The strip shows a pronounced alternation of exposure
direction around the drawdown and recovery. This is consistent with active
de-risking and re-risking, but not with a complete exit from risky assets.

### `best_model_excess_over_spy_abs__validation_val_03_3panel.png`

**File:** `src/agents/PPO_portfolio_allocator_weights/saved_models/00283_config_10019_26_10_02/val_and_test_inference/best_model_excess_over_spy_abs__validation_val_03_3panel.png`

This graph covers validation block `val_03`, from 2024-02-26 to 2025-04-22.
The PAA finishes at approximately **2.1%**, below SPY buy-and-hold at **4.2%**
and equal-weight buy-and-hold at **9.9%**, and below the SAA equal-weight
reference at **5.5%**. The PAA therefore underperforms all three positive-
return references at the endpoint. The black line is comparatively volatile:
it rises early, suffers a deep mid-period loss, and recovers only partially by
the end. Daily-alpha bars therefore alternate strongly and include negative
clusters during the major
drawdown.

The composition panel shows repeated concentration and reversal cycles. Crude
becomes dominant in several sections, while Gold, SPY, and baskets of
international ETFs expand during other sections. The grey cash layer remains
small and relatively stable. The visual impression is therefore of frequent
selection among risky assets rather than a large structural allocation to cash.

Turnover is approximately **48.5 times NAV**, the highest of the hierarchical
validation blocks, while average risky exposure is approximately **91.3%**.
Panel (c) shows repeated high turnover spikes and a mixture of green, red, and
grey bars. The green/red bars correspond to meaningful changes in aggregate
exposure around the drawdown and recovery; grey bars show that many trades
were reallocations within the risky sleeve. The low cash level should not be
read as an absence of risk management: the model changes exposure by
percentage points while remaining mostly invested.

### `best_model_excess_over_spy_abs__test_test_00_3panel.png`

**File:** `src/agents/PPO_portfolio_allocator_weights/saved_models/00283_config_10019_26_10_02/val_and_test_inference/best_model_excess_over_spy_abs__test_test_00_3panel.png`

This graph covers the out-of-sample test block `test_00`, from 2025-07-21 to
2026-08-06. The PAA finishes at approximately **29.5%**, below equal-weight
buy-and-hold at **31.8%**, but above SPY buy-and-hold at **22.2%** and the SAA
equal-weight reference at **13.6%**. The black PAA line rises sharply during
the middle of the test period, reaches a peak above 50%, and then gives back
part of that gain before ending near 30%. SPY and equal-weight buy-and-hold
are steadier; the PAA's advantage over SPY is visible for most of the latter
half but is not sufficient to beat equal-weight buy-and-hold at the endpoint.

The composition is dominated by Crude during several long stretches, with
Gold and the ETF sleeve expanding during rotations away from Crude. SPY
becomes more visible during some defensive or market-led phases. Cash remains
around the high-single-digit range throughout, including during the major
drawdown, so the large performance swings are mainly associated with risky
asset selection and changing exposure rather than a move to an almost
cash-only portfolio.

Total turnover is approximately **36.9 times NAV**, with average risky
exposure of approximately **90.4%**. The turnover panel shows isolated large
green and red bars around the major exposure changes, but also many grey
reallocation bars. Relative to the validation graphs, the test period has
fewer days of heavy trading in some stretches, yet the exposure strip still
shows repeated risk-on and de-risk transitions.

## Random-walk SAA PAA, run 00294, config 20004

### `best_model_excess_over_spy_abs__validation_val_00_3panel.png`

**File:** `src/agents/PAA_with_autoregressive_rnd_walk_SAA/saved_models/00294_config_20004_26_10_06/val_and_test_inference/best_model_excess_over_spy_abs__validation_val_00_3panel.png`

This graph covers validation block `val_00`, from 2005-12-16 to 2007-02-14.
The random-walk-signal PAA finishes at approximately **18.2%**, above SPY
buy-and-hold at **15.2%**, but below equal-weight buy-and-hold at **24.6%**.
It finishes close to the SAA equal-weight reference at **17.9%**. The black
line rises early, has several reversals, and ends only slightly above SPY.
Positive daily-alpha bars occur in the early and later recoveries, while
negative bars cluster around declines; the aggregate edge over SPY is modest.

The composition changes rapidly across the entire period. Crude is often the
largest holding, but Gold, SPY, and the international ETFs repeatedly become
large sleeves. Compared with the hierarchical graph for the same period, the
composition is more visibly irregular and oscillatory, which is consistent
with the injected AR(1) signal rather than a stable SAA signal. Cash remains
small and persistent.

Turnover is approximately **47.5 times NAV**, with average risky exposure of
approximately **91.9%**. Panel (c) has frequent green, red, and grey bars,
including several large spikes. The grey bars demonstrate that high trading
activity often reflects movement between risky assets, not only an increase
or decrease in total exposure. The exposure strip alternates frequently,
showing that the random signal causes repeated risk-on and de-risk decisions.

### `best_model_excess_over_spy_abs__validation_val_01_3panel.png`

**File:** `src/agents/PAA_with_autoregressive_rnd_walk_SAA/saved_models/00294_config_20004_26_10_06/val_and_test_inference/best_model_excess_over_spy_abs__validation_val_01_3panel.png`

This graph covers validation block `val_01`, from 2012-01-12 to 2013-03-20.
The PAA ends at approximately **8.4%**, below SPY buy-and-hold at **20.2%**
and equal-weight buy-and-hold at **13.3%**, but above the SAA equal-weight
reference at **4.1%**. The black line is profitable by the end but remains
below both market buy-and-hold references for most of the second half. The
daily-alpha panel contains more negative than positive contribution over the
full period, explaining the final shortfall to SPY.

The stack plot shows broad rotations among Crude, Gold, SPY, and the
international ETFs. Crude is dominant in some phases, while Gold and
multiple ETF bands expand in others. The cash layer stays near the same small
baseline. The portfolio is therefore continuously changing its risky
allocation, but it is not substantially reducing the total risky sleeve for
long periods.

Turnover is approximately **44.0 times NAV**, and average risky exposure is
approximately **91.8%**. The turnover panel is populated by all three
categories, with grey reallocation bars especially common during the middle
of the period. Larger green and red spikes occur around the strongest
composition changes. The strip's rapid colour changes indicate frequent
short-horizon exposure changes rather than one sustained defensive decision.

### `best_model_excess_over_spy_abs__validation_val_02_3panel.png`

**File:** `src/agents/PAA_with_autoregressive_rnd_walk_SAA/saved_models/00294_config_20004_26_10_06/val_and_test_inference/best_model_excess_over_spy_abs__validation_val_02_3panel.png`

This graph covers validation block `val_02`, from 2018-02-27 to 2019-04-23.
The PAA finishes at approximately **19.7%**, well above SPY buy-and-hold at
**6.7%**, equal-weight buy-and-hold at **-3.8%**, and the approximately
**0.0%** SAA equal-weight reference. The black line experiences interim
drawdowns but recovers strongly and maintains a large positive terminal
advantage. The daily-alpha bars contain repeated positive clusters, which
are particularly visible during the recovery phases.

The composition is highly variable. Crude is frequently large, but Gold and
the international equity ETFs periodically replace it; SPY also expands in
some sections. Cash remains a narrow lower band. This indicates that the
model's strong result is generated while remaining mostly invested and
rotating among risky assets, not by holding cash throughout the adverse
market intervals.

Turnover is approximately **50.5 times NAV**, with average risky exposure of
approximately **91.9%**. This is a high-activity graph: large green and red
bars appear around exposure changes, and many grey bars show simultaneous
reallocation within the risky sleeve. The exposure strip is visibly
alternating, with risk-on phases during recovery and de-risk phases around
drawdowns.

### `best_model_excess_over_spy_abs__validation_val_03_3panel.png`

**File:** `src/agents/PAA_with_autoregressive_rnd_walk_SAA/saved_models/00294_config_20004_26_10_06/val_and_test_inference/best_model_excess_over_spy_abs__validation_val_03_3panel.png`

This graph covers validation block `val_03`, from 2024-02-26 to 2025-04-22.
The PAA ends at approximately **22.3%**, well above SPY buy-and-hold at
**4.2%**, equal-weight buy-and-hold at **9.9%**, and the SAA equal-weight
reference at **5.5%**. The black line shows a strong upward trajectory with
intermediate volatility. The positive daily-alpha bars are concentrated in
the periods where the PAA rises away from the relatively flat benchmarks,
although negative bars appear around reversals.

The portfolio repeatedly changes its dominant holdings. Crude occupies large
areas in several sections, but Gold, SPY, and the international ETFs expand
substantially during other sections. Cash remains low and stable. The
composition therefore shows an actively rotating risky portfolio rather than
a simple cash overlay.

Turnover is approximately **56.3 times NAV**, the highest of the three
families in this validation block, with average risky exposure of
approximately **92.6%**. The turnover bars are dense and include several
large green and red spikes, while grey bars show that much of the activity is
still asset reallocation. The strip shows repeated changes in exposure
direction, but the high average exposure confirms that the model generally
remains invested.

### `best_model_excess_over_spy_abs__test_test_00_3panel.png`

**File:** `src/agents/PAA_with_autoregressive_rnd_walk_SAA/saved_models/00294_config_20004_26_10_06/val_and_test_inference/best_model_excess_over_spy_abs__test_test_00_3panel.png`

This graph covers test block `test_00`, from 2025-07-21 to 2026-08-06.
The PAA ends at approximately **26.6%**, above SPY buy-and-hold at **22.2%**
but below equal-weight buy-and-hold at **31.8%**. It is also well above the
SAA equal-weight reference at **13.6%**. The black line rises strongly but
with visible drawdowns and does not match the best terminal result of the
equal-weight benchmark. Positive daily alpha is common in the rising
sections, but the later path contains several negative bursts.

The composition alternates between Crude-heavy, Gold-heavy, and diversified
ETF-heavy states. SPY and Gold form visibly larger areas during some periods,
while the international ETF bands expand during rotations away from the
dominant commodity positions. Cash remains a small base layer. The graph
therefore supports the conclusion that the PAA is actively selecting risky
assets while only modestly changing aggregate cash exposure.

Turnover is approximately **45.4 times NAV**, with average risky exposure of
approximately **92.0%**. Green and red spikes show repeated risk-on and
de-risk decisions, while grey bars are frequent and indicate reallocations
that do not materially change total risky exposure. Relative to the
hierarchical test graph, this model has a lower terminal return and more
signal-driven variability, while still maintaining a similar high-investment
profile.

## Cross-sectional-only PAA, run 00298, config 30004

### `best_model_excess_over_spy_abs__validation_val_00_3panel.png`

**File:** `src/agents/PAA_cross_sectional_only/saved_models/00298_config_30004_26_10_07/val_and_test_inference/best_model_excess_over_spy_abs__validation_val_00_3panel.png`

This graph covers validation block `val_00`, from 2005-12-16 to 2007-02-14.
The cross-sectional-only PAA finishes at approximately **32.5%**, above SPY
buy-and-hold at **15.2%**, equal-weight buy-and-hold at **24.6%**, and the
SAA equal-weight reference at **17.9%**. The result is close to the
hierarchical PAA's terminal return for this block. The black line rises
strongly but has several drawdowns; positive daily-alpha clusters explain
the sustained advantage over SPY.

The composition panel shows strong cross-sectional rotation. Crude is often
the largest holding, but Gold, SPY, and the international ETFs periodically
take larger shares. Because the signal is cross-sectional-only, there is no
SAA signal injected into the PAA; the SAA line in panel (a) is a reference
benchmark, not an input to this agent. The cash band remains relatively small,
so the strategy's main behaviour is selecting among risky assets.

Turnover is approximately **37.4 times NAV**, with average risky exposure of
approximately **92.9%**. Green and red bars appear around exposure changes,
but grey bars are also frequent and represent cross-sectional reshuffling
without a major aggregate risk change. The exposure strip shows alternating
directional decisions, while the high average exposure indicates that these
decisions generally occur inside an invested portfolio.

### `best_model_excess_over_spy_abs__validation_val_01_3panel.png`

**File:** `src/agents/PAA_cross_sectional_only/saved_models/00298_config_30004_26_10_07/val_and_test_inference/best_model_excess_over_spy_abs__validation_val_01_3panel.png`

This graph covers validation block `val_01`, from 2012-01-12 to 2013-03-20.
The PAA ends at approximately **17.3%**, below SPY buy-and-hold at **20.2%**,
but above equal-weight buy-and-hold at **13.3%** and the SAA equal-weight
reference at **4.1%**. The PAA remains profitable despite not beating SPY.
The daily-alpha bars show that the shortfall to SPY is accumulated through
repeated negative days rather than one isolated terminal event.

The stack plot shows frequent changes among Crude, Gold, SPY, and the
international ETFs. At different times, Crude is dominant; at others, the
portfolio is more diversified across ETF colours and Gold. Cash is a narrow
baseline. This is consistent with a cross-sectional selection strategy:
large visual changes occur within the risky sleeve, and the SAA reference
does not describe the signal driving these choices.

Turnover is approximately **51.7 times NAV**, with average risky exposure of
approximately **91.6%**. The turnover panel is dense, with many grey bars
intermixed with green and red bars. Thus, the high turnover should not be
interpreted as continuous de-risking: much of it represents changing the
relative weights of risky assets. The colour strip shows frequent but
generally short-lived exposure changes.

### `best_model_excess_over_spy_abs__validation_val_02_3panel.png`

**File:** `src/agents/PAA_cross_sectional_only/saved_models/00298_config_30004_26_10_07/val_and_test_inference/best_model_excess_over_spy_abs__validation_val_02_3panel.png`

This graph covers validation block `val_02`, from 2018-02-27 to 2019-04-23.
The PAA ends at approximately **12.2%**, above SPY buy-and-hold at **6.7%**,
equal-weight buy-and-hold at **-3.8%**, and the approximately **0.0%** SAA
equal-weight reference. The PAA therefore produces a positive terminal
return and beats all three comparison lines, although its advantage is
smaller than the hierarchical and random-walk PAA advantages in the same
block. The daily-alpha bars are predominantly positive during the recovery.

The composition shows a relatively stable low cash layer combined with
substantial rotation among Crude, Gold, SPY, and the international ETFs.
Crude becomes dominant in long stretches, while other risky assets expand
during cross-sectional shifts. Since the signal is zero-SAA by construction,
the SAA line is only a common reference and should not be interpreted as an
agent input.

Turnover is approximately **46.9 times NAV**, with average risky exposure of
approximately **92.2%**. The panel contains sizeable green and red exposure
bars around changes in aggregate risk, but a large number of grey bars show
that much of the trading is internal reallocation. The strip records
alternating exposure changes without a prolonged cash-heavy phase.

### `best_model_excess_over_spy_abs__validation_val_03_3panel.png`

**File:** `src/agents/PAA_cross_sectional_only/saved_models/00298_config_30004_26_10_07/val_and_test_inference/best_model_excess_over_spy_abs__validation_val_03_3panel.png`

This graph covers validation block `val_03`, from 2024-02-26 to 2025-04-22.
The PAA finishes at approximately **16.3%**, above SPY buy-and-hold at
**4.2%** and equal-weight buy-and-hold at **9.9%**, and also above the SAA
equal-weight reference at **5.5%**. The black line is positive for much of
the period and retains a clear terminal advantage over every comparison
line. Positive daily-alpha bars are especially visible during the advances,
with negative bars around reversals.

The composition panel shows cross-sectional switching rather than a stable
single-asset allocation. Crude and Gold each become large at different
times, while SPY and the international ETFs fill the remaining risky sleeve.
Cash remains low. As in the other cross-sectional graphs, the broad
composition changes should be read as selection among risky assets; the
SAA-equal-weight curve is included for comparison only.

Turnover is approximately **37.2 times NAV**, with average risky exposure of
approximately **92.8%**. The lower turnover relative to `val_01` and `val_02`
does not mean that the portfolio is static: the composition still changes
substantially. Green and red bars mark net exposure changes, while grey bars
capture the many reallocations that preserve approximately the same overall
risky exposure.

### `best_model_excess_over_spy_abs__test_test_00_3panel.png`

**File:** `src/agents/PAA_cross_sectional_only/saved_models/00298_config_30004_26_10_07/val_and_test_inference/best_model_excess_over_spy_abs__test_test_00_3panel.png`

This graph covers test block `test_00`, from 2025-07-21 to 2026-08-06.
The PAA finishes at approximately **31.8%**, essentially matching
equal-weight buy-and-hold at **31.8%**. It finishes above SPY buy-and-hold at
**22.2%** and well above the SAA equal-weight reference at **13.6%**. The
black line rises strongly and tracks the equal-weight benchmark closely at
the endpoint, although the two paths differ materially during the period.
The daily-alpha panel shows alternating positive and negative contributions,
so the terminal match with equal-weight is the result of the full path and
not of identical daily returns.

The composition alternates between Crude-heavy, Gold-heavy, and
international-ETF-heavy states, with SPY also becoming prominent in some
phases. Cash remains a small persistent layer. The cross-sectional-only
agent therefore achieves its strong test result primarily through risky-asset
selection and rotation, not by holding a large cash position.

Turnover is approximately **36.5 times NAV**, with average risky exposure of
approximately **92.7%**. Panel (c) contains a mixture of grey
reallocation bars and isolated green/red spikes. This indicates that the
portfolio changes its asset rankings frequently while only occasionally
making a larger net exposure adjustment. The exposure strip confirms
repeated risk-on and de-risk movements, but the high average exposure and
small cash band show that these are incremental changes rather than complete
entries into or exits from the market.

## Cross-checks for comparing validation and test graphs

The five blocks are shared across the three active checkpoints, which makes
within-block comparisons meaningful:

- `val_00` is the earliest historical validation block.
- `val_01` is the second validation block.
- `val_02` is the third validation block.
- `val_03` is the most recent validation block.
- `test_00` is the out-of-sample test block.

The safest comparison is to compare the terminal cumulative returns in panel
(a), then use the daily-alpha bars to understand how the difference developed.
Panel (b) explains *which assets* produced the result, while panel (c)
separates trading intensity from aggregate risk-direction changes. In
particular, a high turnover value does not by itself imply aggressive
de-risking: grey bars can represent substantial trading that leaves total
equity exposure nearly unchanged. Across the active checkpoints, average
risky exposure remains around 90--93%, so the graphs should generally be
interpreted as highly invested portfolios with active cross-sectional
rotation and incremental exposure adjustments, rather than as strategies that
regularly move most capital into cash.

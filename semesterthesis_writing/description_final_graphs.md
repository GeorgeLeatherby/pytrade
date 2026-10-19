# Description final graphs

## Descriptions of the SAA inference graphs

The graphs below are the twelve PNG figures produced for run
`00306_config_01062_26_10_13`. In the aggregate figures, the vertical axis is a
wealth index with the initial value normalised to 100 (`t_0 = 100`), and the
horizontal axis is trading day. The blue line is the equal-weight combined
agent portfolio, the red line is a buy-and-hold position in SPY, and the green
line is a 100% cash position earning EFFR. In the asset-analysis figures, blue
is the agent position and orange is the buy-and-hold benchmark for the named
asset. The small lower chart in each asset panel shows transaction volume in
dollars: green bars are positive purchases/increases and red bars are negative
sales/decreases. The zero line separates the two directions.

## `aggregate_saa_portfolio_test_20051216_20070214_26_10_13_12_19.png`

This figure compares the combined SAA portfolio with SPY buy-and-hold and
cash from 16 December 2005 to 14 February 2007. The x-axis runs from trading
day 0 to approximately day 290; the y-axis covers roughly 96 to 121 index
points. All series start at 100. The blue agent line rises quickly during the
first 25 days to about 104, briefly jumps to approximately 107 around day 28,
and then fluctuates between about 103 and 109 through day 100. It undergoes a
pronounced drawdown from about 109 at day 98 to just below 100 around day 123,
which is the most important adverse event for the agent in this period because
it temporarily removes almost all of the gains accumulated during the first
half of the run. From about day 145 onward, the blue series recovers strongly:
it moves above 110 near day 160, reaches about 115 around day 230, dips to
roughly 115 at day 260, and finishes near 119.8. The red SPY line is more
volatile early, falls below 100 several times, reaches a low near 97.0 around
day 120, and then climbs persistently after day 145 to finish near 115.2. The
green cash line is almost a straight, shallow increase from 100 to about
105.8. Thus, the agent finishes above both benchmarks, with reported returns
of approximately +19.8% for the agent, +15.2% for SPY, and +5.8% for cash.
The widening blue-red separation after day 150 shows that the agent captured
the later advance more effectively than SPY, while the temporary blue
crossing below 100 near day 123 shows that the advantage was not achieved
without interim loss.

## `aggregate_saa_portfolio_test_20120112_20130320_26_10_13_12_19.png`

This aggregate chart covers 12 January 2012 to 20 March 2013. The axes again
show trading day on x and the normalised index on y, with the visible y-range
approximately 93 to 122. The green cash benchmark remains nearly horizontal
at 100, ending just above it at about 100.2. The blue agent begins at 100,
increases to roughly 104 by day 40, and then loses ground between days 75 and
100. Its principal low is about 94.3 around day 94, making this the key
drawdown event: the agent is below both the cash level and the red benchmark
at that point. It recovers above 100 around day 130, rises to about 104 by
day 145, and makes a step-like move to approximately 109 near day 165.
After day 200 it remains mostly between 106 and 113, ending near 113.2.
The red SPY series initially outperforms the agent, reaching about 109 around
day 50. It then falls to approximately 99 near day 95, recovers through the
middle of the sample, and rises sharply after day 235, ending near 120.2.
Consequently, SPY finishes well above the agent in this period. The summary
box reports approximately +13.2% for the agent, +20.2% for SPY, and +0.2% for
cash. The important interpretation is that the agent limits the early and
mid-period decline relative to SPY but does not participate fully in the
strong final SPY rally.

## `aggregate_saa_portfolio_test_20180227_20190423_26_10_13_12_19.png`

This figure spans 27 February 2018 to 23 April 2019. The x-axis covers about
290 trading days and the y-axis ranges from roughly 84 to 108. The blue agent
starts at 100 and rises to approximately 104 by day 55, then reaches about
106--107 around day 145. The most notable common stress event begins near
day 160: both lines fall sharply, but the blue line bottoms near 92.5 around
day 207 while the red SPY line falls much further, to roughly 85.3 at about
day 208. The agent therefore loses value but retains a substantial relative
advantage during the trough. The blue line recovers to approximately 99 by
day 225, crosses 100 around day 245, and ends near 103.8. SPY recovers more
slowly, returns to 100 only around day 245, and finishes near 106.7. The
green cash line increases gradually from 100 to about 102.4. The reported
returns are approximately +3.8% for the agent, +6.7% for SPY, and +2.4% for
cash. The chart therefore shows a defensive advantage during the sharp
drawdown, but insufficient participation in the subsequent recovery to beat
SPY over the full interval.

## `aggregate_saa_portfolio_test_20240226_20250422_26_10_13_12_19.png`

This chart covers 26 February 2024 to 22 April 2025, with trading day on the
x-axis and a normalised index on the y-axis (approximately 97 to 122). The
blue agent and red SPY lines both start at 100. During the first 35 days, both
rise, but the red line then drops to about 98 around day 38 while the blue
line stays close to 100. The agent climbs in stages to about 105 by day 80,
experiences a temporary low near 101 around day 110, and then rises to roughly
110 around day 150. It remains between about 108 and 111 until day 220 before
accelerating to approximately 117 near day 245. A later dip to about 111 at
day 270 is followed by a strong rebound; the agent finishes near 119.4.
SPY rises much more strongly through the middle of the sample, reaching about
120 around days 195--245, but then suffers a severe late drawdown. It falls
from approximately 121 near day 245 to below 100 around day 280, before a
partial recovery to about 104.2 at the end. The green cash line increases
steadily to about 105.8. The reported final returns are approximately +19.4%
for the agent, +4.2% for SPY, and +5.8% for cash. The late red collapse and
the blue recovery are the decisive events: the agent gives up some mid-period
upside but avoids most of the benchmark's terminal loss and finishes far
ahead.

## `asset-specific_saa_portfolio_test_page1_20051216_20070214_26_10_13_12_19.png`

This six-panel asset figure covers 16 December 2005 to 14 February 2007. Each
upper panel has trading day on the x-axis and an index starting at 100 on the
y-axis; blue is the agent and orange is buy-and-hold. Each lower panel uses
the same x-axis and shows dollar transaction volume, with green positive bars
and red negative bars.

* **Crude:** The agent rises briefly to about 111 near day 30, then stays
  close to 100--106. Buy-and-hold climbs to roughly 130 around days 140--165
  before falling sharply. Both strategies reach their deepest late dip near
  day 270 (about 88--90), after which the agent ends near 101 and buy-and-hold
  near 100. The agent's final return is about +1.46%, versus approximately
  -0.10% for buy-and-hold; the large late fall explains why the agent's
  maximum drawdown is visually much larger than its modest final gain.
* **EWG:** Buy-and-hold rises almost continuously after day 120 and finishes
  near 139, while the agent climbs in steps to about 125. The agent has a
  conspicuous low near 101 around day 120 but recovers rapidly; the later
  agent-versus-benchmark crossover around day 220 shows relative
  underperformance during the final rally. The lower chart contains repeated
  alternating purchase and sale clusters, especially around days 120--200
  and 250--280.
* **EWH:** Both lines fall around day 115--125 to about 95--100, then recover.
  The agent finishes near 124 and buy-and-hold near 129; buy-and-hold is
  stronger after day 230. The agent's low around day 120 and the late
  divergence are the main events. Transaction bars become particularly large
  near days 250--280, indicating substantial repositioning during the final
  move.
* **EWJ:** The agent experiences a deep low near day 120, around 90, but then
  makes step increases and ends around 116. Buy-and-hold peaks near 117 around
  day 95, falls to roughly 95 near day 125, and finishes near 112. The agent's
  recovery and final outperformance are supported by several purchase/sale
  clusters in the lower panel.
* **EWQ:** After a volatile start, the agent reaches approximately 118 by day
 160 and continues upward to about 132. Buy-and-hold briefly peaks near 118
  around day 95, falls toward 105 near day 125, and finishes near 131. The
  agent's sustained late rise and its small final lead are the key features;
  the lower panel shows frequent two-sided rebalancing.
* **EWS:** Buy-and-hold rises from 100 to approximately 120 around day 100,
  falls to about 104 near day 125, and then accelerates to nearly 150 by the
  end. The agent suffers a smaller trough near 100 at the same time but only
  reaches about 121. The persistent divergence after day 170 is the clearest
  example of the agent lagging a strong benchmark trend. The summary on the
  figure ranks this asset as the largest negative alpha, approximately
  -28.98%.

## `asset-specific_saa_portfolio_test_page2_20051216_20070214_26_10_13_12_19.png`

This companion page uses the same axes, colours, normalisation, and
transaction-volume convention as the preceding page, for the remaining
assets in the 2005--2007 interval.

* **EWT:** Buy-and-hold is highly erratic, reaching about 122 near day 95,
  falling to roughly 96 around day 145, and ending close to 117. The agent
  falls to about 94 near day 120 and then recovers more steadily to about 115.
  The agent is below buy-and-hold at the end, but its smoother post-day-160
  progression shows reduced exposure to the benchmark's sharp swings.
* **EWU:** The agent rises from 100 to about 111 around day 95, drops to
  approximately 104 near day 125, and then advances to about 122. Buy-and-hold
  takes longer to recover from its low near day 125 but finishes around 124,
  slightly higher. The two lines converge repeatedly, and the green/red
  transaction clusters show active switching during those reversals.
* **EWY:** Buy-and-hold reaches about 122 near day 95, falls to around 98 at
  day 125, and later recovers to approximately 117. The agent has a deeper
  trough, near 92 around day 120, and ends around 107. Thus the late
  benchmark recovery is not matched by the agent; the panel's summary reports
  about -11.03% alpha.
* **Gold:** Buy-and-hold is the most volatile series in this panel: it spikes
  above 140 around day 100, falls to roughly 112 near day 125, and ends near
  133. The agent increases in relatively smooth steps to about 136 and ends
  above buy-and-hold. The crossovers around days 150--220 show the agent
  holding value while the benchmark oscillates. This is the strongest positive
  relative result on the page, with approximately +3.57% alpha.
* **SPY:** Both series follow a similar broad path: an early increase to
  roughly 105, a decline toward 98 around day 125, and a sustained recovery
  after day 150. The agent and benchmark finish close to 114--115, with the
  orange line marginally higher at the end. The lower panel contains only a
  few concentrated transaction clusters, unlike the more actively traded
  assets.

The text block at lower right ranks the asset-level alpha values. It shows
positive relative performance for EWJ, Gold, and Crude, near-zero or mildly
negative relative performance for SPY and EWT, and the largest losses relative
to buy-and-hold for EWS, EWY, and EWH. This ranking matters because the
combined portfolio result is the average of the asset-level outcomes rather
than a result driven by one chart alone.

## `asset-specific_saa_portfolio_test_page1_20120112_20130320_26_10_13_12_19.png`

This page covers 12 January 2012 to 20 March 2013 and contains Crude, EWG,
EWH, EWJ, EWQ, and EWS. The upper panels use the normalised index (blue agent,
orange buy-and-hold); the lower panels show dollar purchases and sales.

* **Crude:** Both strategies begin near 100, rise to roughly 105--110 by day
  50, and then fall sharply. The benchmark reaches its low near 79 around
  day 110, while the agent bottoms around 83--84. The agent then recovers to
  about 103 and remains above the benchmark, which ends near 94. This is a
  clear defensive success during the second half, reflected by approximately
  +8.00% alpha.
* **EWG:** Buy-and-hold rises to about 117 around day 55, falls below 95 near
  day 105, and finishes around 126. The agent initially stays near 105,
  declines less severely, and then makes large upward steps after day 150,
  ending near 138. The late agent lead is the dominant event and produces the
  page's largest positive alpha, approximately +22.79%.
* **EWH:** The benchmark rises to about 115 near day 35, falls to around 100
  near day 95, and then climbs to approximately 128 before ending near 122.
  The agent is flatter initially, recovers in stages after day 150, and ends
  near 120. It avoids much of the benchmark's early volatility but does not
  match its final peak.
* **EWJ:** Both series fall around days 85--105 to the mid- to high-90s. The
  agent then recovers stepwise and plateaus near 107 after day 220. Buy-and-
  hold remains weak until about day 220 and then rises rapidly to roughly 118
  at the end. This late benchmark surge explains the agent's approximately
  -10.28% alpha.
* **EWQ:** The agent falls to about 86 around day 90, then climbs strongly,
  reaching roughly 119 by day 170 and about 125 at the end. Buy-and-hold
  reaches approximately 115 early, declines to about 92 near day 125, and
  ends near 123. The agent's recovery is more persistent and gives a small
  positive relative result, approximately +1.45% alpha.
* **EWS:** Buy-and-hold starts at 100, rises above 110, drops near 102 around
  day 100, and then recovers to about 120 by the end. The agent remains close
  to 100 for much of the first half, falls to approximately 91 near day 90,
  and recovers only to about 105. The sustained orange-blue separation after
  day 150 is the principal event and corresponds to approximately -15.15%
  alpha.

## `asset-specific_saa_portfolio_test_page2_20120112_20130320_26_10_13_12_19.png`

This page shows EWT, EWU, EWY, Gold, and SPY for the same 2012--2013 period.
All upper panels plot the normalised asset index and all lower panels plot
green positive and red negative transaction volume.

* **EWT:** Buy-and-hold rises above 112 in the first 40 days, falls to about
  95 near day 125, and later fluctuates around 108--112. The agent stays near
  100 initially, reaches about 110 around day 165, and ends near 106. The
  agent's smoother but lower final path yields approximately -1.51% alpha.
* **EWU:** Both series decline to a low near day 90--100, with the agent
  reaching about 91 and buy-and-hold about 95. They recover after day 130;
  the agent makes step increases and ends near 113, while buy-and-hold ends
  near 112. The late crossover is the important event and gives a small
  positive alpha of approximately +1.38%.
* **EWY:** Buy-and-hold rises above 110 early, drops to around 98 near day
  100, and later reaches about 122 before ending near 109. The agent begins
  lower, falls to around 98 at the same time, then advances in steps to about
  112. The agent reduces the early benchmark lead but does not capture the
  late benchmark peak; alpha is approximately +2.16% according to the figure.
* **Gold:** The benchmark shows two pronounced peaks, near 108 around days
  30 and 175, separated by a fall toward 94--96. The agent rises more
  smoothly to about 105, remains near 102--105 through the second peak, and
  finishes around 100 while buy-and-hold finishes near 97. The agent's
  relative protection during the final decline produces approximately +2.77%
  alpha.
* **SPY:** The blue and orange series move closely together: both rise from
  100 to approximately 108 near day 55, fall to around 98 near day 95, and
  then recover to about 120 at the end. The curves cross frequently, and the
  small final separation favours the benchmark. The summary reports a small
  negative agent alpha of approximately -0.48%.

The ranking block confirms that the largest positive relative result on this
page is EWG, followed by Crude, EWQ, Gold, and EWY; EWU and SPY are close to
neutral, while EWS and EWJ are the main negative contributors.

## `asset-specific_saa_portfolio_test_page1_20180227_20190423_26_10_13_12_19.png`

This page covers 27 February 2018 to 23 April 2019 for Crude, EWG, EWH,
EWJ, EWQ, and EWS. The plotted indices start at 100; blue is the agent,
orange is buy-and-hold, and the lower bar charts show positive and negative
transaction volume.

* **Crude:** The two lines track closely at first and rise to about 110--120
  around day 150. They then collapse together around days 180--210, with the
  agent reaching approximately 74 and buy-and-hold about 68. The agent
  recovers to around 90, whereas buy-and-hold reaches about 105 by the end.
  The common crash is the defining event; the agent reduces the final loss
  but still underperforms the strong benchmark recovery.
* **EWG:** The agent rises gradually to about 110 near day 145 and remains
  mostly above 100. Buy-and-hold declines from approximately 101 to 75 around
  day 210 and finishes below 88. The persistent blue-over-orange gap,
  especially from day 150 onward, is the strongest defensive result on the
  page and corresponds to about +22.79% alpha.
* **EWH:** The agent stays around 100--106 and finishes near 112, while the
  benchmark falls from about 102 to 83--85 around days 165--210 before
  recovering to approximately 103. The agent's relatively stable path
  avoids the benchmark's prolonged decline and gives approximately +8.93%
  alpha.
* **EWJ:** Buy-and-hold falls from roughly 99 to about 80 near day 210 and
  ends near 90. The agent also dips, reaching about 90 at the same point, but
  recovers to approximately 100. The common trough followed by the wider
  blue-orange separation is the central event; alpha is approximately
  +10.28%.
* **EWQ:** The benchmark declines from about 103 to 81 around day 210 and
  recovers only to about 97. The agent reaches a high near 116 around day
  150, falls to about 99 near day 210, and ends near 108. The agent's
  outperformance is visible throughout the second half and is approximately
  +11.59%.
* **EWS:** Both series rise early and fall around day 80--100, but the agent
  recovers to about 105 while buy-and-hold stays near 86--92 for much of the
  remainder. The sustained blue lead from about day 90 onward produces
  approximately +12.87% alpha.

The page demonstrates that the agent's strongest 2018--2019 contribution is
downside protection: it is above buy-and-hold for every listed asset at the
end, although Crude remains the exception in terms of final absolute recovery.

## `asset-specific_saa_portfolio_test_page2_20180227_20190423_26_10_13_12_19.png`

This page contains EWT, EWU, EWY, Gold, and SPY for 2018--2019. The upper
plots use the same 100-based index, blue agent line, and orange buy-and-hold
line. The lower plots show the timing and direction of transactions.

* **EWT:** Both series oscillate around 100 during the first half. The agent
  then remains mostly above 100, reaches about 105 near day 150, falls to
  approximately 93 around day 210, and ends near 101. Buy-and-hold falls
  much more sharply after day 150, reaches roughly 82 near day 210, and
  finishes near 97. The agent's avoidance of the prolonged benchmark decline
  yields approximately +4.39% alpha.
* **EWU:** Buy-and-hold rises above 105 early, then declines to about 82 near
  day 210 before recovering to approximately 96. The agent falls more
  gradually, reaches about 89 near day 210, and ends close to 99. The
  relative advantage is largest during the late drawdown and is approximately
  +2.81% alpha.
* **EWY:** The agent climbs to roughly 112 by day 150, briefly falls to about
  95 near day 170, and ends close to 105. Buy-and-hold steadily loses value
  after day 70, reaches around 79--80 near day 210, and ends near 87. The
  persistent blue-orange separation is the main event and corresponds to
  approximately +18.44% alpha.
* **Gold:** Buy-and-hold falls from about 102 to approximately 90 around day
  120, recovers to about 100 near day 250, and ends below 97. The agent
  remains near 98--102 and finishes around 102. The agent therefore avoids
  most of the benchmark's decline, producing approximately +5.50% alpha.
* **SPY:** Both curves rise from 100 to roughly 107--108 by day 150, fall
  abruptly around day 210 to the mid-80s or low-90s, and then recover above
  105. The blue line finishes slightly above the orange line, but the
  difference is small; the summary reports approximately +1.63% alpha.

The ranking block shows positive alpha for all five assets on this page. The
largest contribution is EWY, followed by EWU, EWS, EWQ, and EWJ; Gold, EWT,
and SPY provide smaller but still positive relative results.

## `asset-specific_saa_portfolio_test_page1_20240226_20250422_26_10_13_12_19.png`

This page covers 26 February 2024 to 22 April 2025 and shows Crude, EWG, EWH,
EWJ, EWQ, and EWS. The upper panels are indexed asset wealth paths; blue is
the agent and orange is buy-and-hold. The lower panels show dollar purchases
above zero and sales below zero.

* **Crude:** Buy-and-hold rises above 110 early, falls to around 96 near day
  55, and then declines toward 88--90 through the middle of the period. A
  late fall near days 260--280 takes it below 80 before a small rebound.
  The agent is less extreme, generally between 90 and 101, and finishes near
  97. The agent therefore avoids the benchmark's terminal collapse, but its
  final level remains above the starting value only modestly.
* **EWG:** Both lines rise, with the agent reaching about 123 and buy-and-
  hold briefly reaching nearly 130 around day 265. Buy-and-hold then has a
  very sharp drop near day 275 to about 110, followed by a rebound to 125.
  The agent remains close to 122--123 through that event and ends slightly
  above the benchmark. The late shock is the key reason for the agent's
  approximately -2.36% alpha despite its more stable path.
* **EWH:** The agent spikes to about 127 near day 155, remains mostly between
  110 and 120, and finishes near 118. Buy-and-hold peaks near 122 at the same
  time, falls to around 97 near day 225, and ends near 102. The agent's
  sustained lead after day 160 is a major positive result, approximately
  +16.70% alpha.
* **EWJ:** The agent increases almost monotonically, with step changes near
  days 110, 150, 180, and 230, and ends near 126. Buy-and-hold oscillates
  around 100, falls to approximately 89 near day 275, and finishes near 100.
  The large and persistent blue-orange gap makes EWJ one of the strongest
  positive contributors, at approximately +25.91% alpha.
* **EWQ:** Both series suffer several declines, including a common low around
  day 200 near 90--93. The agent subsequently rises to about 105 and ends
  near 108, whereas buy-and-hold experiences a sharp late fall near day 280
  to about 89 before ending near 99. The agent's late recovery and avoidance
  of the final trough produce approximately +9.00% alpha.
* **EWS:** Both lines trend upward from 100, but buy-and-hold is generally
  above the agent until a sharp fall near day 280. The agent finishes near
  142 after a final upward jump, while buy-and-hold finishes near 131. The
  late reversal is decisive and gives approximately +11.10% alpha.

The lower transaction panels show concentrated rebalancing around the major
turning points, especially near days 150, 200, 250, and 275. The page's
ranking identifies EWJ, EWH, EWS, EWQ, and Crude as positive relative
contributors, while EWG is slightly negative.

## `asset-specific_saa_portfolio_test_page2_20240226_20250422_26_10_13_12_19.png`

This final page contains EWT, EWU, EWY, Gold, and SPY for 2024--2025. It uses
the same index and transaction-volume conventions as the other
asset-specific pages.

* **EWT:** The agent rises from 100 to around 121 by day 170 and stays near
  118--123 until a sharp late decline. Buy-and-hold reaches about 122 near
  day 90, then trends lower and falls to roughly 88 near day 280. Both
  decline late, but the agent finishes near 106 versus about 95 for
  buy-and-hold. The agent's protection over the second half yields
  approximately +10.13% alpha.
* **EWU:** Buy-and-hold rises to around 114 several times, but drops abruptly
  to about 101 near day 275 and recovers to 114. The agent is lower for most
  of the period, reaching about 99 at its mid-period low, then rises sharply
  after day 230 and ends around 120. The late agent surge is the important
  event and gives approximately +5.99% alpha.
* **EWY:** The agent increases from 100 to about 110 by day 145 and then
  accelerates to roughly 142 at the end. Buy-and-hold falls from about 107
  to the low 90s around days 200--215 and ends near 85. This is the clearest
  sustained divergence on the page; it produces the largest overall alpha,
  approximately +56.59%.
* **Gold:** Buy-and-hold rises from 100 to about 125 around day 160 and then
  accelerates to nearly 170 at the end. The agent follows a much flatter
  path, ending around 126. The expanding orange-blue gap after day 120 is
  the principal negative event for the agent and corresponds to
  approximately -41.81% alpha.
* **SPY:** The two series track each other very closely. They rise from 100
  to about 120 around day 200, fluctuate in the 115--121 range, and then
  fall sharply near days 260--280 before ending around 104. The agent is
  marginally below buy-and-hold at the end, giving approximately -0.38%
  alpha. The close overlap indicates that the SAA decisions had little
  asset-specific effect on SPY in this interval.

The page-level ranking emphasises the dispersion of the 2024--2025 results:
EWY is the strongest positive relative contribution, while Gold is by far
the largest negative contribution. The SPY lines are almost indistinguishable
and therefore contribute little to the difference between the agent and the
benchmark.

# Descriptions for run `00312_config_01068_26_10_14`

The following twelve descriptions refer to run `00312_config_01068`, which
uses the same figure layout and conventions as the preceding run. In the four
aggregate figures, the horizontal axis is trading day and the vertical axis
is the portfolio index normalised to 100 at the start (`t_0 = 100`). The blue
series is the equal-weight combined agent portfolio, the red series is SPY
buy-and-hold, and the green series is a 100% cash position earning EFFR. In
the asset-specific figures, the upper plots use the same normalised index:
blue is the agent and orange is buy-and-hold for the named asset. The lower
plot beneath each asset uses the same trading-day axis and shows transaction
volume in dollars; green bars indicate positive purchases or increases in
position and red bars indicate negative sales or reductions.

## `aggregate_saa_portfolio_test_20051216_20070214_26_10_14_22_34.png`

This aggregate graph covers 16 December 2005 to 14 February 2007. The x-axis
runs from trading day 0 to approximately 290 and the y-axis spans roughly 96
to 136 index points. All three series begin at 100. The blue agent portfolio
rises rapidly during the first 25 days, reaches about 105 around day 15, and
then continues upward with several plateaus and short reversals. It reaches
approximately 108 around day 28, 112--113 around day 75, and about 117 near
day 100. A substantial but temporary correction occurs between days 100 and
125: the agent falls from about 117 to a trough near 107 at day 120. This is
the principal agent drawdown event in the graph, but the subsequent recovery
is unusually strong. The blue line returns to roughly 116 near day 135,
crosses 120 around day 155, and then advances almost continuously to about
130 near day 240 and approximately 135 at the end.

The red SPY line is much lower and more volatile. It oscillates around
100--105 during the first half, falls below 100 near day 120, and then begins
a sustained recovery after approximately day 145. It reaches about 110 near
day 230 and finishes near 115.2. The green cash line increases smoothly from
100 to approximately 105.8 and remains below both risky portfolios for most
of the period. The final values correspond to approximately +34.9% for the
agent, +15.2% for SPY, and +5.8% for cash. The widening blue-red gap after
day 150 is the decisive feature: the agent captures a much larger part of the
long upward movement than the SPY benchmark while recovering quickly from its
mid-period decline.

## `aggregate_saa_portfolio_test_20120112_20130320_26_10_14_22_34.png`

This graph covers 12 January 2012 to 20 March 2013. The x-axis again shows
trading day, while the y-axis displays the normalised index from approximately
93 to 122. The green cash line is almost flat at 100, ending only slightly
above it at about 100.2. The blue agent starts at 100 and rises to roughly
105 by day 35. It then fluctuates between about 101 and 105 until around day
75, before falling sharply. The main agent drawdown extends from about day 75
to day 95: the line reaches approximately 94 near day 95, briefly recovers,
and remains below 100 until about day 130. This is important because the
agent is then below both cash and SPY despite its earlier gains.

From approximately day 130 onward, the agent recovers in several steps. It
reaches about 105 near day 150, jumps to approximately 108--109 around day
165, and then remains mostly between 106 and 109 through day 230. It rises
again to about 112 near day 235, stays near 112 through the final quarter,
and ends close to 112.1. SPY follows a stronger path in this period: it
reaches about 109 around day 50, falls to approximately 99 near day 95, and
then climbs steadily after day 130 to finish near 120.2. The graph therefore
shows the agent reducing risk during part of the later advance but missing
the final SPY rally. Reported returns are approximately +12.1% for the
agent, +20.2% for SPY, and +0.2% for cash.

## `aggregate_saa_portfolio_test_20180227_20190423_26_10_14_22_34.png`

This figure covers 27 February 2018 to 23 April 2019. The x-axis extends to
approximately trading day 290 and the y-axis ranges from about 84 to 108.
The agent begins at 100 and rises to around 102--103 during the first 40
days. It then fluctuates around 100--102 through day 155. A sharp decline
starts near day 160, taking the blue line below 98 by day 170 and into the
mid-90s during days 175--200. The agent reaches its lowest visible point,
approximately 89, near day 207. It subsequently recovers to about 95 near
day 220, rises gradually to about 99 around day 245, and finishes near
101.4.

SPY experiences a similar but deeper shock. The red line reaches roughly 107
around day 140, falls sharply after day 155, and reaches a trough near 85.3
around day 208. It then recovers more strongly than the agent, crossing 100
near day 245 and finishing around 106.7. The green cash line rises gradually
from 100 to approximately 102.4. Thus, the agent remains more stable than
SPY during the deepest part of the drawdown, but its weaker recovery leaves it
below both SPY and cash at the end. The displayed returns are approximately
+1.4% for the agent, +6.7% for SPY, and +2.4% for cash.

## `aggregate_saa_portfolio_test_20240226_20250422_26_10_14_22_34.png`

This aggregate graph covers 26 February 2024 to 22 April 2025. The x-axis
shows approximately 290 trading days and the y-axis ranges from about 97 to
122. The cash line rises smoothly from 100 to approximately 105.8. The blue
agent initially rises to around 104 by day 30, falls back toward 101 around
day 40, and then progresses upward in stages. It reaches about 105 near day
80, briefly falls to approximately 100 around day 110, and then climbs to
about 108 by day 145. Between days 150 and 220 it fluctuates mainly between
106 and 109. A renewed rise begins around day 220, taking the agent to about
112 near day 245. It experiences a sharp temporary drop to approximately 102
around day 275, followed by a rapid recovery to about 111 at the end.

SPY rises much further during the middle of the period, reaching roughly 120
around days 195--245. It then undergoes a severe late drawdown: from about
121 near day 245, the red line falls below 100 around day 280. A partial
recovery brings it only to about 104.2 at the end. The agent therefore
finishes well above SPY despite not matching its mid-period peak. The final
reported returns are approximately +11.1% for the agent, +4.2% for SPY, and
+5.8% for cash. The late red collapse and the agent's rebound are the key
events explaining the positive aggregate alpha.

## `asset-specific_saa_portfolio_test_page1_20051216_20070214_26_10_14_22_34.png`

This six-panel figure covers 16 December 2005 to 14 February 2007. Each upper
panel has trading day on the horizontal axis and a normalised index on the
vertical axis; blue is the agent and orange is buy-and-hold. The lower
subpanel beneath each asset uses dollar volume, with green bars above zero
representing purchases or increases and red bars below zero representing
sales or reductions.

* **Crude:** The two series initially move together from 100 to about 117
  around day 25. Buy-and-hold then reaches approximately 130 near day 85,
  while the agent remains closer to 120--123. Both rise again toward 132
  around day 150 before declining sharply. The agent reaches about 106 near
  day 205 and later falls to roughly 95 around day 265, while buy-and-hold
  reaches a lower trough near 87 at about day 270. The agent finishes near
  109 and buy-and-hold near 101, so the agent avoids part of the terminal
  decline. The transaction panel shows frequent two-sided rebalancing,
  particularly during the rise before day 170 and the late decline.
* **EWG:** The agent and benchmark rise together to around 120 near day 100,
  after which the agent separates decisively. The blue line rises through
  approximately 130 near day 150, 140 near day 200, and 150 near day 260,
  finishing near 155. Buy-and-hold falls toward 102 around day 120 and
  recovers only to about 139 at the end. The persistent blue lead after day
  120 is the dominant event. The lower panel contains repeated purchase and
  sale clusters throughout the period, including large positive bars near
  days 150 and 250.
* **EWH:** Both series rise from 100 to about 110--113 near day 90, then
  fall toward 100--103 around day 120. From approximately day 140 onward,
  both trend upward. The agent moves slightly ahead during the later rise,
  reaches about 127 near day 270, and ends close to 131; buy-and-hold ends
  near 129--130. The two paths converge again at the end, so the final
  difference is small. Transactions remain frequent, with alternating
  green and red bars concentrated in the final third.
* **EWJ:** The agent climbs from 100 to roughly 120 by day 100, drops to
  about 100 near day 125, and then recovers rapidly to 118 around day 165.
  It remains near 120 through day 230 and rises further to approximately
  130 at the end. Buy-and-hold reaches about 117 near day 95, falls below
  95 around day 125, and ends near 111. The agent's sustained lead after
  day 150 is the important result; the transaction panel shows intensive
  rebalancing around the large reversal near days 100--140.
* **EWQ:** The agent rises steadily from 100 to approximately 120 around
  day 100, briefly falls back, and then resumes a strong trend. It reaches
  about 130 near day 160, 140 near day 230, and approximately 148 at the
  end. Buy-and-hold reaches only about 118 near day 95 and finishes near
  132. The growing gap after day 140 is the clearest feature of this panel,
  accompanied by repeated purchase and sale bars in the lower chart.
* **EWS:** The agent and buy-and-hold initially follow a similar upward path
  to about 115--120 near day 95. Both then fall toward 103--106 around day
  125. They recover thereafter, with the agent reaching approximately 147
  and buy-and-hold about 150 at the end. The late convergence means that
  buy-and-hold retains a small final advantage despite the agent's strong
  absolute increase. Transaction volume is frequent and substantial during
  the long recovery.

## `asset-specific_saa_portfolio_test_page2_20051216_20070214_26_10_14_22_34.png`

This companion page covers EWT, EWU, EWY, Gold, and SPY over the same period,
using the same normalised index and transaction-volume conventions.

* **EWT:** The agent rises sharply from 100 to about 120 by day 90, briefly
  falls toward 112 around day 120, and then resumes a strong upward trend.
  It reaches approximately 140 around day 230 and finishes near 156. Buy-and-
  hold peaks near 122 around day 95, falls below 100 near day 130, and
  recovers only to about 118 at the end. The widening blue-orange separation
  after day 140 is the principal event and produces the largest positive
  alpha shown on this page, approximately +36.77%.
* **EWU:** Both lines start at 100 and fluctuate around 100--106 during the
  first half. Buy-and-hold briefly reaches about 115 near day 95 and later
  falls toward 101 around day 125. Both series then recover; the agent rises
  in steps to approximately 122, while buy-and-hold reaches about 124 near
  day 250 and finishes near 123. The benchmark therefore has a modest final
  lead, despite the agent's smoother path.
* **EWY:** The agent increases to about 127 around day 95, falls to near 109
  around day 125, and then resumes an upward trend to approximately 132 at
  the end. Buy-and-hold peaks near 122, falls to about 94 around day 125,
  and recovers only to roughly 118. The agent's greater recovery and final
  level create a large positive separation, approximately +14.27% alpha.
* **Gold:** The benchmark spikes above 140 around day 100, then falls to
  approximately 113 near day 125. The agent follows a flatter path, reaching
  about 120 before the correction and then recovering to roughly 140 at the
  end. Buy-and-hold finishes near 133. The agent's avoidance of the extreme
  benchmark spike-and-decline pattern gives approximately +7.54% alpha.
* **SPY:** Both curves show the same broad trajectory: an increase to about
  105 near day 95, a fall toward 98 around day 125, and a sustained recovery
  after day 150. Buy-and-hold finishes near 115 and the agent near 114,
  leaving a small negative agent alpha of about -1.01%. Transaction bars are
  concentrated in several purchase and sale clusters rather than spread
  uniformly across the full period.

The alpha ranking printed on the page is led by EWT, followed by EWJ, EWG,
EWQ, EWY, Crude, and Gold. EWH is close to neutral, while SPY, EWU, and EWS
have small negative relative outcomes.

## `asset-specific_saa_portfolio_test_page1_20120112_20130320_26_10_14_22_34.png`

This page covers 12 January 2012 to 20 March 2013 and contains Crude, EWG,
EWH, EWJ, EWQ, and EWS. The upper panels show the agent in blue and
buy-and-hold in orange on a 100-based index. The lower panels show positive
and negative dollar transaction volume.

* **Crude:** Both series begin near 100 and rise modestly, but buy-and-hold
  reaches approximately 110 near day 25 while the agent remains close to
  102. A sharp joint decline begins near day 75; the agent reaches a deep
  low around 75 near day 110, while buy-and-hold bottoms near 78. The agent
  recovers toward 90 around day 165, falls again near day 210, and finishes
  around 88. Buy-and-hold ends near 94. The agent's deeper and more
  persistent weakness is the main event, resulting in approximately -5.41%
  alpha.
* **EWG:** Buy-and-hold rises to about 117 near day 50, falls below 95 near
  day 105, and then recovers to approximately 126. The agent initially
  follows the benchmark but begins a strong stepwise rise after day 130,
  reaching about 120 around day 165 and approximately 134 at the end.
  The blue lead during the last half of the graph is the main positive
  event and corresponds to approximately +8.71% alpha.
* **EWH:** The benchmark rises to approximately 115 around day 35, falls
  toward 100 near day 95, and then climbs to about 128 before ending around
  122. The agent starts flatter, drops to about 97 near day 95, and then
  catches up during the second half. It ends close to 125, slightly above
  buy-and-hold. The convergence and small final lead produce approximately
  +2.48% alpha.
* **EWJ:** Buy-and-hold rises to about 111 near day 50, falls to roughly 97
  around day 95, and then fluctuates between 98 and 105 before a late rise
  to approximately 117. The agent falls to about 95 near day 95 but later
  holds mostly between 102 and 106, ending near 110. The benchmark's late
  surge overtakes the agent and explains the negative alpha of approximately
  -7.85%.
* **EWQ:** Both series decline from early highs near 108--115 to lows around
  95 near day 120. The agent then rises in stages, reaching about 115 near
  day 165 and ending around 121. Buy-and-hold recovers more unevenly and
  ends around 121 as well. The lines are close at the endpoint, with the
  agent retaining only a small positive alpha of approximately +0.49%.
* **EWS:** Buy-and-hold rises above 110 early and remains near 115--122
  after recovering from a dip around day 100. The agent stays close to 100
  during the first half, falls to approximately 92 near day 95, and
  recovers only to about 107 at the end. The persistent orange-blue gap is
  the most important feature and yields approximately -12.39% alpha.

## `asset-specific_saa_portfolio_test_page2_20120112_20130320_26_10_14_22_34.png`

This page contains EWT, EWU, EWY, Gold, and SPY for the 2012--2013
validation period. Each upper chart uses a normalised index and each lower
chart displays green purchases or increases and red sales or reductions.

* **EWT:** Buy-and-hold rises above 112 in the first 40 days and then falls
  toward 95 around day 125. The agent initially reaches only about 106,
  declines to approximately 94 near day 95, and then recovers in stages.
  Both series rise above 107 late in the period; the agent finishes near
  109 and buy-and-hold near 108. The final convergence and small agent lead
  correspond to approximately +1.48% alpha.
* **EWU:** Buy-and-hold reaches around 109 early, falls to roughly 95 near
  day 100, and then rises to about 112 at the end. The agent has a deeper
  trough near 93 around day 95 but subsequently makes step increases and
  finishes near 112 as well. The two lines nearly overlap at the endpoint,
  giving an almost neutral alpha of approximately +0.19%.
* **EWY:** Buy-and-hold rises to about 113 near day 45, falls to roughly 97
  near day 95, and later reaches about 117 before ending around 109. The
  agent remains near 103 in the first half, falls to about 96 near day 95,
  and recovers to approximately 108. The benchmark's stronger early rise
  and late recovery leave the agent with approximately -2.05% alpha.
* **Gold:** The agent rises to approximately 112 near day 170 and remains
  above the benchmark for much of the second half. Buy-and-hold falls from
  about 108 to below 95 near day 90, later recovers to about 108, and ends
  near 97. The agent ends near 104, so the late blue-orange separation is
  the key positive event, yielding approximately +6.41% alpha.
* **SPY:** Both curves rise from 100 to around 108 near day 50, fall to about
  98 near day 95, and recover during the second half. Buy-and-hold finishes
  near 120 while the agent finishes around 115. The final benchmark lead
  produces approximately -5.51% alpha. The transaction panel shows several
  large positive and negative clusters around the principal reversals.

The ranking block identifies EWG and Gold as the strongest positive
contributors, followed by EWH and EWT. EWQ and EWU are close to neutral,
while EWY, Crude, SPY, EWJ, and EWS have negative relative outcomes.

## `asset-specific_saa_portfolio_test_page1_20180227_20190423_26_10_14_22_34.png`

This page covers 27 February 2018 to 23 April 2019 for Crude, EWG, EWH,
EWJ, EWQ, and EWS. The axes and colour conventions are the same as on the
other asset-specific pages.

* **Crude:** The agent and benchmark track one another closely through the
  first 160 days, rising toward 120 near day 150. They then fall sharply
  together, reaching lows around 72--75 near day 210. The agent recovers
  to approximately 101 and buy-and-hold to about 105 at the end. The common
  crash is the central event; the agent limits the final loss somewhat but
  remains below buy-and-hold.
* **EWG:** The agent stays near 100--102 during the first half and reaches a
  high near 103 around day 145. It then falls to about 89 near day 210,
  recovers through 95, and finishes around 101. Buy-and-hold declines much
  more severely, reaching about 75 near day 210 and ending near 88. The
  sustained blue lead, especially after day 150, gives approximately
  +13.80% alpha.
* **EWH:** Buy-and-hold falls from about 102 to roughly 83 near day 170,
  remains weak through day 220, and recovers to about 103. The agent is more
  stable, fluctuating around 100--105, reaching a low near 89 around day
  170 and ending near 106. The agent's protection during the benchmark
  decline creates approximately +2.81% alpha.
* **EWJ:** The agent begins near 100, reaches about 102 around day 50, and
  then falls to approximately 86 near day 210. It recovers to about 94 at
  the end. Buy-and-hold drops to roughly 80 around day 210 and finishes
  near 90. The agent remains above the benchmark throughout the later
  decline and ends with approximately +4.44% alpha.
* **EWQ:** The agent initially rises to about 107, then falls to roughly
  89 near day 210. Its recovery is strong, reaching approximately 105 at
  the end. Buy-and-hold declines further to about 81 near day 210 and ends
  near 97. The widening late blue-orange gap is the main event and gives
  approximately +8.09% alpha.
* **EWS:** Both series rise modestly early, but buy-and-hold falls sharply
  after about day 75 and remains between 81 and 92 for much of the period.
  The agent fluctuates mostly between 94 and 101, briefly dips near 93, and
  ends around 105. Buy-and-hold ends near 92. The long period of agent
  outperformance produces approximately +12.88% alpha.

Overall, this page shows defensive relative performance across all six
assets, with the strongest positive effects in EWS, EWG, and EWQ. The lower
panels show transaction clusters around the large drawdown near days 160--220
and the subsequent recovery.

## `asset-specific_saa_portfolio_test_page2_20180227_20190423_26_10_14_22_34.png`

This page contains EWT, EWU, EWY, Gold, and SPY for 2018--2019. The upper
panels show normalised asset values, while the lower panels show positive and
negative dollar transaction volume.

* **EWT:** The two lines move closely around 100--103 until approximately day
  155. They then fall sharply, with buy-and-hold reaching about 82 near day
  210 and the agent reaching approximately 90. Both recover afterward; the
  agent ends near 101 and buy-and-hold near 97. The agent's smaller drawdown
  and higher final level produce approximately +3.49% alpha.
* **EWU:** Buy-and-hold rises to about 105 near day 60, falls to roughly 82
  around day 210, and recovers to approximately 97. The agent follows a
  similar path but reaches only about 89 at the trough and ends close to
  97. The two lines converge at the end, resulting in almost neutral alpha
  of approximately -0.10%.
* **EWY:** The agent remains near 100--105 through the first half, falls to
  about 89 around day 170, and recovers to approximately 103. Buy-and-hold
  declines much more persistently, reaching about 79--80 near day 210 and
  ending near 87. The agent's protection from the prolonged benchmark
  decline is the decisive event and gives approximately +15.85% alpha.
* **Gold:** Both series decline from about 102 to approximately 91--93 near
  day 120 and recover after day 170. The agent reaches roughly 104 around
  day 245 and ends near 99, while buy-and-hold ends around 96.5. The agent
  therefore retains a modest advantage, approximately +2.51% alpha.
* **SPY:** The lines broadly overlap: both rise to around 106 near day 150,
  fall sharply near day 210 to the mid-80s or low-90s, and recover above
  104. Buy-and-hold ends slightly above the agent, giving approximately
  -2.59% alpha. Transaction activity is concentrated around the large
  mid-period fall and the later recovery.

The page-level ranking is led by EWY, EWG, and EWS, followed by EWQ, EWJ,
EWT, EWH, and Gold. EWU is effectively neutral, while SPY is slightly
negative.

## `asset-specific_saa_portfolio_test_page1_20240226_20250422_26_10_14_22_34.png`

This page covers 26 February 2024 to 22 April 2025 for Crude, EWG, EWH,
EWJ, EWQ, and EWS. The upper panels show agent and buy-and-hold indices,
while the lower panels show the direction and magnitude of transactions.

* **Crude:** The agent and benchmark initially rise together, reaching around
  110 near day 30. The benchmark then declines toward 88--90 during the
  middle of the period, while the agent remains mostly between 98 and 105
  and reaches about 111 near day 225. Buy-and-hold experiences a severe
  terminal drop below 80 around day 280; the agent also falls late but
  finishes near 96.5. The agent's avoidance of the prolonged benchmark
  decline is the clearest event and produces approximately +12.56% alpha.
* **EWG:** Buy-and-hold rises to approximately 110 around day 150 and then
  accelerates to nearly 130 around day 265. The agent remains lower,
  generally between 98 and 108, before rising to about 115 at the end.
  The agent does not match the benchmark's long late rally, and the final
  orange-blue gap corresponds to approximately -10.56% alpha.
* **EWH:** Buy-and-hold is highly volatile, with a spike above 120 near day
  155, declines toward 92 around day 275, and ends near 102. The agent
  remains mostly between 95 and 108, reaches about 112 around day 200, and
  finishes near 104. The agent's smoother path and slightly higher ending
  value give approximately +2.76% alpha.
* **EWJ:** The agent rises from 100 to about 112 around days 190--260,
  experiences a sharp temporary fall near day 275, and finishes near 107.
  Buy-and-hold remains mostly between 97 and 104, falls to approximately 89
  near day 275, and ends near 100. The sustained blue lead and avoidance of
  the late benchmark trough yield approximately +6.51% alpha.
* **EWQ:** Both lines decline from around 102 to roughly 88--90 near day
  200--220 and then recover. The agent reaches about 102 near day 250,
  falls sharply near day 280, and ends close to 99. Buy-and-hold follows a
  similar late pattern and finishes near 99 as well. The close overlap
  explains the small positive alpha of approximately +0.44%.
* **EWS:** Both strategies trend upward from 100. Buy-and-hold reaches about
  129 near day 200 and approximately 134 around day 260, then drops sharply
  near day 280. The agent rises more steadily to approximately 135 at the
  end, overtaking the benchmark after the late reversal. This crossover is
  the key event and results in approximately +4.28% alpha.

The lower panels show frequent rebalancing, especially for Crude, EWH, EWJ,
and EWS. The page's positive contributors are led by Crude, EWY, EWJ, EWS,
and EWT, while EWG is the principal negative contributor.

## `asset-specific_saa_portfolio_test_page2_20240226_20250422_26_10_14_22_34.png`

This final 00312 page covers EWT, EWU, EWY, Gold, and SPY over 2024--2025.
The upper charts use the 100-based index and the blue-agent/orange-benchmark
colour scheme; the lower charts show transaction volume in dollars.

* **EWT:** Buy-and-hold rises to roughly 122 near day 90 and fluctuates
  between about 110 and 121 for most of the period. The agent remains lower,
  mainly between 103 and 111, before both lines fall sharply near day 275.
  The agent finishes near 99 and buy-and-hold near 95.5. The agent's modest
  protection during the final fall produces approximately +3.03% alpha.
* **EWU:** Buy-and-hold reaches about 114 near days 125--150, falls toward
  102 around day 215, and rises again before a sharp dip near day 275. The
  agent follows a somewhat smoother path, reaches about 112, and then
  surges to approximately 117 at the end. The final agent lead is the main
  positive event and gives approximately +2.76% alpha.
* **EWY:** Both lines rise early, but buy-and-hold declines steadily after
  about day 120, reaching roughly 79 near day 280 and ending near 85.
  The agent remains substantially higher, generally around 95--103, and
  finishes near 93. The persistent blue-over-orange separation, especially
  after day 160, yields approximately +7.78% alpha.
* **Gold:** Buy-and-hold rises almost continuously from 100 to nearly 170 at
  the end. The agent also rises, but more slowly: it reaches about 130 near
  day 170, fluctuates around 126--140, and finishes near 153. The widening
  orange-blue gap in the final quarter is the principal negative event and
  gives approximately -15.09% alpha.
* **SPY:** Both curves rise from 100 to around 120 near day 200, fluctuate
  near their highs, and then fall sharply around days 260--280. The agent
  ends close to 104 and buy-and-hold near 104.2, so the two series are
  almost indistinguishable. The small final difference corresponds to
  approximately -1.01% alpha.

The ranking block shows the largest positive relative result for Crude,
followed by EWY, EWJ, EWS, EWT, EWU, EWH, and EWQ. Gold is the largest
negative contributor because buy-and-hold experiences the strongest late
increase, while SPY is only slightly negative.

# TensorBoard training and validation metrics for the selected SAA runs

The six graphs in this section are stored in the [`src/utils/graphs/` folder](C:/dev/pytrade/src/utils/graphs).
They contain **only SAA training runs** from SAA run IDs **00312, 00306,
00308, and 00314**, corresponding respectively to configuration identifiers
**01068, 01062, 01064, and 01070**. This run provenance is important: future
graphs with similar metric names may be generated for PAA runs, but those
graphs are not represented or described in this section. A graph with a
similar filename must therefore not be assumed to belong to these SAA runs
unless its run IDs are explicitly confirmed.

The horizontal axis in every figure is the training step, shown in units of
approximately one million steps by the scientific-notation multiplier on the
axis. The legend colours are consistent across all six graphs:

- **Blue:** SAA run 00306, configuration 01062.
- **Orange:** SAA run 00308, configuration 01064.
- **Green:** SAA run 00312, configuration 01068.
- **Red:** SAA run 00314, configuration 01070.

The five training graphs contain a faint, highly variable raw trace and a
thicker 25-step centred rolling mean. The rolling mean is the main curve to
use when interpreting the long-term training progression; the faint trace
shows the size of the short-term update noise and occasional outliers.

## `validation_pv_minus_selected_asset_bh_abs_mean.png`

This graph shows the validation metric
`validation/pv_minus_selected_asset_bh_abs_mean`. The x-axis covers
approximately 0 to 3.5 million training steps, while the y-axis ranges from
about -400 to +400. Unlike the training plots, the validation curves are
shown as direct sampled values rather than as a rolling mean. The metric
measures the average difference between portfolio value and the selected
asset's buy-and-hold value according to the logged metric name. Positive
values indicate that the validation portfolio is above the selected
buy-and-hold reference on the logged scale; negative values indicate that it
is below it. Because the plotted values can be negative despite the suffix
`abs_mean`, the graph should be interpreted as the logged scalar as shown,
not as an assumption that the plotted quantity is mathematically
non-negative.

The blue 00306 curve begins around -350, rises through zero near 0.2 million
steps, and reaches roughly +300 around 0.35 million. It then fluctuates
strongly, with repeated peaks between about +150 and +260 and negative
excursions near 1.1, 1.6, 1.85, and 2.3 million steps. Its most prominent
late positive spike is near 2.05 million steps at approximately +260. After
2.4 million, it remains close to zero with alternating positive and negative
values, reaches a late low near -80 around 3.2 million, and finishes slightly
below zero.

The orange 00308 curve starts near -300 and remains negative through roughly
0.45 million steps. It crosses above zero around 0.5 million, rises to about
+180 near 0.8 million, and then falls back through zero around 1.05 million.
Its most distinctive event is a sharp rise beginning near 1.25 million:
the curve reaches approximately +290 around 1.35 million and remains very
high, generally between +110 and +290, until about 2.25 million steps.
This sustained positive interval indicates the strongest validation
advantage among the four runs during the middle of training. The curve
terminates near +180 around 2.3 million steps, earlier than the other runs.

The green 00312 curve begins near -350, rises most rapidly, crosses zero
around 0.15 million, and reaches the overall maximum of approximately +390
near 0.4 million steps. It then declines from about +380 toward +250 by
0.8 million and oscillates around +100 to +200 through roughly 1.35 million.
After a brief positive spike near 1.55 million, it falls below zero around
1.9 million and remains predominantly negative thereafter. It reaches values
around -120 to -130 near 2.5--2.8 million and finishes near -120 around
2.95 million. The early peak and later sustained negative region show a
substantial change in validation behaviour over training.

The red 00314 curve begins near -390 and rises above zero almost immediately.
It remains mostly positive during the first million steps, generally between
about +40 and +220, with a local peak near +220 around 0.8 million. It
falls below zero around 1.25 million and reaches approximately -170 near
1.6 million. The curve later recovers toward zero, then develops a second
positive phase after about 2.8 million steps, reaching approximately +150
near 3.2 million before finishing around +130. The late positive recovery
distinguishes it from the green curve, which remains negative at the end.

The graph therefore does not show a common monotonic validation improvement.
Instead, the runs peak at different stages: 00312 has the highest early
peak, 00308 has the longest middle-training positive period, 00314 improves
again late, and 00306 remains comparatively volatile around zero in the
final part of the plot.

## `train_pv_minus_selected_asset_bh_abs_mean.png`

This graph shows the training counterpart
`train/pv_minus_selected_asset_bh_abs_mean`. The x-axis covers approximately
0 to 3.5 million steps and the y-axis extends from about -1,000 to +3,700.
The faint lines show raw training measurements, while the thicker lines show
the 25-step centred rolling means. All four rolling means begin slightly
below zero and then rise, so the broad training pattern is substantially
more positive than the early validation samples.

The blue 00306 rolling mean increases from roughly -150 at the beginning to
about +700 by 1.2 million steps. It climbs more strongly around 1.9--2.1
million, reaching approximately +1,500, then fluctuates between about +1,400
and +1,800 through 3.1 million. A further rise near 3.2--3.4 million takes
it to about +2,100 at the end. The raw blue trace is very noisy throughout,
with repeated short-lived spikes and drops around the rolling mean.

The orange 00308 rolling mean rises from approximately -200 to around +600
by 1.0 million steps. It remains the lowest of the four curves for much of
the middle training period, fluctuating around +700 to +1,000 through about
2.0 million. It increases to roughly +1,300 near 2.3 million, then ends
around +1,400 near 2.35 million. Its shorter horizontal extent indicates
that this run has fewer logged training steps in the plotted data than the
other three runs.

The green 00312 rolling mean rises fastest initially, reaching approximately
+400 by 0.4 million and about +700 near 0.9 million. It continues upward to
around +1,300 near 1.8 million and reaches roughly +1,650 around 2.6--2.8
million. It then levels off and finishes near +1,750 around 3.0 million
steps. The green raw trace includes large positive and negative deviations,
but the rolling mean remains positive and relatively smooth.

The red 00314 rolling mean starts close to -200, rises steadily through the
first 1.5 million steps, and reaches approximately +1,400 near 1.6 million.
It dips modestly around 2.0 million but then resumes its rise. From about
2.3 million onward it is the highest curve, staying near +2,000 to +2,200
and finishing around +2,500 at 3.35 million steps. The red curve therefore
has the strongest late training metric, while the green curve is strongest
among the longer-running curves during the earlier and middle portions.

The graph's main interpretation is that training values generally become
more positive for every run, but the ordering changes over time. Early
progress is led by 00312, the middle is led by 00312 and then 00314, and the
late period is led clearly by 00314. The raw traces also demonstrate that
individual logged updates are much noisier than the underlying rolling
trend.

## `train_approx_kl.png`

This graph shows `train/approx_kl`, an approximate Kullback--Leibler
divergence used to monitor how far the updated policy moves from the previous
policy. The x-axis spans roughly 0 to 3.5 million steps and the y-axis
ranges from zero to approximately 0.028. Values are positive and small
throughout, so the graph indicates policy-update magnitude rather than
portfolio return. The thick curves are 25-step rolling means; faint spikes
are raw observations.

The blue 00306 rolling mean starts near 0.0008, rises gradually to about
0.0025 around 1.0 million, and reaches a modest maximum near 0.0035 around
1.6 million. It then declines almost continuously, falling below 0.0025
around 2.2 million and reaching approximately 0.0010 at the end near
3.45 million. This is the smallest late KL level, indicating increasingly
small policy changes late in training.

The orange 00308 curve begins around 0.0005 and rises to approximately
0.0035 near 1.3 million. It remains close to 0.0030--0.0035 through about
2.3 million before ending near 0.0032. The raw orange trace contains
isolated spikes, including values close to 0.01 around 1.1--1.3 million, but
these do not materially change the rolling trend.

The green 00312 curve starts highest, near 0.0012, and reaches about 0.0030
around 0.6 million. It rises to approximately 0.0055 near 1.7 million and
briefly reaches about 0.0063 around 2.0 million. It then declines gradually,
ending around 0.0035 near 3.0 million. The raw trace includes large isolated
spikes, including one near 0.020 around 1.95 million, which is substantially
above the rolling trend.

The red 00314 curve starts near 0.0008 and rises to approximately 0.0040 by
1.3 million. It experiences its clearest increase around 1.6--1.8 million,
peaking near 0.0072. It then settles around 0.0055--0.0060 through about
2.7 million, declines gradually after 2.8 million, and finishes near 0.0042
at 3.35 million. A raw spike near 1.65 million reaches about 0.027, the
largest isolated observation in the graph.

The four curves therefore show moderate policy-update growth during the
first half of training followed by stabilisation or decline. 00312 and 00314
reach the highest rolling KL levels, while 00306 ends with the smallest
policy-update magnitude. The isolated spikes are transient raw events and
should not be mistaken for the sustained policy behaviour represented by the
rolling means.

## `train_explained_variance.png`

This graph shows `train/explained_variance`, which indicates how well the
value function explains the observed return variation. The x-axis spans
approximately 0 to 3.5 million steps and the y-axis ranges from about 0 to
0.9. Higher values indicate better value-function fit on this diagnostic.
The plotted thick curves are 25-step rolling means, with faint raw traces
showing minibatch-level variability.

All four runs improve substantially. The blue 00306 rolling mean starts near
0.25, reaches about 0.38 around 0.2 million, and then rises through roughly
0.5 near 0.6 million and 0.65 near 1.1 million. It continues to around
0.78 near 1.7 million and approximately 0.88 by 2.6 million. It finishes
near 0.90 around 3.5 million steps.

The orange 00308 curve begins near 0.22 and initially lags the other runs,
remaining around 0.3--0.35 through approximately 0.4 million. It then
increases steadily, reaching about 0.55 around 0.8 million, 0.70 near
1.5 million, and 0.80 around 2.0 million. It ends close to 0.88 at roughly
2.35 million steps. The raw orange observations are particularly scattered
early and during the transition from 0.5 to 1.5 million.

The green 00312 curve has the fastest early improvement. It rises from about
0.23 to 0.40 near 0.25 million and approximately 0.62 near 0.55 million.
It reaches about 0.70 near 1.2 million and approximately 0.84 around
1.8 million. It remains one of the highest curves through about 2.5 million,
then levels near 0.89 and ends around 0.90 at 3.0 million.

The red 00314 curve begins near 0.22, rises to about 0.50 around 0.5 million,
and then follows a steady upward path. It reaches approximately 0.70 near
1.2 million, 0.80 around 1.6 million, and about 0.88 around 2.3 million.
It finishes near 0.89 at 3.35 million. The four curves converge near
0.88--0.90 late in training, indicating broadly similar final value-function
fit despite different early learning speeds.

The important event is the common transition from low and noisy explained
variance to a stable high level. The graph suggests that value estimation
improves across training for all four SAA configurations, with 00312
learning fastest initially and 00306 eventually reaching the highest or
near-highest final level.

## `train_policy_gradient_loss.png`

This graph shows `train/policy_gradient_loss`. The x-axis spans approximately
0 to 3.5 million training steps and the y-axis ranges from about -0.005 to
+0.004. The metric is negative for most of the rolling means after the
initial phase. As in the other training plots, faint lines are raw values
and thick lines are 25-step rolling means. This diagnostic should be read as
an optimisation signal, not as a direct return measure: changes in magnitude
and sign describe the scale and direction of policy-gradient updates.

The blue 00306 rolling mean starts near -0.0005, falls to about -0.0013
around 0.3 million, and then recovers toward approximately -0.0007 near
1.2 million. It briefly becomes less negative around 1.5--1.7 million,
then declines to about -0.0015 around 2.4 million. It remains near
-0.0014 to -0.0016 through the end, making it the least negative late curve.

The orange 00308 rolling mean begins near -0.0005, falls to about -0.0012
by 0.3 million, and becomes less negative around 1.0--1.2 million, reaching
approximately -0.0003. It then declines again to around -0.0010 near
1.8 million and ends near -0.0018 at approximately 2.35 million. The raw
orange trace has several positive spikes, including values above +0.004,
but the rolling signal stays near or below zero.

The green 00312 curve drops most strongly. It begins near -0.0008, reaches
about -0.0018 near 0.3 million, and continues down to approximately -0.0032
around 1.8 million. It briefly becomes less negative near 2.0 million, then
falls again to around -0.0037 at 2.5 million and approximately -0.0040
near 2.9 million. The green raw trace is highly variable, with several
negative excursions below -0.005 and positive outliers above +0.004.

The red 00314 curve begins near -0.0005 and reaches about -0.0015 by
0.5 million. It remains near -0.0016 to -0.0020 through 1.5 million,
becomes more negative near 2.0 million, and fluctuates around -0.0025 to
-0.0030 from 2.3 million onward. It finishes close to -0.0031 at 3.35
million. Thus, 00312 has the most negative late policy-gradient signal,
00314 is intermediate, and 00306 is least negative. The raw spikes show
that the underlying updates remain noisy even when the rolling means are
relatively smooth.

## `train_value_loss.png`

This graph shows `train/value_loss`. The x-axis covers approximately 0 to
3.5 million steps and the y-axis ranges from about 0.1 to 0.7. Lower values
indicate a smaller value-function loss on the plotted training diagnostic.
The thick lines are 25-step rolling means and the pale traces are raw
observations, which contain substantial early outliers.

The blue 00306 rolling mean starts around 0.33, falls to approximately 0.29
near 0.2 million, and then rises to a local maximum near 0.38 around
0.4 million. It declines gradually thereafter, reaching roughly 0.27 near
1.5 million, 0.20 around 2.2 million, and approximately 0.15 near
2.8 million. It ends near 0.14 at 3.5 million, the highest final loss of
the four runs but still substantially below its early peak.

The orange 00308 curve starts near 0.32, rises to about 0.37 around
0.4 million, and then falls steadily. It reaches approximately 0.25 near
0.8 million, 0.19 around 1.5 million, and about 0.14 near 2.0 million.
It remains around 0.14--0.16 through the end of its plotted range at about
2.35 million, with a slight late increase. It has the lowest loss for much
of the middle training interval.

The green 00312 rolling mean begins near 0.38 and falls rapidly to about
0.28 near 0.3 million. It remains close to 0.25--0.30 through 1.2 million,
then decreases to approximately 0.19 near 1.5 million. After a modest
rebound around 2.0 million, it declines again to about 0.13 near 2.8
million and finishes around 0.12 at 3.0 million. Its final rolling loss is
the lowest of the longer curves.

The red 00314 curve starts around 0.34, remains near 0.33 early, and falls
to approximately 0.28 around 0.5 million. It then rises to a distinct local
maximum of about 0.35 around 1.3 million, which is the most prominent
mid-training deterioration among the rolling means. It subsequently falls
to roughly 0.22 near 1.9 million, 0.16 around 2.5 million, and about 0.12
at 3.2 million. It finishes close to 0.13 at 3.35 million.

The common pattern is a reduction in value loss over training, despite
temporary increases and noisy raw observations. The green and red runs end
with the lowest losses, orange has the lowest loss through much of the
middle, and blue decreases more gradually and finishes slightly higher.

## `validation_excess_return_over_spy_abs_mean.png`

This graph is a **PAA validation graph**, not an additional SAA graph. It
contains the hierarchical PAA runs 00315, 00316, 00317, and 00318, with
configuration identifiers 10027, 10028, 10029, and 10030 respectively. The
legend colours are blue for `config_10027`, orange for `config_10028`, green
for `config_10029`, and red for `config_10030`. This is consistent with the
PAA runs described later in this document and should not be confused with
the SAA runs 00312, 00306, 00308, and 00314 described in the preceding
TensorBoard section.

The horizontal axis is the training step, extending from zero to about
18 million steps. The vertical axis is the logged
`validation/excess_return_over_spy_abs_mean` value. It ranges from
approximately -13,000 at the bottom to about +9,000 at the top. Positive
values indicate a positive validation excess-return metric relative to the
SPY reference on the logged scale; negative values indicate an adverse
relative result. The scale is a scalar diagnostic rather than a percentage
axis, so values such as +5,000 should not be read directly as a 5,000%
return. The validation traces are plotted directly, without the faint raw
trace and 25-step rolling mean used for the training metrics.

The blue `config_10027` curve is visible from the beginning to approximately
9.3 million steps. It starts near -1,200, rises gradually toward zero by
roughly 2 million steps, and crosses into positive territory around
2.5--3 million steps. It then climbs strongly, reaching about +2,000 near
3.5 million and a first major high of approximately +3,000--3,500 around
4 million steps. Between roughly 4 and 6 million steps it remains positive
but volatile, repeatedly falling and recovering. A particularly sharp
positive spike appears near 5.3 million steps at around +4,000, followed
by a decline toward approximately +1,000 around 6.2 million. The curve
then oscillates between roughly zero and +3,000 through 8 million steps.
After a final rise near 8.5--9 million, it finishes near +4,000 to +4,300.
Thus, this run develops a sustained positive validation metric after an
initial negative phase, although its path contains substantial reversals.

The orange `config_10028` curve starts close to -1,200 and remains near
-1,000 during the first 2.5--3 million steps. It briefly approaches zero
and becomes slightly positive around 3.2--3.8 million steps, but this
improvement is not maintained. The most important deterioration begins
around 4.1 million steps: the curve falls to approximately -5,000 near
4.6 million, briefly rebounds to a positive spike near +700 around
5.4 million, and then drops sharply again. Around 5.8--6.2 million steps
it reaches approximately -10,000 to -11,000. The trough is near -12,000
around 7.0--7.5 million steps. It recovers partially afterward, fluctuating
mostly between about -7,000 and -10,000, but remains strongly negative
through its final point near 9.3 million steps. This is the weakest curve
in the graph and shows no sustained transition to positive validation
performance.

The green `config_10029` curve is the longest-running trace, continuing to
approximately 18 million steps. It begins near -1,200 and initially
declines slightly, reaching around -1,500 by 2 million steps. It then
falls further, with values around -2,500 to -3,500 between approximately
3.5 and 5.5 million steps. The curve remains highly unstable through the
middle of training: it repeatedly moves between roughly zero and -3,000,
with a deeper trough near -4,500 around 9 million steps. The major
turning point occurs after about 9.5 million steps. The curve rises rapidly,
crosses zero near 10 million, and accelerates to approximately +4,000 by
10.8 million and +6,000--7,000 around 11.3--12 million steps.

After this breakthrough, the green curve remains strongly positive but
continues to fluctuate. It reaches a local high near +7,500 around
11.5--12 million, falls back toward roughly +5,000 near 12.8 million,
and then oscillates between about +5,000 and +7,000 through 14 million.
It rises again to approximately +8,000 around 14.5--15 million steps,
which is its highest sustained region. From there it gradually declines
but stays positive, ending near +7,500 at 18 million steps. The important
interpretive event is the delayed transition from a long negative and
unstable phase to a sustained high positive phase. This late improvement
distinguishes config 10029 from the other three curves and is consistent
with its later training extent.

The red `config_10030` curve begins around -1,200 and rises more quickly
than the orange curve during the first 3 million steps. It reaches a small
positive peak of roughly +300 near 2.5--3 million steps, then falls back
through zero around 3.5 million. From approximately 4 to 5.5 million it
declines toward -2,000, with several short-lived reversals. It reaches
values near -2,500 around 5.5--6 million and fluctuates around -1,500 to
-2,500 through roughly 7 million. A further deterioration occurs between
7 and 8.5 million steps, where it reaches approximately -4,000. It then
recovers somewhat but remains negative, ending around -3,000 at about
9.3 million steps. The red run therefore shows an early positive attempt
followed by a persistent negative validation result.

The graph’s main comparison is between the eventual positive outcomes of
the blue and green curves and the negative outcomes of the orange and red
curves. Config 10027 reaches positive values earlier and ends around
+4,000, whereas config 10029 remains negative for much longer but later
reaches the highest and most sustained positive level, around +7,500 to
+8,000. Configs 10028 and 10030 experience substantial negative phases and
finish below zero. The different horizontal extents also matter: the blue,
orange, and red traces end near 9.3 million steps, while the green trace
continues to 18 million. Consequently, the graph compares both validation
behaviour and different observed training durations; the late green
advantage should be described as a late-training result rather than as a
directly time-matched advantage at every step.

# Validation and test graphs for the best hierarchical PAA run

The following section describes the PNG figures in
[`src/agents/PPO_portfolio_allocator_weights/saved_models/00317_config_10029_26_10_15/val_and_test_inference/`](C:/dev/pytrade/src/agents/PPO_portfolio_allocator_weights/saved_models/00317_config_10029_26_10_15/val_and_test_inference).
This is the **best hierarchical PAA run**, run ID **00317**, configuration
**10029**, using the frozen SAA signal from SAA run **00312** with
configuration **01068**. The checkpoint is
`best_model_excess_over_spy_abs`. The inference graphs use deterministic
policy evaluation and begin each period from a 100% cash position, as
recorded in the accompanying metadata.

There are **five PNG files** in this directory: four validation figures
(`val_00` through `val_03`) and one test figure (`test_00`). The directory
also contains a metadata JSON file and a daily CSV; those supporting files
are not additional PNG graphs. The five figures all have the same three-panel
layout:

1. **Cumulative performance and daily alpha.** The upper plot has trading day
   on the x-axis and cumulative return relative to the start on the y-axis.
   The black line is the PAA portfolio, blue is SPY buy-and-hold, green is
   equal-weight buy-and-hold across the investable assets, dashed magenta is
   the equal-weight SAA portfolio, and the grey dotted line is 100% cash
   earning EFFR. The lower part of this panel shows daily PAA alpha relative
   to SPY in basis points: green bars are positive daily excess performance
   and red bars are negative daily excess performance.
2. **Portfolio composition after daily rebalancing.** This stacked area chart
   uses the same trading-day x-axis and a 0--100% portfolio-weight y-axis.
   Each coloured band represents the post-rebalancing weight in one asset or
   cash. The legend identifies Cash, Crude, EWG, EWH, EWJ, EWQ, EWS, EWT,
   EWU, EWY, Gold, and SPY. A wide band means that the allocator is
   concentrating in that asset; a thin or absent band means little or no
   allocation.
3. **Trade-signal velocity and rebalancing intensity.** The upper subpanel
   plots daily turnover as a percentage of net asset value. Green bars denote
   risk-on changes that increase total equity exposure, red bars denote
   de-risking into cash, and grey bars denote reallocations between risky
   assets without a net change in equity exposure. The narrow heat strip
   labelled `Delta exp.` encodes the daily change in equity exposure: green
   means an increase, red means a decrease, and pale colours indicate a small
   change. The colour scale is clipped at the 95th percentile, so the most
   extreme changes are visually capped.

The PAA and SAA curves should be distinguished carefully. The dashed magenta
SAA equal-weight line is the performance of the upstream SAA signal, whereas
the black PAA line is the hierarchical allocator that uses that signal and
decides the portfolio-level allocation. Consequently, a black line above the
magenta line indicates value added by the PAA allocation layer relative to
the equal-weight SAA portfolio. The daily alpha bars are not the same as the
cumulative return gap: they show the day-by-day PAA-minus-SPY contribution,
while the upper curves show the accumulated result.

## `best_model_excess_over_spy_abs__validation_val_00_3panel.png`

This figure covers validation block `val_00`, from 16 December 2005 to
14 February 2007, over 289 trading days. The upper cumulative-return axis
runs from slightly below -8% to above 32%. All strategies start at zero.
The black PAA line rises rapidly to roughly 7--8% by trading day 15 and
briefly reaches about 9% near day 30. It then experiences a substantial
drawdown: the line falls below zero around days 40--45 and reaches a trough
near -5% around day 40 before recovering. From approximately day 55 to day
100, PAA rises again toward 16--18%, but it remains volatile, with repeated
peaks and pullbacks.

The strongest intermediate PAA advance occurs around days 125--165, when the
black line rises from roughly 8% to a local maximum near 23%. This lead is
not sustained. The portfolio retreats to approximately 15% around day 180,
then fluctuates around 12--16% through day 245. During the final quarter,
the black curve declines sharply: it falls from roughly 15% near day 245 to
about 1% around day 270, briefly approaches zero, and recovers only to about
8.8% at the end. This late drawdown is the most important negative event in
the period because it removes most of the PAA gains accumulated earlier.

SPY, the blue line, is comparatively weak during the first half. It stays
near zero through about day 80, becomes slightly negative around day 120,
and then recovers steadily. It ends at approximately 15.2%, above the PAA
portfolio. Equal-weight buy-and-hold, the green line, has a stronger
long-run outcome than PAA: it rises to around 18% early, fluctuates between
roughly 8% and 15% later, and finishes near 24.6%. The dashed magenta SAA
equal-weight line is the strongest benchmark in this period. It rises above
10% around day 80, reaches roughly 20% near day 160, and ends around 34.9%.
The grey cash line increases gradually to approximately 5--6%.

The final values are approximately **+8.78% for PAA, +15.22% for SPY,
+24.58% for equal-weight buy-and-hold, +34.87% for SAA equal-weight, and
+5.8% for cash**. Thus, PAA beats cash but underperforms every risky
benchmark shown. Its final alpha is about **-6.44 percentage points versus
SPY** and **-26.09 percentage points versus SAA equal-weight**.

The daily-alpha bars explain the path. Positive green bars are concentrated
in the early rise and around the recovery near days 120--170, but red bars
become more frequent during the middle and final portions. Large negative
bars around days 35--45, 110--130, and 255--275 coincide with the visible
PAA drawdowns. The final segment contains repeated red bars, including some
near -200 basis points, showing that the late loss was caused by a sequence
of adverse days rather than one isolated observation.

The composition panel shows highly active and concentrated allocation. Crude
frequently becomes the dominant brown band, at times occupying most of the
portfolio, especially around days 30--55, 125--150, 205--225, and 255--270.
Gold also becomes prominent during several transitions, while EWY, EWJ,
EWH, EWQ, EWS, EWT, EWU, and SPY appear as narrower alternating bands. Cash
is generally a small bottom band, but it widens around de-risking episodes.
The allocator therefore does not maintain a stable diversified mixture; it
rotates rapidly between concentrated asset exposures.

The bottom turnover panel confirms this behaviour. Large green risk-on bars
appear near the beginning, around days 90--110, 145--175, and 215--245.
Large red de-risking bars occur around days 105--125, 165--180, and
255--280. Grey reallocation bars are also frequent. The heat strip alternates
rapidly between green and red, indicating repeated changes in total equity
exposure. In discussion terms, this period demonstrates that the PAA layer
was capable of generating large tactical moves, but those moves did not
translate into superior terminal performance and exposed the portfolio to a
damaging late drawdown.

## `best_model_excess_over_spy_abs__validation_val_01_3panel.png`

This figure covers validation block `val_01`, from 12 January 2012 to
20 March 2013, over 289 trading days. The upper y-axis spans approximately
-8% to 20%. PAA starts at zero and rises quickly to about 12% by day 30.
It reaches a local high near 14% around day 35, then declines gradually to
approximately 9--10% by day 70. A major drawdown follows between about days
75 and 110: the black curve falls through zero, reaches roughly -6% near
day 95, and remains around -4% to -6% until approximately day 120. This
is the dominant adverse event in the graph.

After day 120, PAA recovers sharply. It returns to zero around day 130,
climbs above 10% by approximately day 155, and reaches about 15--17% near
days 165--190. It then fluctuates between roughly 10% and 16% through day
220. The final third is strong: PAA rises above 18% around day 235, reaches
approximately 21% near day 260, and finishes at about 21.1%.

SPY starts more slowly, remains below PAA during the early rise, and reaches
approximately 10% around day 150. It suffers a drawdown around days 190--215
but then recovers strongly, finishing at about 20.2%. Equal-weight buy-and-
hold rises to approximately 11% early, falls below zero around day 110, and
ends near 13.3%. The dashed SAA equal-weight line remains much lower than
PAA for most of the period, ending around 12.1%. Cash remains close to zero,
ending at approximately 0.2%.

The final PAA result is approximately **+21.11%**, compared with **+20.20%
for SPY, +13.27% for equal-weight buy-and-hold, +12.08% for SAA equal-weight,
and approximately +0.2% for cash**. PAA therefore beats SPY by about
**+0.90 percentage points**, beats SAA equal-weight by about **+9.03
percentage points**, and substantially outperforms the non-adaptive
benchmarks. The important qualification is the mid-period drawdown: the
terminal success follows a loss below zero and a prolonged recovery rather
than a smooth upward path.

Daily alpha is positive more often during the recovery from about day 120
through day 190 and again during the final climb. The largest negative red
bars appear around the main drawdown near days 85--115. Green bars become
more frequent around the transition back above zero and during the late
outperformance over SPY. This alignment between positive daily alpha and
the upward black curve supports the interpretation that the final advantage
was accumulated through repeated favourable decisions rather than one
single terminal jump.

The composition chart shows repeated concentration cycles. Crude is often
the largest holding and forms broad brown plateaus around days 70--85,
125--145, and 210--235. Gold occupies large areas during several defensive
or transitional intervals. EWY, EWJ, EWS, EWQ, EWH, EWT, EWU, and SPY
appear in rapidly changing layers, while cash is usually limited to a
small base. During the drawdown around days 75--110, the portfolio rotates
among assets rather than remaining continuously concentrated in one
position, which is consistent with the frequent turnover and alternating
exposure changes below.

Risk-on green turnover bars are especially visible early, near days 120--160,
and around days 230--260. Red bars are frequent during the drawdown and
around later reversals, while grey reallocation bars become prominent in
the second half. The heat strip shows a dense sequence of both positive and
negative exposure changes. This indicates that the allocator actively
managed risk and eventually recovered the early losses, but at the cost of
substantial trading intensity.

## `best_model_excess_over_spy_abs__validation_val_02_3panel.png`

This figure covers validation block `val_02`, from 27 February 2018 to
23 April 2019, over 290 trading days. The upper y-axis runs from roughly
-20% to 45%. PAA starts near zero, dips slightly negative during the first
few days, and then rises to approximately 10--13% by day 35. It remains
between about 10% and 20% through much of the first half, with a local high
near 21% around days 85--100. The black line then fluctuates around
15--20% through day 155 before declining to approximately 10% around day
205. A brief low near 2% occurs around day 215, after which the strongest
PAA advance begins.

From approximately day 220 onward, PAA rises persistently. It reaches about
17% near day 235, 25% around day 250, and 30% near day 265. The final
advance accelerates further, taking the black curve to approximately 45% at
the end. This is the largest and most sustained PAA success among the four
validation periods. Unlike val_00, the late increase is not erased by a
terminal drawdown.

SPY remains close to zero during the first half, rises modestly after day
140, and experiences a sharp decline near day 210. It recovers only to about
6.7% by the end. Equal-weight buy-and-hold is negative for much of the
period, reaches a trough near -18% around day 210, and ends around -3.8%.
The SAA equal-weight line stays near zero or slightly positive and finishes
around +1.4%. Cash rises only slightly, to approximately +2.4%.

The final values are approximately **+44.84% for PAA, +6.71% for SPY,
-3.82% for equal-weight buy-and-hold, +1.41% for SAA equal-weight, and
+2.4% for cash**. PAA therefore beats SPY by approximately **+38.12
percentage points** and SAA equal-weight by approximately **+43.42
percentage points**. This period provides the strongest evidence of
successful hierarchical allocation in the figures: the PAA layer remains
positive through a difficult benchmark period and compounds a major lead in
the final quarter.

The daily-alpha panel is mostly positive during the long PAA rise, especially
around days 20--45, 80--100, and 220--285. Red bars become larger and more
frequent near days 170--215, matching the temporary reduction in cumulative
PAA performance. The late sequence contains many positive green bars and
only intermittent negative bars, which explains the steep final increase.
The result is therefore supported by broad positive daily alpha over the
late period, not merely by one isolated outlier.

The composition chart shows the allocator exploiting concentrated exposures.
Crude is repeatedly dominant, especially during broad plateaus around days
70--85, 125--145, and 205--235. Gold forms large bands during several other
intervals, and EWY, EWJ, EWQ, EWS, EWH, EWT, EWU, and SPY enter as narrower
rotating positions. Cash is generally small but expands during selected
de-risking episodes. The sharp late PAA rise coincides with sustained
concentration in a limited set of assets rather than an equal allocation
across the universe.

Turnover is high throughout. Green risk-on bars are particularly large around
days 65--75, 105--120, 145--155, and 200--220, while red de-risking bars
appear around major reversals. Grey reallocations are also substantial,
showing that the strategy frequently changes the composition of risky
exposure even when total equity exposure is stable. The heat strip is densely
coloured across the whole period. For final discussion, this graph supports
both a performance claim and a caveat: PAA achieved exceptional excess
return, but it did so through active, concentrated, and high-turnover
allocation.

## `best_model_excess_over_spy_abs__validation_val_03_3panel.png`

This graph covers validation block `val_03`, from 26 February 2024 to
22 April 2025, over 290 trading days. The upper y-axis extends from about
-2% to 22%. PAA starts at zero, rises to roughly 7--8% by day 30, and then
oscillates between approximately 4% and 9% through day 70. It reaches a
local high near 13% around day 95, declines toward 4% around day 115, and
then recovers to about 17% near day 135. The black line remains volatile
through the middle, alternating between approximately 5% and 17%.

Around day 190, PAA is near 9%; it then rises to roughly 13% near day 230
and about 17% around day 245. A sharp late reversal follows: the black line
falls below zero near day 260 and reaches approximately -1% around day 265.
It recovers to around 8% near day 275, but then falls again and finishes
near 5.3%. This late sequence is the crucial limitation of the period:
PAA generated sizeable gains before the final quarter but surrendered most
of its advantage during the final drawdown and recovery.

SPY rises steadily from zero to approximately 10% by day 120, reaches about
20% near day 200, and then declines sharply in the final quarter before
ending near 4.2%. Equal-weight buy-and-hold follows a similar path, reaches
approximately 20% near day 200, and finishes near 9.9%. The SAA equal-weight
line ends higher than PAA at approximately 11.1%, while cash rises smoothly
to around 5.8%.

The final values are approximately **+5.31% for PAA, +4.19% for SPY,
+9.89% for equal-weight buy-and-hold, +11.14% for SAA equal-weight, and
+5.8% for cash**. PAA beats SPY by about **+1.12 percentage points**, but
underperforms SAA equal-weight by roughly **-5.83 percentage points** and
equal-weight buy-and-hold by about **-4.58 percentage points**. The result
is therefore mildly positive relative to the market benchmark but not
superior to the other portfolio construction references.

Daily alpha is positive during several early and middle intervals, but red
bars become more visible around days 185--220 and especially around the
late collapse near days 250--270. The bar pattern demonstrates why the
cumulative curve falls so quickly: negative daily excess returns cluster
around the period in which the PAA portfolio loses its earlier lead.

The composition panel displays repeated concentration in Crude, Gold, EWY,
and other individual assets. Crude forms broad brown bands during several
intervals, while Gold becomes prominent around the first and middle
portions. The portfolio repeatedly shifts between concentrated risky
positions and more diversified mixtures. Cash remains a relatively small
band except during de-risking. The turnover panel contains repeated large
green, red, and grey bars, with especially intense activity around the late
reversal. This suggests that the final underperformance was not caused by
inaction: the allocator continued to trade actively, but its late risk
changes did not preserve the earlier gains.

## `best_model_excess_over_spy_abs__test_test_00_3panel.png`

This is the held-out test figure for test block `test_00`, covering 21 July
2025 to 6 August 2026 over 252 trading days. The upper y-axis spans roughly
-5% to 125%. PAA begins close to zero, initially fluctuates slightly below
zero, and then rises gradually to about 10% by day 60. It remains near
15--25% through approximately day 140. The most important event begins near
day 145: PAA accelerates sharply from roughly 25% to around 60% by day 150,
then reaches about 95% near day 165. It remains close to 100% through day
190, rises above 110% around day 205, and reaches a maximum near 124% around
day 205--215.

After the peak, PAA experiences a substantial but incomplete drawdown. The
black line falls from approximately 120% to about 85% around day 220, then
continues downward through the 70% range. It stabilises around 65--70% for
much of the final quarter and finishes near 55.3%. Despite this decline,
the ending value remains far above all comparison portfolios. The test graph
therefore demonstrates both exceptional upside capture and significant
path risk: a reader should not interpret the final +55% as a smooth,
low-volatility accumulation of returns.

SPY rises gradually to about 10% by day 120, briefly falls close to zero
near day 150, and then recovers to approximately 20--22% by the end.
Equal-weight buy-and-hold rises to roughly 25% around day 190, reaches about
40% near day 215, and finishes close to 32%. The SAA equal-weight line
reaches approximately 30--33% and ends near 32.5%. Cash remains almost flat,
ending at approximately 2--3%.

The final values are approximately **+55.30% for PAA, +22.22% for SPY,
+31.80% for equal-weight buy-and-hold, +32.50% for SAA equal-weight, and
approximately +2.5% for cash**. PAA therefore finishes about **+33.08
percentage points above SPY**, **+22.80 points above SAA equal-weight**, and
+23.50 points above equal-weight buy-and-hold. This is the strongest
out-of-sample evidence in the set that the hierarchical allocator can add
substantial value over both the market benchmark and the upstream SAA
portfolio.

The daily-alpha bars are mostly small near the beginning, then become much
larger and predominantly positive around the acceleration near days 135--170.
This cluster includes the largest positive green bars in the test figure.
Negative red bars also become larger during the subsequent correction,
especially around days 150--175 and 205--230, and again near the final
decline. The cumulative result is therefore driven by a concentrated burst
of strong excess performance followed by a large giveback, rather than by
uniform outperformance on every day.

The composition chart shows a highly concentrated regime during the main
profit-generating phase. Crude becomes a very broad brown band around days
70--100, 125--145, and 175--220, often accounting for most of the portfolio.
Gold is prominent during earlier and transitional periods, while EWY, EWJ,
EWS, EWQ, EWH, EWT, EWU, and SPY form rotating secondary exposures. Cash
usually remains a small bottom band but expands during transitions. The
large PAA rise around days 145--210 coincides with sustained concentration,
which provides a plausible visual explanation for both the exceptional gains
and the subsequent drawdown when the concentrated exposure reverses.

The turnover panel shows repeated high-intensity risk-on, de-risking, and
reallocation events. Large green bars appear around the beginning, near
days 70--110, and around the main acceleration. Large red bars become
especially visible during the fall from the peak and near the end, while
grey bars indicate substantial reallocations without changing total equity
exposure. The exposure-change heat strip is densely coloured during the
first 230 days but becomes much paler late in the test period, indicating
smaller net changes in equity exposure after the main adjustment. In a final
trading-performance discussion, this figure supports a strong out-of-sample
performance conclusion while also requiring explicit acknowledgement of
concentration, turnover, and drawdown risk.

## Cross-period interpretation of the PAA graphs

Taken together, the four validation periods and the held-out test period
show regime-dependent but generally adaptive behaviour. PAA underperforms
the strongest upstream SAA result in the older 2005--2007 period, slightly
beats SPY in 2012--2013, produces exceptional excess performance in
2018--2019, and is only modestly ahead of SPY in 2024--2025. The held-out
2025--2026 test period is particularly strong: PAA ends near +55%, more than
doubling the SPY result, although its maximum cumulative gain is much higher
than its terminal gain because of the large late drawdown.

The common mechanism visible in the composition panels is dynamic
concentration. The allocator frequently shifts a large share of capital into
Crude, Gold, or selected equity ETFs, interspersed with periods of broader
allocation and occasional cash increases. This behaviour allows rapid
capture of favourable trends, but it also makes performance sensitive to
reversals in the currently dominant holding. The turnover panels confirm that
the strategy is not a passive overlay: risk-on changes, de-risking, and
asset-to-asset reallocations occur repeatedly and sometimes at high
intensity.

For final discussion, the graphs support three balanced conclusions. First,
the hierarchical PAA can add substantial value beyond the SAA signal and
SPY, most clearly in validation `val_02` and test `test_00`. Second, the
benefit is not uniform across market periods; `val_00` and `val_03` show
that the allocator can finish behind stronger benchmark portfolios even
when it has produced substantial intermediate gains. Third, terminal return
alone is insufficient to characterise the strategy. The cumulative curves,
daily alpha bars, composition bands, and turnover bars together show that
the strongest returns are associated with concentrated exposures and
substantial rebalancing, and therefore must be evaluated alongside
drawdown, stability, and trading-cost considerations.

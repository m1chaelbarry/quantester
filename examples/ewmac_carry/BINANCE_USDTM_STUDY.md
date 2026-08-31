# Raport badania: EWMAC + crypto-carry na BTC USDT-M

**Werdykt: `NOT_VALIDATED`.** Combined Forecast na Binance BTC perpetual (2021-01-01 → 2026-08-30) przechodzi audyt danych, truncation, parity i stress kosztów. Nie przechodzi PBO, DSR ani MCPT. To nie jest zgoda na sizing kapitału.

Przebieg: `python3 examples/ewmac_carry/run_ccxt.py` (flaga `--full`), 2026-08-31. Kod: `ewmac_carry_example`. Ziarna: `market=8`, `mcpt=8`. Kapitał początkowy: 100 000 USDT.

---

## 1. Co dokładnie było testowane

Jeden instrument: **BTC/USDT:USDT** (Binance USD-M perpetual). Jedna pozycja netto: Combined Forecast (EWMAC + Carry Forecast) → jeden `SignalEvent`. `delay=1` (sygnał na close T, fill na open T+1). Kelly **nie** jest w pętli live — tylko diagnostyka implied \(f^*\) po siatce.

| Warstwa | Implementacja |
| --- | --- |
| Trend | EWMAC, siatka `{(16,64), (32,128)}` |
| Carry Forecast | \(F_\text{carry}\) przeciwny do dziennej sumy 3×8h funding; EMA span 2 |
| Wagi | \(w_\text{trend}=1-w_\text{carry}\), \(w_\text{carry}\in\{0.30,0.40,0.50\}\) |
| Skalar | expanding \(S=10/E[\|F_\text{raw}\|]\), \(D_f=1/\sqrt{w^\top\Omega w}\), cap ±20 |
| DVOL | Deribit; `dvol>80` → ×0.5 |
| Crowded-long | OI 3d > 25% **oraz** dzienna suma funding > 0.15% → `cap_long_increase` |
| Live size | `CarverVolTargetSizer`: \(\sigma\) Garman–Klass EWM 20d ×√365, `target_vol` ∈ {0.10, 0.15, 0.20}, Inertia β ∈ {0.10, 0.15, 0.20}, DLR 10%→20% HWM |
| Koszt BASE | taker 5 bps + half-spread 1 bp (`PerpMakerTakerCostModel` VIP0) |
| Funding cash | `cash += -qty × rate × close` na close, przed wyceną |

Hiperparametry siatki są **domyślnymi badawczymi**, nie lockami ADR.

### Dwa tory (czytać osobno)

Fast-track (siatka, PBO, DSR, MCPT, cost-stress, walk-forward, stage-12 OOS) to **twin Combined Forecast** z `FixedUnitSizer`-equivalent: `units = 0.95 × kapitał / close_0` ≈ **3.24 BTC** przez cały IS. Nie ma inertia, DLR ani księgowania funding. To test *sygnału*, nie live booka.

Event engine (stage 2) to **live loop**: Carver vol-target 10%, β=0.10, Funding Settlement, 885 filli. Median \|qty\| ≈ 0.13 BTC. Dlatego Sharpe +0.82 (twin) i +0.28 (event) nie są tym samym eksperymentem.

---

## 2. Dane

| Pole | Wartość |
| --- | --- |
| Źródło OHLCV + funding | Binance USDM REST przez `www.binance.com` (`geo_safe`; `fapi.binance.com` = HTTP 451) |
| Open interest | Binance Vision `futures/um/daily/metrics` (last print dnia); REST OI hist ≈ 30 dni |
| DVOL | Deribit `get_volatility_index_data` resolution 1D (close indeksu) |
| Kalendarz | 2068 sesji UTC daily, 2021-01-01 → 2026-08-30 |
| IS (75%) | 1551 barów, 2021-01-01 → 2025-03-31 |
| OOS (25%, zapieczętowany) | 517 barów, 2025-04-01 → 2026-08-30 |
| Audyt OHLCV | **PASS** (`expected_freq=D`) |

Pokrycie extras po align na daily bars:

| Seria | Pokrycie | Uwagi |
| --- | --- | --- |
| `funding_rate` (suma 3×8h) | 2068/2068 (100%) | średnia 2.97 bps/dzień; 88.0% dni dodatnich; min −25.5 bps, max +48.6 bps; ≈ 10.9%/rok jeśli stać non-stop long |
| `open_interest` | 2068/2068 (100%) | 36 158 → 106 427 kontraktów |
| `dvol` | 1986/2068 (96.0%) | pierwszy print 2021-03-24; średnia 60.2, mediana 55.6; **17.5%** barów > 80 (bramka ×0.5) |

Cena BTC: 29 337 (start) → 82 518 (koniec IS) → 77 635 (koniec OOS).

Buy-and-hold spot (ta sama ścieżka close, bez kosztów, bez funding):

| Okno | Total | Sharpe | Max DD | Calmar |
| --- | --- | --- | --- | --- |
| IS | **+181.3%** | 0.70 | **−76.7%** (847 dni) | 0.36 |
| OOS | −8.8% | 0.06 | −53.0% (328 dni) | −0.12 |
| Całość | +164.6% | 0.59 | −76.7% | 0.24 |

---

## 3. Siatka IS (twin, 54 trial)

Iloczyn: 2 EWMAC × 3 vol × 3 β × 3 \(w_\text{carry}\) = **54 wpisy** w `TrialsRegistry`. Vol i β **nie wchodzą** do fast-track, więc unikalnych ścieżek PnL jest **6**. PBO liczone było na tych 6 kolumnach; DSR dostał N=54 z rejestru (konserwatywnie za dużo prób).

Champion (max annualized Sharpe twin): **EWMAC 16/64, \(w_\text{carry}=0.30\)**. Vol=0.10 i β=0.10 wzięte z pierwszego maksimum w kolejności siatki — mają znaczenie dopiero w event engine.

| EWMAC | \(w_\text{carry}\) | Total IS | Sharpe | Max DD | Calmar | Equity końcowa |
| --- | --- | --- | --- | --- | --- | --- |
| 16/64 | **0.30** | **+116.0%** | **+0.819** | −23.3% (227d) | 0.85 | 216 000 |
| 16/64 | 0.40 | +103.7% | +0.810 | −20.8% (227d) | 0.88 | 203 676 |
| 16/64 | 0.50 | +77.4% | +0.733 | −17.6% (267d) | 0.82 | 177 418 |
| 32/128 | 0.30 | +72.2% | +0.586 | −24.8% (107d) | 0.55 | 172 243 |
| 32/128 | 0.40 | +60.0% | +0.543 | −23.5% (44d) | 0.50 | 160 010 |
| 32/128 | 0.50 | +38.0% | +0.429 | −26.0% (357d) | 0.30 | 137 993 |

Sąsiedzi: min Sharpe **+0.429**. Sensitivity **PASS** (próg sąsiadów > −1.5). Implied Kelly z dziennych zwrotów championa: \(f^*=3.101\), ¼-Kelly = 0.775 — **tylko diagnostyka**; nienakładane na vol-target (ADR 0001).

Combined Forecast championa na IS: średnia F = −0.32, \|F\| = 8.61, σ(F) = 10.22, przy capie ±20 przez 3.8% barów. Long 47.2% / short 48.7%. **Crowded-long: 0.0% barów** — próg OI 3d>25% i funding>0.15% nie wystąpił łącznie.

---

## 4. Event engine (live loop) — IS

Parametry championa: 16/64, \(w_\text{carry}=0.30\), `target_vol=0.10`, `inertia_beta=0.10`, funding booked, koszty BASE.

| Metryka | Wartość |
| --- | --- |
| Total return | **+9.30%** |
| Sharpe (simple, \(N_T\) z kalendarza) | **+0.276** |
| Max DD | **−18.55%** |
| Peak → trough | 2022-09-06 → 2023-10-18 (937 dni do odzyskania HWM) |
| Calmar | 0.114 |
| Equity końcowa | 109 302 |
| Accounting | `equity = cash + MTM` **OK** (cash 122 634, MTM −13 332 — książka skończyła na short) |

Pozycja (1551 barów IS):

| | |
| --- | --- |
| % long / short / flat | 47.1% / 48.6% / 4.3% |
| median \|qty\| | 0.13 BTC |
| mean \|qty\| | 0.44 BTC |
| max \|qty\| | 10.06 BTC (warmup GK / brak DLR na starcie — ogon ryzyka sizeru) |
| Fille | 885 |
| Round-trips | 408 |
| Prowizja (c_t) | 1 198 USDT |
| φ_t (half-spread w BASE) | 240 USDT |
| Funding cash (przybliżenie \(-\,q_t r_t P_t\)) | **−447 USDT** |

Funding jest dodatni przez 88% dni, a trend (waga 0.70) trzyma longa w rajdach — książka **płaci** carry, nie zbiera go. To jest spójne z \(w_\text{carry}=0.30\) i z tym, że podniesienie wagi carry w twin **obniża** Sharpe (0.82 → 0.73).

Zwroty calendar-year event engine (IS; 2025 tylko do 31.03):

| Rok | n | Total | Sharpe |
| --- | --- | --- | --- |
| 2021 | 365 | −0.9% | −0.15 |
| 2022 | 365 | **+19.5%** | **+1.13** |
| 2023 | 365 | −7.5% | −1.99 |
| 2024 | 366 | +1.4% | +0.41 |
| 2025 IS | 90 | −0.1% | −0.15 |

Krawędź IS siedzi prawie cała w 2022 (trend niedźwiedzi + short). 2023 zjada część zysku.

---

## 5. Walk-forward, OOS, koszty (twin)

Walk-forward (expanding train, test 84 bary, 16 foldów, re-opt na unikalnych 6 trójkach): stitched **Sharpe +0.698**, total **+65.4%**, max DD −22.9% (235d), Calmar 0.64. To nadal fast-track.

Cost-stress twin IS (ten sam target Combined Forecast):

| Scenariusz | Taker / spread | Sharpe |
| --- | --- | --- |
| BASE | 5 bps / 2 bps full | **+0.819** |
| CONSERVATIVE | 8 bps / 4 bps + Kaufman/Kyle | +0.645 |
| STRESS | 15 bps / 10 bps + wyższy impact | +0.439 |

Bramka cost-stress **PASS** (progi luźne: BASE > −1, CONSERVATIVE > −1.5). Wszystkie trzy Sharpe dodatnie.

Zapieczętowany OOS, **bez re-opt**, champion 16/64 / \(w=0.30\):

| Tor | Total | Sharpe | Max DD |
| --- | --- | --- | --- |
| Twin (stage 12) | **+12.47%** | **+0.491** | −19.3% (121d) |
| Event engine (świeży kapitał na OOS) | −2.41% | −0.02 | −16.1% (61d) |
| Buy-and-hold BTC | −8.8% | +0.06 | −53.0% |

Twin OOS bije hold. Live sizer na OOS jest około zera. Stage-12 w pipeline raportuje twin — bramka `untouched_oos` **PASS** dotyczy tego toru.

---

## 6. Robustness, która obala walidację

### PBO / CSCV — **FAIL** (0.730)

Próg: PBO < 0.10. 16 bloków, C(16,8) kombinacji, 6 unikalnych trial. PBO = **0.730**. W ~73% podziałów CSCV IS-champion ma logit < 0 (słabo na komplemencie). Siatka wybiera specyfikację, która nie jest stabilna między podokresami.

### DSR — **FAIL** (0.842)

DSR = **0.8419**, próg 0.95, N=54 z rejestru. Deflacja za multiple testing zjada IS Sharpe. Gdyby N=6 (unikalne PnL), DSR byłby wyższy — i tak nie zmienia PBO ani MCPT.

### MCPT — **FAIL** (p = 0.327)

`n_reps=101` (w tym original). `permute_joint_bars` (cały bar + extras sklejone). p = (1 + #{perm ≥ orig}) / 101 = **0.3267** → ok. 32/100 permutacji bije Sharpe +0.819. Gate p < 0.05.

Autokorelacja residuali ceny: **serial_correlation=True** (runs p=0.0096; Ljung-Box p=0.090). Diagnostic rekomenduje **block bootstrap / O-U**, a MCPT i tak zrobił IID shuffle barów. p-value może być źle skalibrowane; nawet wtedy nie ma powodu traktować krawędzi jako skill.

### Nested DD bound

Masters double bootstrap (n_outer=40, n_inner=16 na `--full`): bound **0.567** (56.7% DD). Event IS max DD 18.5% jest wewnątrz; twin IS 23.3% też. Bound nie jest bramką tworzącą `VALIDATED`.

### Truncation i parity — **PASS**

Truncation: 1531 wierszy po obcięciu 20 barów, 0 mismatch. Parity event↔fast przy `FixedUnitSizer` i **wyłączonym** funding: max \|Δ equity\| = 2.91e3, rel **1.18%** < 5%.

CPCV: **N/A** (reguła, nie fitted model).

---

## 7. Tabela bramek (stage 13)

| Bramka | Status | Mandatory |
| --- | --- | --- |
| data_audit | PASS | tak |
| temporal_truncation | PASS | tak |
| event_vectorized_parity | PASS | nie |
| execution_cost_stress | PASS | tak |
| cpcv | NOT_APPLICABLE | tak |
| **pbo** | **FAIL (0.730)** | tak |
| **dsr** | **FAIL (0.842 < 0.95)** | tak |
| untouched_oos | PASS | tak |
| **monte_carlo_robustness** | **FAIL (MCPT p=0.327)** | tak |
| sensitivity_analysis | PASS | nie |
| portfolio_accounting_invariants | PASS | tak |
| execution_assumptions_documented | PASS | tak |

**Status końcowy: `NOT_VALIDATED`.**

---

## 8. Interpretacja (bez dopasowywania narracji post hoc)

1. Twin z prawie 1× notional na starcie (3.24 BTC) wygląda jak trend-follower z dodatnim IS Sharpe i dodatnim OOS twin. To **nie** jest wielkość, którą da Carver 10% vol — live book robi +9% na IS przy DD 18.5% i **płaci** funding.
2. Większy \(w_\text{carry}\) systematycznie **psuje** twin Sharpe. Funding 88% czasu dodatni; carry chce shorta, trend chce longa. 60/40 z raportu nie wygrał nawet siatki (wygrało 70/30).
3. PBO 0.73 + MCPT p=0.33: IS ranking 16/64 jest kruchy. 2022 nosi event PnL; 2023 jest głęboko ujemny.
4. Crowded-long nigdy nie strzelił — filtr jest martwy przy tych progach na tej próbce.
5. Max 10 BTC vs mediana 0.13 BTC: sizer bez twardego capu notional na starcie GK. Osobny hazard vs sam forecast.
6. Serial correlation w cenie: następny MCPT powinien być blokowy, nie IID bar-shuffle. To nie uratuje PBO.

Nie ma podstaw, by ten Combined Forecast uznać za skill na BTC perp w tym oknie. Hipoteza „trend + carry na funding” nie przeszła bramek, które same zdefiniowaliśmy.

---

## 9. Reprodukcja

```bash
pip install 'quantester[ccxt]'
python3 examples/ewmac_carry/run_ccxt.py          # --full domyślnie
# lżejszy MCPT:
python3 examples/ewmac_carry/run_ccxt.py --quick
```

Artefakty przebiegu: `examples/ewmac_carry/output/` (`BTC_PERP_CCXT.csv`, `trials.db`, `validation_report.json`, tearsheet). Ten plik opisuje **ten** run; etykieta `synthetic extras` w zapisanym `validation_report.txt` z 2026-08-31 jest pomyłką ówczesnego `stage_report` (poprawiona w kodzie po fakcie) — dane były live Binance+Deribit.

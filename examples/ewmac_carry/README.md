# EWMAC + Crypto Carry

Combined Forecast on a BTC USDT-M perpetual: EWMAC trend + funding Carry Forecast
in one net `SignalEvent`. Live size is the opt-in Carver vol-target sizer
(Inertia Buffer + Drawdown De-lever). Kelly is a diagnostic only.

```bash
# Offline demo (synthetic extras, 13-stage research pipeline)
python examples/ewmac_carry/run.py
python examples/ewmac_carry/run.py --full   # heavier MCPT

# Real CCXT (needs pip install "quantester[ccxt]" + network)
python examples/ewmac_carry/run_ccxt.py
```

A green `VALIDATED` on the synthetic demo means the **workflow** ran. It is not
permission to size live capital. Grid hyperparameters are study defaults.

Live Binance USDT-M (funding paginated, OI from Vision dumps, Deribit DVOL).
Regional `fapi.binance.com` 451 is rerouted via `geo_safe=True`. No GBM fallback.

Raport z przebiegu 2021-01-01 → 2026-08-30: [`BINANCE_USDTM_STUDY.md`](BINANCE_USDTM_STUDY.md) (`NOT_VALIDATED`).

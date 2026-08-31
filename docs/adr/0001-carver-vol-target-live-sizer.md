# Opt-in Carver vol-target live sizer

D5 keeps Kelly, Vince \(f^*\), and vol-target as research libraries; live sizing defaults to cash-base percent. EWMAC + crypto carry needs Carver’s \(N \propto F/(\sigma \cdot 10)\) and an Inertia Buffer in the event loop, so those become an **opt-in** `PortfolioManager` sizer. The default sizer does not change. Combined Forecast stays on the strategy; quantity stays in the portfolio. Fractional Kelly is not layered on vol-target (that would double-leverage).

**Considered options.** Strategy-computed \(N_{\text{pos}}\) (violates the strategy contract). Direction-only `PercentEquitySizer` (does not implement this strategy).

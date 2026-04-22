# AI Decision System Memory

## Product Goal

The user is building an automated cryptocurrency paper-trading system that can later evolve into an AI-assisted decision maker. The goal is not just to execute a fixed technical strategy. The system should collect market, macro, on-chain, and strategy-performance evidence, then decide whether to enter, hold, exit, pause, or resume trading.

The current simulation target is:

- Start capital: 1,000,000 KRW
- Simulation window: 2026-04-20 21:10 KST to 2026-04-25 18:00 KST
- Desired daily objective: up to 3% per 24-hour period
- Risk posture: paper trading only, no real orders until the decision process is trustworthy
- Key reporting principle: show expected value, fees, sample size, drawdown, halted state, and decision reasons clearly

## User Preferences And Lessons Learned

- The system must keep running when the local PC is off.
- GitHub Actions can be used, but scheduled runs can fail or lag. The trading engine must tolerate missed ticks and catch up when it runs again.
- A long halted state is a problem unless it has a clear reason and a recovery rule.
- Halt reasons must be analyzed automatically. The system should not simply stop forever after a risk event.
- The daily profit target should reset by rolling 24-hour windows from the actual start point, not only by calendar midnight.
- Entry decisions should prefer assets with plausible upside toward the 3% objective, not only assets with short-term trend strength.
- Manual research categories should become automatic inputs:
  - macro conditions
  - on-chain metrics
  - strategy comparison
  - aggregate buying pressure / total-buy analysis
  - blocked-entry analysis
  - realized performance and expected value
- The user is willing to connect additional APIs if they improve decision quality.

## Current Engine Direction

The system should operate in layers:

1. Data collection
   - Upbit candles and top KRW markets
   - BTC regime and volatility
   - Fear & Greed Index
   - Bitcoin on-chain activity when available
   - Later: exchange orderbook, funding, stablecoin flows, global macro risk proxies

2. Rule-based safety gate
   - Never rely on AI alone for risk limits.
   - Enforce max position fraction, loss limits, stop-loss, time-stop, and cooldowns.
   - Forced exits must remain deterministic.

3. Candidate scoring
   - Score candidates by trend, volume, volatility-adjusted position size, expected upside, and market context.
   - Avoid buying only because something has already pumped.
   - Track blocked candidates and later price movement to judge whether filters are too strict or too loose.

4. AI decision layer
   - The AI should act as a decision reviewer or allocator, not as an uncontrolled trade executor.
   - It should receive structured JSON evidence and return structured JSON:
     - action: buy, hold, sell, pause, resume
     - confidence
     - thesis
     - invalidation condition
     - expected upside
     - expected downside
     - risk notes
   - If the AI response is missing, invalid, or too risky, the deterministic safety layer should reject the decision.
   - Current implementation starts with `provider=local`, which creates the same structured review object without requiring an API key.
   - When `provider=openai` and `OPENAI_API_KEY` are present, the system can call the OpenAI Responses API with a JSON schema and then fall back to local review if the API is unavailable.

5. Reporting
   - The report must show current capital prominently.
   - It must show why the system is halted or active.
   - It must show whether decisions are based on enough sample size.
   - It must distinguish realized profit from unrealized portfolio value.

## Useful APIs To Consider

Free or low-friction sources:

- Upbit public API: candles, ticker, market list, orderbook
- Alternative.me Fear & Greed Index: sentiment proxy
- Blockchain.com stats API: BTC transaction and network summary
- CoinGecko API: market cap, volume, dominance, global market data
- Binance public API: global ticker, futures funding, open interest where available
- CryptoCompare / CoinMarketCap: broader market metrics, if API key is available

Potential paid or key-based sources:

- Glassnode: on-chain metrics
- CryptoQuant: exchange flows and on-chain indicators
- Santiment: social/on-chain metrics
- IntoTheBlock: token-level analytics
- TradingView or equivalent data provider: broader technical indicators
- OpenAI API: AI decision review and structured reasoning

## Implementation Notes

- Keep API keys out of the repository. Use environment variables or GitHub Secrets.
- Every external data collector must fail soft. Missing macro/on-chain data should reduce confidence or become neutral, not crash the simulation.
- Store each decision input and output in logs so the strategy can be audited later.
- The AI layer should be added behind a feature flag such as `ai_decision.enabled`.
- Current automatic collectors include Fear & Greed, Blockchain.com stats, CoinGecko global market data, and Binance BTCUSDT 24h ticker data. These must remain soft-fail collectors.
- Before real trading, require a paper-trading evaluation period with enough completed trades for a meaningful sample.

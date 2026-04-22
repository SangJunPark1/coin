# Cloud paper simulation

This project can run without the local PC by using GitHub Actions for compute and GitHub Pages for the live HTML report.

## Schedule

- Start: 2026-04-20 21:10 KST
- End: 2026-04-25 18:00 KST
- Cadence: every 5 minutes, best effort
- Starting cash: 1,000,000 KRW
- Mode: paper simulation only, no real orders

GitHub Actions scheduled workflows use UTC cron and the shortest supported interval is 5 minutes. Runs can be delayed by GitHub queueing, so the first tick may start shortly after 21:10 KST instead of exactly on the second.

## Files

- `.github/workflows/cloud-simulation.yml`: wakes up every 5 minutes.
- `coin_mvp/cloud_tick.py`: loads the previous state, runs one simulation tick, saves state, and refreshes the report.
- `config.cloud.json`: cloud-only configuration using 1,000,000 KRW starting cash.
- `data/cloud_state.json`: persisted broker/risk/position state.
- `data/cloud_trades.csv`: cloud paper trade journal.
- `logs/cloud_events.jsonl`: cloud event log.
- `docs/index.html`: GitHub Pages report.

## GitHub setup

1. Create a public GitHub repository.
2. Push this project to the repository.
3. Open the repository on GitHub.
4. Go to `Settings > Actions > General`.
5. Under `Workflow permissions`, choose `Read and write permissions`.
6. Go to `Settings > Pages`.
7. Set the source to `Deploy from a branch`.
8. Choose branch `main` and folder `/docs`.
9. Open `Actions > Cloud Paper Simulation`.
10. Run it once manually with `reset=true` if you want a clean cloud simulation.

The live report URL will usually look like:

```text
https://YOUR_GITHUB_USERNAME.github.io/YOUR_REPOSITORY_NAME/
```

## Stopping

After 2026-04-25 18:00 KST, the script marks the simulation as finished and stops changing the state. You can also disable the workflow manually in GitHub Actions.

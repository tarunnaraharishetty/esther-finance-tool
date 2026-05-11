# Alpaca paper account — setup notes

Short handoff doc so future-you can pick up where we left off.

## 1. Where to paste the keys

Open `.env` in the project root (create it from `.env.example` first if it
doesn't exist):

```powershell
copy .env.example .env
notepad .env
```

Replace these two lines with the **paper** key and secret you generated at
<https://app.alpaca.markets/paper/dashboard/overview>:

```
ALPACA_API_KEY=PK_your_real_paper_key_here
ALPACA_API_SECRET=your_real_paper_secret_here
```

Leave everything else as-is. In particular:

```
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

must stay pointed at the paper endpoint.

## 2. Starting the live dashboard tomorrow

```powershell
cd "C:\Users\tarun\Downloads\esther finance tool"

# verify your setup (no orders submitted, just probes)
esther doctor

# launch the live dashboard once doctor is all-green
esther dashboard
```

First launch downloads FinBERT (~440 MB). Subsequent launches are instant.

If you want to skip the model download:

```powershell
esther dashboard --no-sentiment
```

If you want to demo without any Alpaca calls:

```powershell
esther dashboard --mock
```

## 3. Useful commands

```powershell
esther                        # show the banner / command list
esther status                 # print the active config (secrets masked)
esther doctor                 # run all preflight checks
esther doctor --init-env      # scaffold .env from .env.example
esther recommend              # one-shot recommendations for the watchlist
esther backtest --symbol SPY  # walk-forward backtest vs buy-and-hold
esther dashboard              # live Textual dashboard (paper, observational)
```

Keyboard inside the dashboard: `q` quit · `r` refresh now · `p` pause/resume · `↑` `↓` select a row.

## 4. Safety notes

- **Paper trading only.** Every live-data command refuses to start unless
  `ALPACA_BASE_URL` is the paper endpoint. Don't change it.
- **Never use live keys with this code.** If you ever generate a live key
  by accident, leave it in your Alpaca dashboard — don't paste it here.
- **Keep `.env` private.** It's gitignored, but:
  - Don't email it, screenshot it, or paste it into chats.
  - If the keys leak, regenerate them in the Alpaca dashboard
    (View → Generate New Key) — the old pair is invalidated instantly.
- **The dashboard never submits orders.** It's read-only.
- **`esther run` defaults to dry-run.** Order submission requires the
  explicit `--execute` flag, and even then only against paper.

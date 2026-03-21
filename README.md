# DigitalBlackOutIran

A simple static site that publishes **one automated post per day** about the **economic impact of the Iranian internet outage** on the **world economy plus the top 10 largest economies**.

## What this starter includes

- Daily scheduled publishing with **GitHub Actions**
- Static archive page with a **NetBlocks-like card layout**
- One post per day with:
  - chart image
  - title
  - meta description
  - Open Graph tags
  - Twitter card tags
  - JSON-LD Article schema
  - RSS feed and sitemap
- No CMS and no admin panel required
- Fully static output, so it can be hosted on **GitHub Pages** or **Cloudflare Pages**

## Important honesty note

This project should **not** claim to measure audited or observed real-world GDP loss unless you have a defensible methodology and verified data.

The current starter uses a **transparent model-based exposure estimate**:

`daily exposure = GDP_2025 × outage severity × exposure weight × base shock rate / 365`

That means the output is an **estimate**, not a proven macroeconomic loss number.

## Data model

### 1) Outage signal
By default, the workflow tries to read the latest Iran-related item from `https://netblocks.org/reports/` and infer a basic daily signal. If that fails, it falls back to `data/fallback_signal.json`.

### 2) GDP baseline
The demo uses `data/gdp_2025_demo.json`.

For production, replace this file with your vetted **2025 GDP baseline**. A practical source is the IMF World Economic Outlook 2025 database, while the World Bank API is useful for latest published GDP values when you want a fallback baseline.

### 3) Exposure weights
`data/exposure_weights_demo.json` contains demo weights. Replace them with your own research-backed country exposure model.

## Deployment flow

1. Push this repo to GitHub.
2. Enable **GitHub Pages** in the repository.
3. The workflow in `.github/workflows/daily.yml` runs every day. On a fresh deploy, it backfills every missing daily post from the outage start date in `data/fallback_signal.json` through the latest available day, then continues with one new post per day.
4. It generates a new post, chart, archive page, feed, and sitemap.
5. It commits the new content and deploys the static site automatically.

## Local run

```bash
pip install -r requirements.txt
BOOTSTRAP_DEMO=1 python scripts/build.py
```

Then open `site/index.html`.

## Files you will probably edit first

- `config/site.json` → branding and domain
- `data/gdp_2025_demo.json` → replace with real 2025 GDP baseline
- `data/exposure_weights_demo.json` → replace with your exposure logic
- `scripts/build.py` → refine the estimation model and source adapter

## Suggested next upgrade

The first upgrade should be replacing the demo weights with a stronger model, for example:

- bilateral trade exposure to Iran
- energy price sensitivity
- shipping/logistics dependency
- remittance or platform dependency
- sector-specific vulnerability weights

That would make the site much more credible than using GDP alone.

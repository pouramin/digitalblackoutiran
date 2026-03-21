#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from bs4 import BeautifulSoup
import requests

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'config'
DATA = ROOT / 'data'
CONTENT = ROOT / 'content' / 'posts'
SITE = ROOT / 'site'
IMG = SITE / 'assets' / 'img'
CSS = SITE / 'assets' / 'css'


@dataclass
class Signal:
    source: str
    country: str
    current_connectivity_pct: float
    days_since_drop: int
    summary: str
    series: List[Dict[str, Any]]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9]+', '-', text)
    return re.sub(r'-+', '-', text).strip('-')


def fmt_money(value: float) -> str:
    abs_v = abs(value)
    if abs_v >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:.2f}T"
    if abs_v >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    if abs_v >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    return f"${value:,.0f}"


def fmt_pct(value: float) -> str:
    return f"{value:.2f}%"


def read_site_config() -> Dict[str, Any]:
    return load_json(CONFIG / 'site.json')


def read_model_config() -> Dict[str, Any]:
    return load_json(CONFIG / 'model.json')


def read_gdp() -> Dict[str, Any]:
    return load_json(DATA / 'gdp_2025_demo.json')


def read_weights() -> Dict[str, float]:
    return load_json(DATA / 'exposure_weights_demo.json')


def fetch_signal_from_netblocks() -> Optional[Signal]:
    url = os.getenv('NETBLOCKS_LATEST_URL', 'https://netblocks.org/reports/')
    timeout = int(os.getenv('HTTP_TIMEOUT', '20'))
    try:
        resp = requests.get(url, timeout=timeout, headers={'User-Agent': 'Mozilla/5.0'})
        resp.raise_for_status()
    except Exception:
        return None

    soup = BeautifulSoup(resp.text, 'html.parser')
    cards = soup.select('article, .post, .card, .elementor-post')
    iran_text = None
    for card in cards:
        text = ' '.join(card.get_text(' ', strip=True).split())
        if 'iran' in text.lower() and ('blackout' in text.lower() or 'internet' in text.lower()):
            iran_text = text
            break
    if iran_text is None:
        page_text = ' '.join(soup.get_text(' ', strip=True).split())
        if 'iran' not in page_text.lower():
            return None
        iran_text = page_text

    day_match = re.search(r'(?:day|hour)\s+(\d+)', iran_text, flags=re.I)
    days = 0
    if day_match:
        days = int(day_match.group(1))
        if 'hour' in day_match.group(0).lower():
            days = max(1, math.floor(days / 24))

    lowered = iran_text.lower()
    connectivity = 1.0
    if 'restored' in lowered or 'momentarily back online' in lowered:
        connectivity = 8.0
    elif 'limited domestic intranet' in lowered or 'still unavailable' in lowered or 'blackout' in lowered:
        connectivity = 1.0

    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=27)
    series = []
    pre_drop_days = max(3, 27 - max(days, 1))
    for i in range(28):
        d = start + timedelta(days=i)
        if i < pre_drop_days:
            pct = 96 - (i % 4)
        else:
            pct = connectivity
        series.append({'date': d.isoformat(), 'connectivity_pct': pct})

    summary = re.sub(r'\s+', ' ', iran_text)[:240].strip()
    return Signal(
        source='netblocks',
        country='Iran',
        current_connectivity_pct=connectivity,
        days_since_drop=max(days, 1),
        summary=summary,
        series=series,
    )


def read_fallback_signal() -> Signal:
    raw = load_json(DATA / 'fallback_signal.json')
    return Signal(**raw)


def get_signal() -> Signal:
    live = fetch_signal_from_netblocks()
    if live:
        return live
    return read_fallback_signal()


def first_outage_date(signal: Signal, threshold_pct: float) -> Optional[date]:
    for item in signal.series:
        if float(item['connectivity_pct']) <= threshold_pct:
            return datetime.fromisoformat(item['date']).date()
    return None


def signal_trimmed_to_date(base_signal: Signal, current_date: date, threshold_pct: float) -> Optional[Signal]:
    trimmed = [x for x in base_signal.series if datetime.fromisoformat(x['date']).date() <= current_date]
    if not trimmed:
        return None
    outage_start = first_outage_date(base_signal, threshold_pct)
    if outage_start and current_date >= outage_start:
        days_since = (current_date - outage_start).days + 1
    else:
        days_since = 0
    return Signal(
        source=base_signal.source,
        country=base_signal.country,
        current_connectivity_pct=float(trimmed[-1]['connectivity_pct']),
        days_since_drop=max(days_since, 1),
        summary=base_signal.summary,
        series=trimmed,
    )


def align_signal_to_date(base_signal: Signal, target_date: date, threshold_pct: float) -> Signal:
    series = sorted(base_signal.series, key=lambda x: x['date'])
    if not series:
        return base_signal
    last_date = datetime.fromisoformat(series[-1]['date']).date()
    last_pct = float(series[-1]['connectivity_pct'])
    while last_date < target_date:
        last_date += timedelta(days=1)
        series.append({'date': last_date.isoformat(), 'connectivity_pct': last_pct})
    aligned = Signal(
        source=base_signal.source,
        country=base_signal.country,
        current_connectivity_pct=float(series[-1]['connectivity_pct']),
        days_since_drop=base_signal.days_since_drop,
        summary=base_signal.summary,
        series=series,
    )
    trimmed = signal_trimmed_to_date(aligned, target_date, threshold_pct)
    return trimmed or aligned


def backfill_history(site_cfg: Dict[str, Any], model_cfg: Dict[str, Any], base_signal: Optional[Signal] = None) -> None:
    signal = base_signal or read_fallback_signal()
    threshold = float(model_cfg.get('full_shutdown_threshold_pct', 5.0))
    outage_start = first_outage_date(signal, threshold)
    if outage_start is None:
        return
    latest_date = datetime.fromisoformat(signal.series[-1]['date']).date()
    existing_dates = {load_json(path)['date'] for path in CONTENT.glob('*.json') if path.is_file()}
    current_date = outage_start
    while current_date <= latest_date:
        if current_date.isoformat() not in existing_dates:
            trimmed_signal = signal_trimmed_to_date(signal, current_date, threshold)
            if trimmed_signal is not None:
                build_one(current_date, site_cfg, model_cfg, signal_override=trimmed_signal)
        current_date += timedelta(days=1)


def compute_daily_impact(signal: Signal, gdp: Dict[str, Any], weights: Dict[str, float], model: Dict[str, Any]) -> Dict[str, Any]:
    severity = max(0.0, min(1.0, (100.0 - signal.current_connectivity_pct) / 100.0))
    annualized_shock = model['base_annualized_shock_rate'] * severity
    per_country = []
    for code, meta in gdp.items():
        if code == 'WORLD':
            continue
        weight = float(weights.get(code, 0.0))
        loss = float(meta['gdp_usd']) * annualized_shock * weight / 365.0
        relative = annualized_shock * weight * 100.0
        per_country.append({
            'code': code,
            'name': meta['name'],
            'gdp_usd': float(meta['gdp_usd']),
            'weight': weight,
            'daily_loss_usd': loss,
            'relative_pct_of_annual_gdp': relative,
        })

    per_country.sort(key=lambda x: x['daily_loss_usd'], reverse=True)
    world_loss = sum(x['daily_loss_usd'] for x in per_country)
    world_gdp = float(gdp['WORLD']['gdp_usd'])
    world_relative = (world_loss * 365.0 / world_gdp) * 100.0
    return {
        'severity': severity,
        'annualized_shock_rate': annualized_shock,
        'world_daily_loss_usd': world_loss,
        'world_relative_pct_of_annual_gdp': world_relative,
        'countries': per_country,
    }


def make_chart(signal: Signal, impact: Dict[str, Any], chart_path: Path, title_date_end: Optional[str] = None) -> None:
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    dates = [datetime.fromisoformat(x['date']).date() for x in signal.series]
    impact_pct = [((100 - float(x['connectivity_pct'])) / 100.0) * impact['world_relative_pct_of_annual_gdp'] * 100 for x in signal.series]
    # keep within readable range and normalize against today's impact
    today_pct = max(impact_pct[-1], 0.0001)
    normalized = [min(100.0, (v / today_pct) * 100.0) for v in impact_pct]

    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    fig.patch.set_facecolor('#0b0f14')
    ax.set_facecolor('#0b0f14')

    ax.plot(dates, normalized, linewidth=2.6, color='#98d08b')
    ax.fill_between(dates, normalized, 0, color='#98d08b', alpha=0.18)

    ax.set_ylim(0, 120)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels([f'{v}%' for v in [0, 20, 40, 60, 80, 100]], color='#cbd5e1', fontsize=11)
    ax.tick_params(axis='x', colors='#cbd5e1', labelsize=10)

    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=7))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%-m/%-d'))

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, axis='both', color='white', alpha=0.12, linewidth=0.8)

    start_outage = None
    for item in signal.series:
        if float(item['connectivity_pct']) <= 5.0:
            start_outage = datetime.fromisoformat(item['date']).date()
            break
    if start_outage:
        y_level = 63
        ax.annotate('', xy=(dates[-1], y_level), xytext=(start_outage, y_level), arrowprops=dict(arrowstyle='<->', color='#ff2d2d', linewidth=1.8))
        ax.text(start_outage + (dates[-1] - start_outage) / 2, y_level + 6, 'ESTIMATED SHOCK WINDOW', color='#ffb000', ha='center', va='bottom', fontsize=12, fontweight='bold')

    end_txt = title_date_end or dates[-1].isoformat()
    ax.set_title(f'Estimated Economic Exposure - Iran: {dates[0].isoformat()} to {end_txt} UTC', color='#e5e7eb', fontsize=18, pad=22)
    ax.set_ylabel('Daily shock index (normalized)', color='#cbd5e1', fontsize=13, labelpad=16)

    fig.text(0.055, 0.09, '■ Iran-linked global exposure', color='#98d08b', fontsize=11)
    fig.text(0.80, 0.09, 'min', color='#38bdf8', fontsize=12, fontweight='bold')
    fig.text(0.865, 0.09, f"{min(normalized):.0f}%", color='#e5e7eb', fontsize=12)
    fig.text(0.91, 0.09, 'current', color='#38bdf8', fontsize=12, fontweight='bold')
    fig.text(0.985, 0.09, f"{normalized[-1]:.0f}%", color='#e5e7eb', fontsize=12, ha='right')

    plt.tight_layout(rect=[0.03, 0.12, 0.98, 0.94])
    fig.savefig(chart_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)


def build_post(today: date, signal: Signal, impact: Dict[str, Any], site_cfg: Dict[str, Any]) -> Dict[str, Any]:
    title = f"Iran outage economic impact update — day {signal.days_since_drop}"
    slug = f"{today.isoformat()}-{slugify(title)}"
    top3 = impact['countries'][:3]
    bullet = '; '.join(f"{c['name']}: {fmt_money(c['daily_loss_usd'])}/day" for c in top3)
    description = (
        f"Day {signal.days_since_drop} of the Iran outage: our model estimates {fmt_money(impact['world_daily_loss_usd'])} in daily global GDP exposure. "
        f"Top exposed economies today: {bullet}."
    )
    excerpt = (
        f"Using a {2025} GDP baseline and configurable exposure weights, today's model estimates global exposure at "
        f"{fmt_money(impact['world_daily_loss_usd'])} per day with Iran's connectivity at {signal.current_connectivity_pct:.1f}%."
    )
    body = [
        f"Update: Iran's outage is now at day {signal.days_since_drop}, with current connectivity modeled at {signal.current_connectivity_pct:.1f}%.",
        f"This site does not claim audited economic losses. It publishes a transparent daily estimate based on a 2025 GDP baseline, outage severity, and configurable country exposure weights.",
        f"Today's estimated global GDP exposure is {fmt_money(impact['world_daily_loss_usd'])} per day, equal to {fmt_pct(impact['world_relative_pct_of_annual_gdp'])} of annual world GDP on an annualized basis.",
        f"Most exposed major economies today: {bullet}."
    ]
    return {
        'date': today.isoformat(),
        'title': title,
        'slug': slug,
        'description': description[:160],
        'excerpt': excerpt,
        'body_paragraphs': body,
        'signal': {
            'source': signal.source,
            'summary': signal.summary,
            'current_connectivity_pct': signal.current_connectivity_pct,
            'days_since_drop': signal.days_since_drop,
        },
        'impact': {
            'world_daily_loss_usd': impact['world_daily_loss_usd'],
            'world_relative_pct_of_annual_gdp': impact['world_relative_pct_of_annual_gdp'],
            'countries': impact['countries']
        },
        'chart_image': f"/assets/img/{today.isoformat()}-world.png",
        'author': site_cfg['author']
    }


def load_existing_posts() -> List[Dict[str, Any]]:
    posts = []
    for path in sorted(CONTENT.glob('*.json')):
        try:
            posts.append(load_json(path))
        except Exception:
            continue
    posts.sort(key=lambda x: x['date'], reverse=True)
    return posts


def save_post(post: Dict[str, Any]) -> None:
    save_json(CONTENT / f"{post['slug']}.json", post)


def render_page(title: str, description: str, canonical: str, body: str, site_cfg: Dict[str, Any], article_json_ld: Optional[str] = None, og_image: Optional[str] = None) -> str:
    json_ld = f'<script type="application/ld+json">{article_json_ld}</script>' if article_json_ld else ''
    og = og_image or f"{site_cfg['site_url'].rstrip('/')}/assets/img/social-card.png"
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>{title}</title>
  <meta name=\"description\" content=\"{description}\">
  <link rel=\"canonical\" href=\"{canonical}\">
  <meta property=\"og:type\" content=\"article\">
  <meta property=\"og:title\" content=\"{title}\">
  <meta property=\"og:description\" content=\"{description}\">
  <meta property=\"og:url\" content=\"{canonical}\">
  <meta property=\"og:image\" content=\"{og}\">
  <meta name=\"twitter:card\" content=\"summary_large_image\">
  <meta name=\"twitter:title\" content=\"{title}\">
  <meta name=\"twitter:description\" content=\"{description}\">
  <meta name=\"twitter:image\" content=\"{og}\">
  <meta name=\"twitter:site\" content=\"{site_cfg.get('x_handle', '')}\">
  <meta name=\"twitter:creator\" content=\"{site_cfg.get('x_handle', '')}\">
  <link rel=\"alternate\" type=\"application/rss+xml\" title=\"{site_cfg['site_name']} RSS\" href=\"{site_cfg['site_url'].rstrip('/')}/feed.xml\">
  <link rel=\"stylesheet\" href=\"/assets/css/styles.css\">
  {json_ld}
</head>
<body>
  {body}
</body>
</html>"""


def render_index(posts: List[Dict[str, Any]], site_cfg: Dict[str, Any]) -> None:
    cards = []
    for post in posts[: site_cfg['cards_per_page']]:
        cards.append(f"""
        <article class=\"card\">
          <div class=\"card-head\">
            <span class=\"badge\">Daily Report</span>
            <time datetime=\"{post['date']}\">{post['date']}</time>
          </div>
          <h2><a href=\"/posts/{post['slug']}/\">{post['title']}</a></h2>
          <p>{post['excerpt']}</p>
          <a class=\"thumb-link\" href=\"/posts/{post['slug']}/\"><img src=\"{post['chart_image']}\" alt=\"{post['title']} chart\"></a>
        </article>
        """)
    body = f"""
    <main class=\"shell\">
      <header class=\"hero\">
        <p class=\"eyebrow\">NEW REPORTS AND UPDATES</p>
        <h1>{site_cfg['site_name']}</h1>
        <p class=\"lead\">{site_cfg['tagline']}</p>
        <p class=\"micro\">One automated post per day. Charts, SEO tags, RSS feed, archive, and deployment can all run without admin intervention.</p>
      </header>
      <section class=\"grid\">
        {''.join(cards)}
      </section>
    </main>
    """
    html = render_page(
        title=site_cfg['site_name'],
        description=site_cfg['tagline'],
        canonical=f"{site_cfg['site_url'].rstrip('/')}/",
        body=body,
        site_cfg=site_cfg,
        article_json_ld=None,
        og_image=f"{site_cfg['site_url'].rstrip('/')}{posts[0]['chart_image']}" if posts else None,
    )
    (SITE / 'index.html').write_text(html, encoding='utf-8')


def render_post(post: Dict[str, Any], site_cfg: Dict[str, Any]) -> None:
    top_rows = []
    for c in post['impact']['countries'][:10]:
        top_rows.append(
            f"<tr><td>{c['name']}</td><td>{fmt_money(c['daily_loss_usd'])}/day</td><td>{fmt_pct(c['relative_pct_of_annual_gdp'])}</td></tr>"
        )
    paras = ''.join(f'<p>{p}</p>' for p in post['body_paragraphs'])
    body = f"""
    <main class=\"shell article-shell\">
      <article class=\"article\">
        <a class=\"back-link\" href=\"/\">← Back to archive</a>
        <p class=\"eyebrow\">AUTOMATED DAILY UPDATE</p>
        <h1>{post['title']}</h1>
        <div class=\"meta-row\">
          <time datetime=\"{post['date']}\">{post['date']}</time>
          <span>Source: {post['signal']['source']}</span>
          <span>Connectivity: {post['signal']['current_connectivity_pct']:.1f}%</span>
        </div>
        <img class=\"hero-chart\" src=\"{post['chart_image']}\" alt=\"{post['title']} chart\">
        <div class=\"stat-grid\">
          <section class=\"stat\"><span>Estimated global exposure</span><strong>{fmt_money(post['impact']['world_daily_loss_usd'])}/day</strong></section>
          <section class=\"stat\"><span>Annualized world GDP effect</span><strong>{fmt_pct(post['impact']['world_relative_pct_of_annual_gdp'])}</strong></section>
          <section class=\"stat\"><span>Outage day counter</span><strong>{post['signal']['days_since_drop']}</strong></section>
        </div>
        <div class=\"prose\">{paras}</div>
        <section class=\"table-wrap\">
          <h2>Top 10 exposed economies</h2>
          <table>
            <thead><tr><th>Economy</th><th>Estimated exposure</th><th>Annualized GDP effect</th></tr></thead>
            <tbody>{''.join(top_rows)}</tbody>
          </table>
        </section>
        <section class=\"note\">
          <h2>Method note</h2>
          <p>This is a model-based exposure estimate, not a verified macroeconomic loss statement. The daily figure is derived from a 2025 nominal GDP baseline, outage severity, and configurable country exposure weights designed to approximate market, energy, shipping, and regional sensitivity.</p>
          <p>Signal summary: {post['signal']['summary']}</p>
        </section>
      </article>
    </main>
    """
    article_ld = json.dumps({
        '@context': 'https://schema.org',
        '@type': 'Article',
        'headline': post['title'],
        'datePublished': post['date'],
        'dateModified': post['date'],
        'description': post['description'],
        'author': {'@type': 'Person', 'name': post['author']},
        'image': [f"{site_cfg['site_url'].rstrip('/')}{post['chart_image']}"],
        'mainEntityOfPage': f"{site_cfg['site_url'].rstrip('/')}/posts/{post['slug']}/"
    })
    html = render_page(
        title=f"{post['title']} | {site_cfg['site_name']}",
        description=post['description'],
        canonical=f"{site_cfg['site_url'].rstrip('/')}/posts/{post['slug']}/",
        body=body,
        site_cfg=site_cfg,
        article_json_ld=article_ld,
        og_image=f"{site_cfg['site_url'].rstrip('/')}{post['chart_image']}"
    )
    out_dir = SITE / 'posts' / post['slug']
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'index.html').write_text(html, encoding='utf-8')


def render_feed(posts: List[Dict[str, Any]], site_cfg: Dict[str, Any]) -> None:
    items = []
    for post in posts[:20]:
        url = f"{site_cfg['site_url'].rstrip('/')}/posts/{post['slug']}/"
        items.append(f"""
        <item>
          <title>{post['title']}</title>
          <link>{url}</link>
          <guid>{url}</guid>
          <pubDate>{datetime.fromisoformat(post['date']).strftime('%a, %d %b %Y 00:00:00 +0000')}</pubDate>
          <description><![CDATA[{post['description']}]]></description>
        </item>""")
    rss = f"""<?xml version=\"1.0\" encoding=\"UTF-8\" ?>
<rss version=\"2.0\">
<channel>
<title>{site_cfg['site_name']}</title>
<link>{site_cfg['site_url'].rstrip('/')}/</link>
<description>{site_cfg['tagline']}</description>
{''.join(items)}
</channel>
</rss>"""
    (SITE / 'feed.xml').write_text(rss, encoding='utf-8')


def render_sitemap(posts: List[Dict[str, Any]], site_cfg: Dict[str, Any]) -> None:
    urls = [f"<url><loc>{site_cfg['site_url'].rstrip('/')}/</loc></url>"]
    for post in posts:
        urls.append(f"<url><loc>{site_cfg['site_url'].rstrip('/')}/posts/{post['slug']}/</loc><lastmod>{post['date']}</lastmod></url>")
    xml = f"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<urlset xmlns=\"http://www.sitemaps.org/schemas/sitemap/0.9\">{''.join(urls)}</urlset>"""
    (SITE / 'sitemap.xml').write_text(xml, encoding='utf-8')


def write_css() -> None:
    CSS.mkdir(parents=True, exist_ok=True)
    (CSS / 'styles.css').write_text(
        """
:root{--bg:#f2f4f7;--card:#dfe5ec;--text:#1f2937;--muted:#5b6472;--accent:#5b5cf6;--surface:#ffffff;--stroke:#ccd4dd;}
*{box-sizing:border-box}html,body{margin:0;padding:0}body{font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;background:var(--bg);color:var(--text)}
a{color:inherit;text-decoration:none}img{max-width:100%;display:block}
.shell{max-width:1200px;margin:0 auto;padding:40px 24px 80px}.hero{padding:16px 0 24px;text-align:center}.eyebrow{letter-spacing:.18em;font-size:.78rem;font-weight:700;color:#8a93a3;margin:0 0 10px}.hero h1{font-size:clamp(2.2rem,5vw,3.6rem);margin:.1em 0}.lead{max-width:850px;margin:0 auto 12px;font-size:1.08rem;color:var(--muted)}.micro{color:#7b8494;font-size:.95rem}
.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:24px}.card{background:var(--card);border-radius:10px;padding:18px;box-shadow:0 1px 0 rgba(15,23,42,.05)}.card-head{display:flex;justify-content:space-between;gap:12px;align-items:center;font-size:.85rem;color:var(--muted);margin-bottom:12px}.badge{display:inline-flex;padding:4px 8px;border-radius:999px;background:#111827;color:#fff;font-size:.75rem;font-weight:600}.card h2{font-size:1.25rem;line-height:1.3;margin:0 0 10px}.card p{color:#364152;line-height:1.6;min-height:96px}.thumb-link img{border-radius:10px;border:1px solid rgba(15,23,42,.08);margin-top:10px}
.article-shell{max-width:900px}.article{background:var(--surface);padding:28px;border-radius:18px;box-shadow:0 10px 30px rgba(15,23,42,.06)}.back-link{display:inline-block;margin-bottom:16px;color:var(--accent);font-weight:600}.article h1{font-size:clamp(2rem,4vw,3rem);line-height:1.15;margin:.1em 0 .25em}.meta-row{display:flex;flex-wrap:wrap;gap:16px;color:var(--muted);font-size:.95rem;margin-bottom:20px}.hero-chart{border-radius:16px;border:1px solid var(--stroke);margin:16px 0 20px}.stat-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:18px 0 24px}.stat{background:#f8fafc;border:1px solid var(--stroke);border-radius:16px;padding:16px}.stat span{display:block;color:var(--muted);font-size:.9rem;margin-bottom:6px}.stat strong{font-size:1.25rem}.prose p,.note p{line-height:1.8;color:#344054}.table-wrap{margin-top:28px}.table-wrap h2,.note h2{font-size:1.3rem}table{width:100%;border-collapse:collapse;background:#fff;border:1px solid var(--stroke);border-radius:12px;overflow:hidden}th,td{text-align:left;padding:14px;border-bottom:1px solid var(--stroke)}th{background:#f8fafc;font-size:.92rem}.note{margin-top:28px;padding:18px;border-radius:16px;background:#fafbfc;border:1px solid var(--stroke)}
@media (max-width: 960px){.grid{grid-template-columns:1fr 1fr}.stat-grid{grid-template-columns:1fr}}@media (max-width: 680px){.grid{grid-template-columns:1fr}.shell{padding:24px 16px 60px}.article{padding:20px}.card p{min-height:auto}}
        """.strip(),
        encoding='utf-8'
    )


def ensure_social_card(site_cfg: Dict[str, Any]) -> None:
    target = IMG / 'social-card.png'
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 6.3), dpi=120)
    fig.patch.set_facecolor('#0b0f14')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.text(0.05, 0.74, f"{site_cfg['site_name']}", color='white', fontsize=28, fontweight='bold', va='top')
    ax.text(0.05, 0.56, 'Iran internet blackout', color='#98d08b', fontsize=19, va='top')
    ax.text(0.05, 0.36, 'Automated daily charts and model-based exposure estimates.', color='#94a3b8', fontsize=17)
    ax.text(0.05, 0.18, 'World + Top 10 economies • SEO ready • admin-free publishing', color='#98d08b', fontsize=16)
    fig.savefig(target, facecolor=fig.get_facecolor())
    plt.close(fig)


def build_one(for_date: date, site_cfg: Dict[str, Any], model_cfg: Dict[str, Any], signal_override: Optional[Signal] = None) -> Dict[str, Any]:
    raw_signal = signal_override or get_signal()
    threshold = float(model_cfg.get('full_shutdown_threshold_pct', 5.0))
    signal = align_signal_to_date(raw_signal, for_date, threshold)
    gdp = read_gdp()
    weights = read_weights()
    impact = compute_daily_impact(signal, gdp, weights, model_cfg)
    post = build_post(for_date, signal, impact, site_cfg)
    chart_path = IMG / f"{for_date.isoformat()}-world.png"
    make_chart(signal, impact, chart_path, title_date_end=for_date.isoformat())
    save_post(post)
    return post


def bootstrap_demo(site_cfg: Dict[str, Any], model_cfg: Dict[str, Any], days: int = 3) -> None:
    base_signal = read_fallback_signal()
    threshold = float(model_cfg.get('full_shutdown_threshold_pct', 5.0))
    outage_start = first_outage_date(base_signal, threshold)
    latest_date = datetime.fromisoformat(base_signal.series[-1]['date']).date()
    if outage_start is None:
        return
    if days <= 0:
        start_date = outage_start
    else:
        start_date = max(outage_start, latest_date - timedelta(days=days - 1))
    current_date = start_date
    while current_date <= latest_date:
        trimmed_signal = signal_trimmed_to_date(base_signal, current_date, threshold)
        if trimmed_signal is not None:
            build_one(current_date, site_cfg, model_cfg, signal_override=trimmed_signal)
        current_date += timedelta(days=1)


def main() -> None:
    SITE.mkdir(parents=True, exist_ok=True)
    IMG.mkdir(parents=True, exist_ok=True)
    CONTENT.mkdir(parents=True, exist_ok=True)

    site_cfg = read_site_config()
    model_cfg = read_model_config()
    write_css()
    ensure_social_card(site_cfg)

    if os.getenv('BOOTSTRAP_DEMO', '0') == '1' and not any(CONTENT.glob('*.json')):
        bootstrap_days = int(os.getenv('BOOTSTRAP_DAYS', '0'))
        bootstrap_demo(site_cfg, model_cfg, days=bootstrap_days)

    if os.getenv('BACKFILL_MISSING_HISTORY', '1') == '1':
        backfill_history(site_cfg, model_cfg, base_signal=read_fallback_signal())

    today = datetime.now(timezone.utc).date()
    existing = load_existing_posts()
    if not any(p['date'] == today.isoformat() for p in existing):
        build_one(today, site_cfg, model_cfg)

    posts = load_existing_posts()
    render_index(posts, site_cfg)
    for post in posts:
        render_post(post, site_cfg)
    render_feed(posts, site_cfg)
    render_sitemap(posts, site_cfg)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import requests
from matplotlib.ticker import FuncFormatter

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


def strip_merge_markers(text: str) -> str:
    if not isinstance(text, str):
        return text
    kept: List[str] = []
    for line in text.splitlines():
        marker = line.strip()
        if marker.startswith('<<<<<<<') or marker.startswith('=======') or marker.startswith('>>>>>>>'):
            continue
        kept.append(line)
    cleaned = '\n'.join(kept)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


def sanitize_data(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: sanitize_data(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_data(v) for v in value]
    if isinstance(value, str):
        return strip_merge_markers(value)
    return value


def load_json(path: Path) -> Any:
    return sanitize_data(json.loads(path.read_text(encoding='utf-8')))


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


def fmt_axis_usd(value_billions: float) -> str:
    if value_billions >= 1000:
        return f"${value_billions / 1000:.2f}T"
    return f"${value_billions:.0f}B"


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
        low = text.lower()
        if 'iran' in low and ('blackout' in low or 'internet' in low):
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
    outage_start = today - timedelta(days=max(days - 1, 0))
    start = outage_start - timedelta(days=4)
    series = []
    current_date = start
    while current_date <= today:
        pct = 96 - ((current_date - start).days % 4) if current_date < outage_start else connectivity
        series.append({'date': current_date.isoformat(), 'connectivity_pct': pct})
        current_date += timedelta(days=1)

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
    return live if live else read_fallback_signal()


def first_outage_date(signal: Signal, threshold_pct: float) -> Optional[date]:
    for item in sorted(signal.series, key=lambda x: x['date']):
        if float(item['connectivity_pct']) <= threshold_pct:
            return datetime.fromisoformat(item['date']).date()
    return None


def align_signal_to_date(base_signal: Signal, target_date: date, threshold_pct: float) -> Signal:
    series = sorted([dict(x) for x in base_signal.series], key=lambda x: x['date'])
    if not series:
        return base_signal
    last_date = datetime.fromisoformat(series[-1]['date']).date()
    last_pct = float(series[-1]['connectivity_pct'])
    while last_date < target_date:
        last_date += timedelta(days=1)
        series.append({'date': last_date.isoformat(), 'connectivity_pct': last_pct})
    trimmed = [x for x in series if datetime.fromisoformat(x['date']).date() <= target_date]
    outage_start = first_outage_date(base_signal, threshold_pct)
    days_since = (target_date - outage_start).days + 1 if outage_start and target_date >= outage_start else 1
    return Signal(
        source=base_signal.source,
        country=base_signal.country,
        current_connectivity_pct=float(trimmed[-1]['connectivity_pct']),
        days_since_drop=max(days_since, 1),
        summary=base_signal.summary,
        series=trimmed,
    )


def date_range(start_date: date, end_date: date) -> List[date]:
    days: List[date] = []
    current = start_date
    while current <= end_date:
        days.append(current)
        current += timedelta(days=1)
    return days


def outage_share_for_day(day_number: int, sensitivity: float, model: Dict[str, Any]) -> float:
    base = float(model['base_day1_loss_share_of_daily_gdp'])
    log_strength = float(model['log_escalation_strength'])
    linear_start = int(model['linear_escalation_start_day'])
    linear_per_day = float(model['linear_escalation_per_day_after_start'])
    cap = float(model['max_loss_share_of_daily_gdp'])
    multiplier = 1.0 + log_strength * math.log1p(max(day_number - 1, 0))
    if day_number > linear_start:
        multiplier += linear_per_day * (day_number - linear_start)
    share = base * sensitivity * multiplier
    return min(cap, share)


def shock_ratio_for_day(day_number: int, model: Dict[str, Any]) -> float:
    start = float(model.get('shock_day1_share_of_direct_daily_loss', 0.40))
    floor = float(model.get('shock_floor_share_of_direct_daily_loss', 0.05))
    half_life = float(model.get('shock_half_life_days', 5.0))
    if half_life <= 0:
        return floor
    decay = math.exp(-math.log(2) * max(day_number - 1, 0) / half_life)
    return floor + (start - floor) * decay


def compute_blackout_simulation(signal: Signal, gdp: Dict[str, Any], weights: Dict[str, float], model: Dict[str, Any]) -> Dict[str, Any]:
    threshold = float(model.get('full_shutdown_threshold_pct', 5.0))
    outage_start = first_outage_date(signal, threshold) or datetime.fromisoformat(signal.series[-1]['date']).date()
    dates = [
        datetime.fromisoformat(x['date']).date()
        for x in sorted(signal.series, key=lambda x: x['date'])
        if datetime.fromisoformat(x['date']).date() >= outage_start
    ]
    if not dates:
        dates = [datetime.fromisoformat(signal.series[-1]['date']).date()]

    entries: List[Dict[str, Any]] = []
    for code, meta in gdp.items():
        sensitivity = float(weights.get(code, 1.0))
        annual_gdp = float(meta['gdp_usd'])
        daily_gdp = annual_gdp / 365.0
        daily_series = []
        cumulative = 0.0
        cumulative_shock = 0.0
        for idx, current_date in enumerate(dates, start=1):
            share = outage_share_for_day(idx, sensitivity, model)
            daily_loss = daily_gdp * share
            shock_ratio = shock_ratio_for_day(idx, model)
            shock_daily = daily_loss * shock_ratio
            cumulative += daily_loss
            cumulative_shock += shock_daily
            daily_series.append({
                'date': current_date.isoformat(),
                'day': idx,
                'loss_share_of_daily_gdp': share,
                'daily_loss_usd': daily_loss,
                'cumulative_loss_usd': cumulative,
                'shock_share_of_direct_daily_loss': shock_ratio,
                'shock_daily_usd': shock_daily,
                'cumulative_shock_usd': cumulative_shock,
                'shock_adjusted_daily_total_usd': daily_loss + shock_daily,
                'shock_adjusted_cumulative_total_usd': cumulative + cumulative_shock,
            })
        entries.append({
            'code': code,
            'name': meta['name'],
            'rank': int(meta.get('rank', 0)),
            'gdp_usd': annual_gdp,
            'sensitivity': sensitivity,
            'daily_loss_usd': daily_series[-1]['daily_loss_usd'],
            'cumulative_loss_usd': daily_series[-1]['cumulative_loss_usd'],
            'loss_share_of_daily_gdp': daily_series[-1]['loss_share_of_daily_gdp'],
            'shock_daily_usd': daily_series[-1]['shock_daily_usd'],
            'cumulative_shock_usd': daily_series[-1]['cumulative_shock_usd'],
            'shock_share_of_direct_daily_loss': daily_series[-1]['shock_share_of_direct_daily_loss'],
            'shock_adjusted_daily_total_usd': daily_series[-1]['shock_adjusted_daily_total_usd'],
            'shock_adjusted_cumulative_total_usd': daily_series[-1]['shock_adjusted_cumulative_total_usd'],
            'cumulative_pct_of_annual_gdp': (daily_series[-1]['cumulative_loss_usd'] / annual_gdp) * 100.0,
            'series': daily_series,
        })

    world = next(x for x in entries if x['code'] == 'WORLD')
    countries = sorted([x for x in entries if x['code'] != 'WORLD'], key=lambda x: x['daily_loss_usd'], reverse=True)
    return {
        'reference_days': len(dates),
        'world': world,
        'countries': countries,
        'dates': [d.isoformat() for d in dates],
    }


def _fmt_billions_tick(x: float, _pos: int) -> str:
    if abs(x) >= 1000:
        return f"${x / 1000:.1f}T"
    return f"${x:.0f}B"


def make_chart(simulation: Dict[str, Any], chart_path: Path, title_date_end: str) -> None:
    chart_path.parent.mkdir(parents=True, exist_ok=True)

    def _series_in_billions(entry: Dict[str, Any], key: str) -> List[float]:
        return [point[key] / 1_000_000_000 for point in sorted(entry['series'], key=lambda x: x['date'])]

    world_series = sorted(simulation['world']['series'], key=lambda x: x['date'])
    usa = next((x for x in simulation['countries'] if x['code'] == 'USA'), None)
    china = next((x for x in simulation['countries'] if x['code'] == 'CHN'), None)

    dates = [datetime.fromisoformat(x['date']).date() for x in world_series]
    world_billions = _series_in_billions(simulation['world'], 'cumulative_loss_usd')
    world_total_billions = _series_in_billions(simulation['world'], 'shock_adjusted_cumulative_total_usd')
    usa_billions = _series_in_billions(usa, 'cumulative_loss_usd') if usa else []
    china_billions = _series_in_billions(china, 'cumulative_loss_usd') if china else []

    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    fig.patch.set_facecolor('#0b0f14')
    ax.set_facecolor('#0b0f14')

    world_line, = ax.plot(dates, world_billions, linewidth=3.0, color='#98d08b', label='World cumulative', zorder=3)
    ax.fill_between(dates, world_billions, 0, color='#98d08b', alpha=0.10, zorder=1)

    world_total_line, = ax.plot(dates, world_total_billions, linewidth=3.3, color='#e879f9', label='World cumulative + shock', zorder=5)
    ax.fill_between(dates, world_total_billions, world_billions, color='#fb7185', alpha=0.16, zorder=2)

    usa_line = None
    china_line = None
    if usa_billions:
        usa_line, = ax.plot(dates, usa_billions, linewidth=2.4, color='#60a5fa', label='United States cumulative', zorder=4)
    if china_billions:
        china_line, = ax.plot(dates, china_billions, linewidth=2.4, color='#f59e0b', label='China cumulative', zorder=4)

    all_series = [world_billions, world_total_billions]
    if usa_billions:
        all_series.append(usa_billions)
    if china_billions:
        all_series.append(china_billions)
    ymax = max(max(series) for series in all_series if series) * 1.12 if all_series else 1
    ax.set_ylim(0, ymax)

    ax.tick_params(axis='x', colors='#cbd5e1', labelsize=10)
    ax.tick_params(axis='y', colors='#cbd5e1', labelsize=11)
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_billions_tick))
    if len(dates) <= 4:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        if len(dates) == 1:
            ax.set_xlim(dates[0] - timedelta(days=1), dates[0] + timedelta(days=1))
    else:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=7))
    try:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%-m/%-d'))
    except Exception:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, axis='both', color='white', alpha=0.12, linewidth=0.8)

    ax.set_title(
        f'Simulated Cumulative Blackout Loss - World: {dates[0].isoformat()} to {title_date_end} UTC',
        color='#e5e7eb',
        fontsize=18,
        pad=22,
    )
    ax.set_ylabel('Cumulative loss', color='#cbd5e1', fontsize=13, labelpad=16)

    handles = [world_line, world_total_line]
    if usa_line:
        handles.append(usa_line)
    if china_line:
        handles.append(china_line)
    legend = ax.legend(handles=handles, loc='upper left', facecolor='#0b0f14', edgecolor='none', framealpha=0.0, fontsize=11)
    for text_item in legend.get_texts():
        label = text_item.get_text()
        if label == 'World cumulative':
            text_item.set_color('#98d08b')
        elif label == 'World cumulative + shock':
            text_item.set_color('#e879f9')
        elif label == 'United States cumulative':
            text_item.set_color('#60a5fa')
        elif label == 'China cumulative':
            text_item.set_color('#f59e0b')
        else:
            text_item.set_color('#e5e7eb')

    fig.text(0.52, 0.09, 'World', color='#98d08b', fontsize=11, fontweight='bold')
    fig.text(0.565, 0.09, fmt_money(world_series[-1]['cumulative_loss_usd']) if world_series else '$0', color='#e5e7eb', fontsize=11)
    fig.text(0.52, 0.06, 'World + shock', color='#e879f9', fontsize=11, fontweight='bold')
    fig.text(0.64, 0.06, fmt_money(world_series[-1]['shock_adjusted_cumulative_total_usd']) if world_series else '$0', color='#e5e7eb', fontsize=11)
    if usa:
        fig.text(0.50, 0.03, 'US', color='#60a5fa', fontsize=11, fontweight='bold')
        fig.text(0.525, 0.03, fmt_money(usa['cumulative_loss_usd']), color='#e5e7eb', fontsize=11)
    if china:
        fig.text(0.60, 0.03, 'China', color='#f59e0b', fontsize=11, fontweight='bold')
        fig.text(0.66, 0.03, fmt_money(china['cumulative_loss_usd']), color='#e5e7eb', fontsize=11)
    fig.text(0.80, 0.03, 'Shock added', color='#fb7185', fontsize=9.5, fontweight='bold')
    fig.text(0.98, 0.03, fmt_money(simulation['world']['cumulative_shock_usd']), color='#e5e7eb', fontsize=11, ha='right')

    plt.tight_layout(rect=[0.03, 0.12, 0.98, 0.94])
    fig.savefig(chart_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)

def trim_meta_description(text: str, limit: int = 158) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) <= limit:
        return text
    trimmed = text[: limit - 1]
    if ' ' in trimmed:
        trimmed = trimmed.rsplit(' ', 1)[0]
    return trimmed.rstrip(' ,;:-') + '…'


def build_seo_description(today: date, simulation: Dict[str, Any]) -> str:
    day_num = simulation['reference_days']
    world = simulation['world']
    countries = simulation['countries'][:3]
    top1 = countries[0] if len(countries) > 0 else {'name': 'Top economy', 'daily_loss_usd': 0}
    top2 = countries[1] if len(countries) > 1 else top1
    top3 = countries[2] if len(countries) > 2 else top2
    world_daily = fmt_money(world['daily_loss_usd'])
    world_cum = fmt_money(world['cumulative_loss_usd'])

    templates = [
        f"{today.isoformat()} internet blackout simulation: day {day_num}, world scenario {world_daily}/day. {top1['name']} leads at {fmt_money(top1['daily_loss_usd'])}/day.",
        f"Day {day_num} economic impact simulation: if internet went dark locally, {top1['name']} would lose {fmt_money(top1['daily_loss_usd'])}/day; world {world_daily}/day.",
        f"{today.isoformat()} update: simulated internet blackout losses put {top1['name']} first, {top2['name']} second, with world exposure at {world_daily}/day.",
        f"Internet blackout day {day_num}: modeled daily loss reaches {world_daily} worldwide, with {top1['name']} and {top2['name']} leading the country ranking.",
        f"SEO update for {today.isoformat()}: simulated blackout losses show {top1['name']} at {fmt_money(top1['daily_loss_usd'])}/day and world exposure at {world_daily}/day.",
        f"Day {day_num} blackout simulation report: {top1['name']} {fmt_money(top1['daily_loss_usd'])}/day, {top2['name']} {fmt_money(top2['daily_loss_usd'])}/day, world {world_daily}/day.",
        f"{today.isoformat()} blackout simulation archive: {top1['name']}, {top2['name']} and {top3['name']} top the loss table; world scenario totals {world_daily}/day.",
        f"Internet outage simulation day {day_num}: world daily loss {world_daily}, cumulative world loss {world_cum}; {top1['name']} remains the largest economy case.",
    ]
    return trim_meta_description(templates[(day_num - 1) % len(templates)])


def build_excerpt(today: date, simulation: Dict[str, Any]) -> str:
    day_num = simulation['reference_days']
    world = simulation['world']
    countries = simulation['countries'][:2]
    names = ' and '.join(c['name'] for c in countries)
    text = (
        f"{today.isoformat()} / day {day_num}: a same-length local internet blackout simulation for the world and top economies, "
        f"with {names} leading the modeled daily losses and the world scenario at {fmt_money(world['daily_loss_usd'])}/day."
    )
    return trim_meta_description(text, limit=220)


def build_post(today: date, signal: Signal, simulation: Dict[str, Any], site_cfg: Dict[str, Any], model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    day_num = simulation['reference_days']
    title = f"Internet blackout simulation update — day {day_num}"
    slug = f"{today.isoformat()}-{slugify(title)}"
    top3 = simulation['countries'][:3]
    bullet = '; '.join(f"{c['name']}: {fmt_money(c['daily_loss_usd'])}/day" for c in top3)
    description = build_seo_description(today, simulation)
    excerpt = build_excerpt(today, simulation)
    body = [
        f"This post does not say these economies are currently losing money because Iran is offline. It uses Iran's blackout duration as a time clock and simulates what a same-length domestic internet blackout would cost each economy on day {day_num}.",
        f"Under the world scenario, a blackout of this length implies {fmt_money(simulation['world']['daily_loss_usd'])} in daily GDP loss and {fmt_money(simulation['world']['cumulative_loss_usd'])} in cumulative loss since day 1.",
        f"For the large-economy scenarios, the biggest simulated daily losses today are: {bullet}.",
        "Method: 2025 nominal GDP baseline × country-specific digital sensitivity coefficient × an outage-duration escalation curve that gets harsher as the blackout continues.",
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
        'simulation': simulation,
        'chart_image': f"/assets/img/{today.isoformat()}-world.png",
        'author': site_cfg['author'],
        'model_note': model_cfg.get('notes', ''),
    }


def save_post(post: Dict[str, Any]) -> None:
    save_json(CONTENT / f"{post['slug']}.json", post)


def load_existing_posts() -> List[Dict[str, Any]]:
    posts = []
    for path in sorted(CONTENT.glob('*.json')):
        try:
            posts.append(load_json(path))
        except Exception:
            continue
    posts.sort(key=lambda x: x['date'], reverse=True)
    return posts


def footer_html(site_cfg: Dict[str, Any]) -> str:
    x_url = site_cfg.get('x_url', '').strip()
    if not x_url:
        return ''
    return f"""
    <footer class=\"site-footer\">
      <a class=\"x-link\" href=\"{escape(x_url)}\" target=\"_blank\" rel=\"noopener noreferrer\" aria-label=\"Follow on X\">
        <svg viewBox=\"0 0 24 24\" aria-hidden=\"true\"><path d=\"M18.901 1.153h3.68l-8.04 9.19L24 22.846h-7.406l-5.8-7.584-6.637 7.584H.474l8.6-9.83L0 1.154h7.594l5.243 6.932 6.064-6.932Zm-1.298 19.479h2.039L6.486 3.26H4.298l13.305 17.372Z\"></path></svg>
      </a>
    </footer>
    """


def render_page(title: str, description: str, canonical: str, body: str, site_cfg: Dict[str, Any], article_json_ld: Optional[str] = None, og_image: Optional[str] = None, include_chart_js: bool = False) -> str:
    json_ld = f'<script type="application/ld+json">{article_json_ld}</script>' if article_json_ld else ''
    og = og_image or f"{site_cfg['site_url'].rstrip('/')}/assets/img/social-card.png"
    footer = footer_html(site_cfg)
    chart_script = '<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>' if include_chart_js else ''
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>{escape(title)}</title>
  <meta name=\"description\" content=\"{escape(description)}\">
  <link rel=\"canonical\" href=\"{escape(canonical)}\">
  <meta property=\"og:type\" content=\"article\">
  <meta property=\"og:title\" content=\"{escape(title)}\">
  <meta property=\"og:description\" content=\"{escape(description)}\">
  <meta property=\"og:url\" content=\"{escape(canonical)}\">
  <meta property=\"og:image\" content=\"{escape(og)}\">
  <meta name=\"twitter:card\" content=\"summary_large_image\">
  <meta name=\"twitter:title\" content=\"{escape(title)}\">
  <meta name=\"twitter:description\" content=\"{escape(description)}\">
  <meta name=\"twitter:image\" content=\"{escape(og)}\">
  <meta name=\"twitter:site\" content=\"{escape(site_cfg.get('x_handle', ''))}\">
  <meta name=\"twitter:creator\" content=\"{escape(site_cfg.get('x_handle', ''))}\">
  <link rel=\"alternate\" type=\"application/rss+xml\" title=\"{escape(site_cfg['site_name'])} RSS\" href=\"{escape(site_cfg['site_url'].rstrip('/'))}/feed.xml\">
  <link rel=\"stylesheet\" href=\"/assets/css/styles.css\">
  {json_ld}
  {chart_script}
</head>
<body>
  {body}
  {footer}
</body>
</html>"""


def render_index(posts: List[Dict[str, Any]], site_cfg: Dict[str, Any]) -> None:
    cards = []
    cards_to_render = posts[: int(site_cfg.get('cards_per_page', len(posts)))]
    for post in cards_to_render:
        cards.append(f"""
        <article class=\"card\">
          <div class=\"card-head\">
            <span class=\"badge\">Daily Simulation</span>
            <time datetime=\"{escape(post['date'])}\">{escape(post['date'])}</time>
          </div>
          <h2><a href=\"/posts/{escape(post['slug'])}/\">{escape(post['title'])}</a></h2>
          <p>{escape(post['excerpt'])}</p>
          <a class=\"thumb-link\" href=\"/posts/{escape(post['slug'])}/\"><img src=\"{escape(post['chart_image'])}\" alt=\"{escape(post['title'])} chart\"></a>
        </article>
        """)
    body = f"""
    <main class=\"shell\">
      <header class=\"hero\">
        <p class=\"eyebrow\">DAILY BLACKOUT SIMULATION</p>
        <h1>{escape(site_cfg['site_name'])}</h1>
        <p class=\"lead\">{escape(site_cfg['tagline'])}</p>
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
        og_image=f"{site_cfg['site_url'].rstrip('/')}{posts[0]['chart_image']}" if posts else None,
    )
    (SITE / 'index.html').write_text(html, encoding='utf-8')


def chart_data_for_post(post: Dict[str, Any]) -> List[Dict[str, Any]]:
    world_series = sorted(post['simulation']['world']['series'], key=lambda x: x['date'])
    usa_entry = next((x for x in post['simulation']['countries'] if x['code'] == 'USA'), None)
    china_entry = next((x for x in post['simulation']['countries'] if x['code'] == 'CHN'), None)
    usa_series = {item['date']: item for item in sorted(usa_entry['series'], key=lambda x: x['date'])} if usa_entry else {}
    china_series = {item['date']: item for item in sorted(china_entry['series'], key=lambda x: x['date'])} if china_entry else {}

    points = []
    for item in world_series:
        usa_point = usa_series.get(item['date'], {})
        china_point = china_series.get(item['date'], {})
        points.append(
            {
                'date': item['date'],
                'day': item['day'],
                'world_cumulative_billions': round(item['cumulative_loss_usd'] / 1_000_000_000, 2),
                'world_cumulative_usd': round(item['cumulative_loss_usd'], 2),
                'world_total_cumulative_billions': round(item['shock_adjusted_cumulative_total_usd'] / 1_000_000_000, 2),
                'world_total_cumulative_usd': round(item['shock_adjusted_cumulative_total_usd'], 2),
                'world_cumulative_shock_billions': round(item['cumulative_shock_usd'] / 1_000_000_000, 2),
                'world_cumulative_shock_usd': round(item['cumulative_shock_usd'], 2),
                'world_daily_billions': round(item['daily_loss_usd'] / 1_000_000_000, 2),
                'world_daily_usd': round(item['daily_loss_usd'], 2),
                'world_shock_billions': round(item['shock_daily_usd'] / 1_000_000_000, 2),
                'world_shock_usd': round(item['shock_daily_usd'], 2),
                'usa_cumulative_billions': round(float(usa_point.get('cumulative_loss_usd', 0.0)) / 1_000_000_000, 2),
                'usa_cumulative_usd': round(float(usa_point.get('cumulative_loss_usd', 0.0)), 2),
                'china_cumulative_billions': round(float(china_point.get('cumulative_loss_usd', 0.0)) / 1_000_000_000, 2),
                'china_cumulative_usd': round(float(china_point.get('cumulative_loss_usd', 0.0)), 2),
            }
        )
    return points


def render_interactive_chart(post: Dict[str, Any]) -> str:
    chart_points = chart_data_for_post(post)
    labels = [p['date'] for p in chart_points]
    world_values = [p['world_cumulative_billions'] for p in chart_points]
    world_total_values = [p['world_total_cumulative_billions'] for p in chart_points]
    usa_values = [p['usa_cumulative_billions'] for p in chart_points]
    china_values = [p['china_cumulative_billions'] for p in chart_points]
    shock_values = [p['world_shock_billions'] for p in chart_points]
    payload = json.dumps(chart_points, ensure_ascii=False)
    labels_json = json.dumps(labels, ensure_ascii=False)
    world_json = json.dumps(world_values, ensure_ascii=False)
    world_total_json = json.dumps(world_total_values, ensure_ascii=False)
    usa_json = json.dumps(usa_values, ensure_ascii=False)
    china_json = json.dumps(china_values, ensure_ascii=False)
    shock_json = json.dumps(shock_values, ensure_ascii=False)
    chart_id = f"world-chart-{post['date'].replace('-', '')}"
    shock_chart_id = f"world-shock-chart-{post['date'].replace('-', '')}"
    summary_id = f"world-chart-summary-{post['date'].replace('-', '')}"

    usa_now = next((x for x in post['simulation']['countries'] if x['code'] == 'USA'), {'cumulative_loss_usd': 0.0})
    china_now = next((x for x in post['simulation']['countries'] if x['code'] == 'CHN'), {'cumulative_loss_usd': 0.0})

    return f"""
        <section class="interactive-chart-block">
          <div class="chart-header">
            <h2>Interactive cumulative loss chart</h2>
            <p>The main chart shows world cumulative loss, world cumulative + accumulated shock, the United States, and China. The red mini-chart below isolates the daily shock premium so it does not distort the cumulative view.</p>
          </div>
          <div class="chart-summary" id="{summary_id}">Day {post['simulation']['reference_days']} · {post['date']} · World {fmt_money(post['simulation']['world']['cumulative_loss_usd'])} · World + shock {fmt_money(post['simulation']['world']['shock_adjusted_cumulative_total_usd'])} · US {fmt_money(usa_now['cumulative_loss_usd'])} · China {fmt_money(china_now['cumulative_loss_usd'])} · Shock {fmt_money(post['simulation']['world']['shock_daily_usd'])}/day · Shock added {fmt_money(post['simulation']['world']['cumulative_shock_usd'])}</div>
          <div class="chart-stage chart-stage-main">
            <canvas id="{chart_id}"></canvas>
          </div>
          <div class="chart-stage chart-stage-mini">
            <canvas id="{shock_chart_id}"></canvas>
          </div>
          <noscript><img class="hero-chart" src="{escape(post['chart_image'])}" alt="{escape(post['title'])} chart"></noscript>
        </section>
        <script>
        (function() {{
          const chartPoints = {payload};
          const labels = {labels_json};
          const worldValues = {world_json};
          const worldTotalValues = {world_total_json};
          const usaValues = {usa_json};
          const chinaValues = {china_json};
          const shockValues = {shock_json};
          const canvas = document.getElementById({json.dumps(chart_id)});
          const shockCanvas = document.getElementById({json.dumps(shock_chart_id)});
          const summary = document.getElementById({json.dumps(summary_id)});
          if (!canvas || !shockCanvas || typeof Chart === 'undefined') return;

          const formatUSD = (value) => {{
            const abs = Math.abs(value);
            if (abs >= 1e12) return '$' + (value / 1e12).toFixed(2) + 'T';
            if (abs >= 1e9) return '$' + (value / 1e9).toFixed(2) + 'B';
            if (abs >= 1e6) return '$' + (value / 1e6).toFixed(2) + 'M';
            return '$' + Math.round(value).toLocaleString();
          }};
          const formatAxis = (billions) => billions >= 1000 ? '$' + (billions / 1000).toFixed(2) + 'T' : '$' + Math.round(billions) + 'B';
          const updateSummary = (point) => {{
            summary.textContent =
              'Day ' + point.day + ' · ' + point.date +
              ' · World ' + formatUSD(point.world_cumulative_usd) +
              ' · World + shock ' + formatUSD(point.world_total_cumulative_usd) +
              ' · US ' + formatUSD(point.usa_cumulative_usd) +
              ' · China ' + formatUSD(point.china_cumulative_usd) +
              ' · Shock ' + formatUSD(point.world_shock_usd) + '/day' +
              ' · Shock added ' + formatUSD(point.world_cumulative_shock_usd);
          }};
          const mainChart = new Chart(canvas, {{
            type: 'line',
            data: {{
              labels,
              datasets: [
                {{
                  label: 'World cumulative',
                  data: worldValues,
                  yAxisID: 'y',
                  borderColor: '#98d08b',
                  backgroundColor: 'rgba(152, 208, 139, 0.10)',
                  fill: true,
                  borderWidth: 3,
                  tension: 0.28,
                  pointRadius: 0,
                  pointHoverRadius: 5,
                  pointHitRadius: 20,
                  order: 3
                }},
                {{
                  label: 'World cumulative + shock',
                  data: worldTotalValues,
                  yAxisID: 'y',
                  borderColor: '#e879f9',
                  backgroundColor: 'rgba(251, 113, 133, 0.18)',
                  fill: '-1',
                  borderWidth: 3.3,
                  tension: 0.28,
                  pointRadius: 0,
                  pointHoverRadius: 5,
                  pointHitRadius: 20,
                  order: 2
                }},
                {{
                  label: 'United States cumulative',
                  data: usaValues,
                  yAxisID: 'y',
                  borderColor: '#60a5fa',
                  backgroundColor: 'rgba(96, 165, 250, 0)',
                  fill: false,
                  borderWidth: 2.4,
                  tension: 0.28,
                  pointRadius: 0,
                  pointHoverRadius: 5,
                  pointHitRadius: 20,
                  order: 4
                }},
                {{
                  label: 'China cumulative',
                  data: chinaValues,
                  yAxisID: 'y',
                  borderColor: '#f59e0b',
                  backgroundColor: 'rgba(245, 158, 11, 0)',
                  fill: false,
                  borderWidth: 2.4,
                  tension: 0.28,
                  pointRadius: 0,
                  pointHoverRadius: 5,
                  pointHitRadius: 20,
                  order: 4
                }}
              ]
            }},
            options: {{
              responsive: true,
              maintainAspectRatio: false,
              interaction: {{ mode: 'index', intersect: false }},
              plugins: {{
                legend: {{
                  display: true,
                  position: 'bottom',
                  labels: {{
                    color: '#cbd5e1',
                    usePointStyle: true,
                    boxWidth: 10,
                    boxHeight: 10
                  }}
                }},
                tooltip: {{
                  backgroundColor: '#111827',
                  titleColor: '#f9fafb',
                  bodyColor: '#f9fafb',
                  borderColor: 'rgba(255,255,255,0.14)',
                  borderWidth: 1,
                  displayColors: true,
                  callbacks: {{
                    title: (items) => items[0] ? items[0].label : '',
                    label: (context) => {{
                      const point = chartPoints[context.dataIndex];
                      if (context.dataset.label === 'World cumulative') return 'World · Day ' + point.day + ' · ' + formatUSD(point.world_cumulative_usd);
                      if (context.dataset.label === 'World cumulative + shock') return 'World + shock · Day ' + point.day + ' · ' + formatUSD(point.world_total_cumulative_usd);
                      if (context.dataset.label === 'United States cumulative') return 'United States · Day ' + point.day + ' · ' + formatUSD(point.usa_cumulative_usd);
                      return 'China · Day ' + point.day + ' · ' + formatUSD(point.china_cumulative_usd);
                    }}
                  }}
                }}
              }},
              scales: {{
                x: {{
                  ticks: {{ color: '#cbd5e1', maxRotation: 0, autoSkip: true, maxTicksLimit: 7 }},
                  grid: {{ color: 'rgba(255,255,255,0.10)' }}
                }},
                y: {{
                  beginAtZero: true,
                  position: 'left',
                  ticks: {{ color: '#cbd5e1', callback: (value) => formatAxis(value) }},
                  title: {{ display: true, text: 'Cumulative loss', color: '#cbd5e1' }},
                  grid: {{ color: 'rgba(255,255,255,0.10)' }}
                }}
              }},
              onHover: (_event, activeEls) => {{
                if (!activeEls || !activeEls.length) return;
                const index = activeEls[0].index;
                const point = chartPoints[index];
                if (point) updateSummary(point);
              }}
            }}
          }});

          const shockChart = new Chart(shockCanvas, {{
            type: 'line',
            data: {{
              labels,
              datasets: [
                {{
                  label: 'World shock premium (daily)',
                  data: shockValues,
                  borderColor: '#ef4444',
                  backgroundColor: 'rgba(239, 68, 68, 0.18)',
                  fill: true,
                  borderWidth: 2.8,
                  tension: 0.22,
                  pointRadius: 0,
                  pointHoverRadius: 4,
                  pointHitRadius: 20
                }}
              ]
            }},
            options: {{
              responsive: true,
              maintainAspectRatio: false,
              interaction: {{ mode: 'index', intersect: false }},
              plugins: {{
                legend: {{
                  display: true,
                  position: 'bottom',
                  labels: {{ color: '#fca5a5', usePointStyle: true, boxWidth: 10, boxHeight: 10 }}
                }},
                tooltip: {{
                  backgroundColor: '#111827',
                  titleColor: '#f9fafb',
                  bodyColor: '#f9fafb',
                  borderColor: 'rgba(255,255,255,0.14)',
                  borderWidth: 1,
                  callbacks: {{
                    title: (items) => items[0] ? items[0].label : '',
                    label: (context) => {{
                      const point = chartPoints[context.dataIndex];
                      return 'Shock premium · Day ' + point.day + ' · ' + formatUSD(point.world_shock_usd) + '/day';
                    }}
                  }}
                }}
              }},
              scales: {{
                x: {{
                  ticks: {{ color: '#cbd5e1', maxRotation: 0, autoSkip: true, maxTicksLimit: 7 }},
                  grid: {{ color: 'rgba(255,255,255,0.08)' }}
                }},
                y: {{
                  beginAtZero: true,
                  ticks: {{ color: '#fca5a5', callback: (value) => formatAxis(value) }},
                  title: {{ display: true, text: 'World shock premium / day', color: '#fca5a5' }},
                  grid: {{ color: 'rgba(255,255,255,0.08)' }}
                }}
              }},
              onHover: (_event, activeEls) => {{
                if (!activeEls || !activeEls.length) return;
                const index = activeEls[0].index;
                const point = chartPoints[index];
                if (point) updateSummary(point);
              }}
            }}
          }});

          const lastPoint = chartPoints[chartPoints.length - 1];
          if (lastPoint) updateSummary(lastPoint);
        }})();
        </script>
    """

def render_post(post: Dict[str, Any], site_cfg: Dict[str, Any]) -> None:
    top_rows = []
    for c in post['simulation']['countries'][:10]:
        top_rows.append(
            f"<tr><td>{escape(c['name'])}</td><td>{fmt_money(c['daily_loss_usd'])}/day</td><td>{fmt_money(c['cumulative_loss_usd'])}</td><td>{fmt_pct(c['cumulative_pct_of_annual_gdp'])}</td></tr>"
        )
    paras = ''.join(f'<p>{escape(p)}</p>' for p in post['body_paragraphs'])
    interactive_chart = render_interactive_chart(post)
    body = f"""
    <main class="shell article-shell">
      <article class="article">
        <a class="back-link" href="/">← Back to archive</a>
        <p class="eyebrow">DAILY BLACKOUT SIMULATION</p>
        <h1>{escape(post['title'])}</h1>
        <div class="meta-row">
          <time datetime="{escape(post['date'])}">{escape(post['date'])}</time>
          <span>Reference clock: Iran blackout day {post['simulation']['reference_days']}</span>
          <span>Scenario: local internet outage in each economy</span>
        </div>
        {interactive_chart}
        <div class="stat-grid">
          <section class="stat stat-primary"><span>World scenario cumulative loss</span><strong>{fmt_money(post['simulation']['world']['cumulative_loss_usd'])}</strong></section>
          <section class="stat stat-primary"><span>World cumulative + shock</span><strong>{fmt_money(post['simulation']['world']['shock_adjusted_cumulative_total_usd'])}</strong></section>
          <section class="stat"><span>World scenario daily loss</span><strong>{fmt_money(post['simulation']['world']['daily_loss_usd'])}/day</strong></section>
          <section class="stat stat-danger"><span>World shock premium today</span><strong>{fmt_money(post['simulation']['world']['shock_daily_usd'])}/day</strong></section>
          <section class="stat"><span>Reference outage day</span><strong>{post['simulation']['reference_days']}</strong></section>
        </div>
        <div class="prose">{paras}</div>
        <section class="table-wrap">
          <h2>Top 10 economy simulations</h2>
          <table>
            <thead><tr><th>Economy</th><th>Simulated loss today</th><th>Cumulative simulated loss</th><th>Loss vs 2025 GDP</th></tr></thead>
            <tbody>{''.join(top_rows)}</tbody>
          </table>
        </section>
        <section class="note">
          <h2>Method note</h2>
          <p>This is a simulation, not a claim that these losses are happening right now. The model uses the length of Iran's blackout as a day counter, then asks what the same-duration domestic internet blackout would cost for the world scenario and the top 10 economies by 2025 GDP.</p>
          <p>The red mini-chart isolates the modeled world shock premium: a separate first-wave disruption layer that starts high, then decays over time as markets partially adapt.</p>
          <p>The pink line in the main chart is the cumulative total after adding that shock layer to the world cumulative loss. The shaded band between pink and green shows how much of the total burden comes from accumulated shock rather than the direct blackout-loss model alone.</p>
          <p>{escape(post['model_note'])}</p>
          <p>Signal summary: {escape(post['signal']['summary'])}</p>
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
        'mainEntityOfPage': f"{site_cfg['site_url'].rstrip('/')}/posts/{post['slug']}/",
    })
    html = render_page(
        title=f"{post['title']} | {site_cfg['site_name']}",
        description=post['description'],
        canonical=f"{site_cfg['site_url'].rstrip('/')}/posts/{post['slug']}/",
        body=body,
        site_cfg=site_cfg,
        article_json_ld=article_ld,
        og_image=f"{site_cfg['site_url'].rstrip('/')}{post['chart_image']}",
        include_chart_js=True,
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
          <title>{escape(post['title'])}</title>
          <link>{escape(url)}</link>
          <guid>{escape(url)}</guid>
          <pubDate>{datetime.fromisoformat(post['date']).strftime('%a, %d %b %Y 00:00:00 +0000')}</pubDate>
          <description><![CDATA[{post['description']}]]></description>
        </item>""")
    rss = f"""<?xml version=\"1.0\" encoding=\"UTF-8\" ?>
<rss version=\"2.0\">
<channel>
<title>{escape(site_cfg['site_name'])}</title>
<link>{escape(site_cfg['site_url'].rstrip('/'))}/</link>
<description>{escape(site_cfg['tagline'])}</description>
{''.join(items)}
</channel>
</rss>"""
    (SITE / 'feed.xml').write_text(rss, encoding='utf-8')


def render_sitemap(posts: List[Dict[str, Any]], site_cfg: Dict[str, Any]) -> None:
    urls = [f"<url><loc>{escape(site_cfg['site_url'].rstrip('/'))}/</loc></url>"]
    for post in posts:
        urls.append(f"<url><loc>{escape(site_cfg['site_url'].rstrip('/'))}/posts/{escape(post['slug'])}/</loc><lastmod>{escape(post['date'])}</lastmod></url>")
    xml = f"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<urlset xmlns=\"http://www.sitemaps.org/schemas/sitemap/0.9\">{''.join(urls)}</urlset>"""
    (SITE / 'sitemap.xml').write_text(xml, encoding='utf-8')


def write_css() -> None:
    CSS.mkdir(parents=True, exist_ok=True)
    (CSS / 'styles.css').write_text(
        """
:root{--bg:#f2f4f7;--card:#dfe5ec;--text:#1f2937;--muted:#5b6472;--accent:#5b5cf6;--surface:#ffffff;--stroke:#ccd4dd;--dark:#111827}
*{box-sizing:border-box}html,body{margin:0;padding:0}body{font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;background:var(--bg);color:var(--text)}
a{color:inherit;text-decoration:none}img{max-width:100%;display:block}
.shell{max-width:1200px;margin:0 auto;padding:40px 24px 80px}.hero{padding:16px 0 24px;text-align:center}.eyebrow{letter-spacing:.18em;font-size:.78rem;font-weight:700;color:#8a93a3;margin:0 0 10px}.hero h1{font-size:clamp(2.2rem,5vw,3.6rem);margin:.1em 0}.lead{max-width:900px;margin:0 auto 12px;font-size:1.08rem;color:var(--muted);line-height:1.7}
.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:24px}.card{background:var(--card);border-radius:10px;padding:18px;box-shadow:0 1px 0 rgba(15,23,42,.05)}.card-head{display:flex;justify-content:space-between;gap:12px;align-items:center;font-size:.85rem;color:var(--muted);margin-bottom:12px}.badge{display:inline-flex;padding:4px 8px;border-radius:999px;background:var(--dark);color:#fff;font-size:.75rem;font-weight:600}.card h2{font-size:1.25rem;line-height:1.3;margin:0 0 10px}.card p{color:#364152;line-height:1.6;min-height:96px}.thumb-link img{border-radius:10px;border:1px solid rgba(15,23,42,.08);margin-top:10px}
.article-shell{max-width:980px}.article{background:var(--surface);padding:28px;border-radius:18px;box-shadow:0 10px 30px rgba(15,23,42,.06)}.back-link{display:inline-block;margin-bottom:16px;color:var(--accent);font-weight:600}.article h1{font-size:clamp(2rem,4vw,3rem);line-height:1.15;margin:.1em 0 .25em}.meta-row{display:flex;flex-wrap:wrap;gap:16px;color:var(--muted);font-size:.95rem;margin-bottom:20px}.hero-chart{border-radius:16px;border:1px solid var(--stroke);margin:16px 0 20px}.interactive-chart-block{margin:18px 0 22px}.chart-header h2{margin:0 0 6px;font-size:1.2rem}.chart-header p{margin:0 0 12px;color:var(--muted);line-height:1.6}.chart-summary{margin:0 0 10px;padding:10px 14px;border-radius:12px;background:#f8fafc;border:1px solid var(--stroke);font-weight:600;color:#334155}.chart-stage{position:relative;padding:14px 14px 6px;border-radius:18px;background:#0b0f14;border:1px solid #1f2937;overflow:hidden}.chart-stage-main{height:520px}.chart-stage-mini{height:220px;margin-top:12px}.chart-stage canvas{width:100%!important;height:100%!important}
.stat-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:18px 0 24px}.stat{background:#f8fafc;border:1px solid var(--stroke);border-radius:16px;padding:16px}.stat-primary{background:#eef6ff;border-color:#bfd7ff}.stat-danger{background:#fff1f2;border-color:#fecdd3}.stat span{display:block;color:var(--muted);font-size:.9rem;margin-bottom:6px}.stat strong{font-size:1.25rem}.prose p,.note p{line-height:1.8;color:#344054}.table-wrap{margin-top:28px}.table-wrap h2,.note h2{font-size:1.3rem}table{width:100%;border-collapse:collapse;background:#fff;border:1px solid var(--stroke);border-radius:12px;overflow:hidden}th,td{text-align:left;padding:14px;border-bottom:1px solid var(--stroke)}th{background:#f8fafc;font-size:.92rem}.note{margin-top:28px;padding:18px;border-radius:16px;background:#fafbfc;border:1px solid var(--stroke)}
.site-footer{display:flex;justify-content:center;padding:0 0 34px}.x-link{display:inline-flex;align-items:center;justify-content:center;width:52px;height:52px;border-radius:999px;background:#111827;box-shadow:0 10px 24px rgba(15,23,42,.14)}.x-link svg{width:22px;height:22px;fill:#fff}
@media (max-width: 960px){.grid{grid-template-columns:1fr 1fr}.stat-grid{grid-template-columns:1fr}.chart-stage-main{height:420px}.chart-stage-mini{height:190px}}@media (max-width: 680px){.grid{grid-template-columns:1fr}.shell{padding:24px 16px 60px}.article{padding:20px}.card p{min-height:auto}.chart-stage-main{height:320px}.chart-stage-mini{height:170px}}
        """.strip(),
        encoding='utf-8',
    )


def ensure_social_card(site_cfg: Dict[str, Any]) -> None:
    target = IMG / 'social-card.png'
    fig = plt.figure(figsize=(12, 6.3), dpi=120)
    fig.patch.set_facecolor('#0b0f14')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.text(0.05, 0.74, site_cfg['site_name'], color='white', fontsize=28, fontweight='bold', va='top')
    ax.text(0.05, 0.54, 'Daily blackout simulation for the world and top 10 economies', color='#98d08b', fontsize=18, va='top')
    ax.text(0.05, 0.35, 'Uses Iran blackout duration as the clock for same-length domestic outage scenarios.', color='#94a3b8', fontsize=16)
    fig.savefig(target, facecolor=fig.get_facecolor())
    plt.close(fig)


def clear_generated_output() -> None:
    for json_file in CONTENT.glob('*.json'):
        json_file.unlink(missing_ok=True)
    if (SITE / 'posts').exists():
        shutil.rmtree(SITE / 'posts')
    for image_file in IMG.glob('*-world.png'):
        image_file.unlink(missing_ok=True)


def build_one(for_date: date, site_cfg: Dict[str, Any], model_cfg: Dict[str, Any], signal_source: Signal) -> Dict[str, Any]:
    threshold = float(model_cfg.get('full_shutdown_threshold_pct', 5.0))
    signal = align_signal_to_date(signal_source, for_date, threshold)
    simulation = compute_blackout_simulation(signal, read_gdp(), read_weights(), model_cfg)
    post = build_post(for_date, signal, simulation, site_cfg, model_cfg)
    make_chart(simulation, IMG / f"{for_date.isoformat()}-world.png", title_date_end=for_date.isoformat())
    save_post(post)
    return post


def main() -> None:
    SITE.mkdir(parents=True, exist_ok=True)
    IMG.mkdir(parents=True, exist_ok=True)
    CONTENT.mkdir(parents=True, exist_ok=True)

    site_cfg = read_site_config()
    model_cfg = read_model_config()
    write_css()
    ensure_social_card(site_cfg)

    signal_source = get_signal()
    threshold = float(model_cfg.get('full_shutdown_threshold_pct', 5.0))
    outage_start = first_outage_date(signal_source, threshold) or datetime.now(timezone.utc).date()
    today = datetime.now(timezone.utc).date()

    clear_generated_output()
    for current_date in date_range(outage_start, today):
        build_one(current_date, site_cfg, model_cfg, signal_source)

    posts = load_existing_posts()
    render_index(posts, site_cfg)
    for post in posts:
        render_post(post, site_cfg)
    render_feed(posts, site_cfg)
    render_sitemap(posts, site_cfg)


if __name__ == '__main__':
    main()

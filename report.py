import os
import json
import webbrowser
from pathlib import Path
from jinja2 import Environment, FileSystemLoader


def _reports_dir() -> Path:
    return Path(__file__).parent / 'reports'


def _prepare_results(results: dict) -> dict:
    """Convert by_strike DataFrames to JSON-serializable lists for Chart.js."""
    prepared = {}
    for ticker, data in results.items():
        if data.get('error'):
            prepared[ticker] = data
            continue
        d = dict(data)
        for key in ('all', 'multi', '0dte'):
            if d.get(key) is not None:
                lvl = dict(d[key])
                bs = lvl.pop('by_strike', None)
                if bs is not None and not bs.empty:
                    lvl['chart_strikes'] = [round(v, 2) for v in bs['strike'].tolist()]
                    lvl['chart_gex']     = [round(v, 4) for v in bs['gex'].tolist()]
                    lvl['chart_vex']     = [round(v, 4) for v in bs['vex'].tolist()]
                    lvl['chart_chex']    = [round(v, 4) for v in bs['chex'].tolist()]
                d[key] = lvl
        prepared[ticker] = d
    return prepared


def build_report(results: dict, date_str: str) -> str:
    prepared = _prepare_results(results)
    template_dir = Path(__file__).parent / 'templates'
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    env.filters['tojson'] = lambda v: json.dumps(v)
    template = env.get_template('report.html')
    return template.render(results=prepared, date_str=date_str)


def save_report(html: str, date_str: str) -> str:
    out_dir = _reports_dir()
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{date_str}.html"
    path.write_text(html, encoding='utf-8')
    return str(path.resolve())


def open_report(path: str) -> None:
    url = 'file:///' + path.replace(os.sep, '/')
    webbrowser.open(url)

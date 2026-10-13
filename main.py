"""Download, train and evaluate an educational pre-shot xG model."""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import pandas as pd
import requests
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
NUM = ['distance', 'angle']
CAT = ['body_part', 'shot_type']
BIN = ['under_pressure', 'first_time']
SEED = 20261013


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def fetch(url):
    if 'raw.githubusercontent.com/hudl/open-data/' in url:
        tail = url.split('raw.githubusercontent.com/hudl/open-data/')[1]
        ref, resource = tail.split('/', 1)
        url = f'https://cdn.jsdelivr.net/gh/statsbomb/open-data@{ref}/{resource}'
    for attempt in range(4):
        try:
            host = ['cdn.jsdelivr.net', 'fastly.jsdelivr.net', 'gcore.jsdelivr.net', 'testingcf.jsdelivr.net'][attempt]
            target = url.replace('cdn.jsdelivr.net', host)
            response = requests.get(target, timeout=(15, 30))
            response.raise_for_status()
            return response.content
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def get_file(url, path):
    if path.exists():
        return path.read_bytes()
    # GitHub's content API is reliable for small specification/license files.
    if '/data/' not in url and 'raw.githubusercontent.com/hudl/open-data/' in url:
        tail = url.split('raw.githubusercontent.com/hudl/open-data/')[1]
        ref, resource = tail.split('/', 1)
        response = requests.get(f'https://api.github.com/repos/hudl/open-data/contents/{resource}',
                                params={'ref': ref}, timeout=30)
        response.raise_for_status()
        content = base64.b64decode(response.json()['content'])
    else:
        content = fetch(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return content


def geometry(x, y):
    dx = 120 - np.asarray(x)
    yy = np.asarray(y)
    distance = np.hypot(dx, yy - 40)
    # Angle between vectors from the shot to the two posts, in radians.
    cross = 8 * np.abs(dx)
    dot = dx * dx + (36 - yy) * (44 - yy)
    return distance, np.arctan2(cross, dot)


def download():
    data = ROOT / 'data'
    manifest_path = data / 'source_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        commit = manifest['commit']
    else:
        response = requests.get('https://api.github.com/repos/hudl/open-data/commits/master', timeout=60)
        response.raise_for_status()
        commit = response.json()['sha']
        manifest = {'repository': 'https://github.com/hudl/open-data', 'commit': commit,
                    'competition_id': 2, 'season_id': 27, 'retrieved_date': '2026-10-13'}
        write_json(manifest_path, manifest)
    base = f'https://raw.githubusercontent.com/hudl/open-data/{commit}'
    matches = json.loads(get_file(f'{base}/data/matches/2/27.json', data / 'matches.json'))
    for name in ['LICENSE.pdf', 'README.md']:
        get_file(f'{base}/{name}', data / name)
    get_file(f'{base}/img/SB - Icon Lockup - Colour positive.png', ROOT / 'assets/statsbomb_logo.png')
    get_file(f'{base}/doc/StatsBomb Open Data Specification v1.1.pdf', data / 'event_specification.pdf')
    # Retain full source events in gzip files; only shot-time fields enter X.
    import gzip
    def one(match):
        mid = match['match_id']
        path = data / 'raw_events' / f'{mid}.json.gz'
        if path.exists():
            raw = gzip.decompress(path.read_bytes())
        else:
            raw = fetch(f'{base}/data/events/{mid}.json')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(gzip.compress(raw, mtime=0))
        events = json.loads(raw)
        rows = []
        for event in events:
            if event.get('type', {}).get('name') != 'Shot':
                continue
            shot = event['shot']
            location = event.get('location', [None, None])
            rows.append({'event_id': event['id'], 'match_id': mid, 'date': match['match_date'],
                         'kick_off': match.get('kick_off', ''), 'home': match['home_team']['home_team_name'],
                         'away': match['away_team']['away_team_name'], 'team': event['team']['name'],
                         'player': event.get('player', {}).get('name', ''), 'minute': event['minute'],
                         'period': event['period'], 'x': location[0], 'y': location[1],
                         'body_part': shot.get('body_part', {}).get('name', 'Unknown'),
                         'shot_type': shot.get('type', {}).get('name', 'Unknown'),
                         'under_pressure': int(event.get('under_pressure', False)),
                         'first_time': int(shot.get('first_time', False)),
                         'goal': int(shot['outcome']['name'] == 'Goal')})
        return rows, {'match_id': mid, 'sha256': hashlib.sha256(raw).hexdigest(), 'shots': len(rows)}
    all_rows, hashes = [], []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(one, match) for match in matches]
        for i, future in enumerate(as_completed(futures), 1):
            rows, digest = future.result()
            all_rows.extend(rows)
            hashes.append(digest)
            if i % 40 == 0 or i == len(matches):
                print(f'Downloaded {i}/{len(matches)} matches', flush=True)
    df = pd.DataFrame(all_rows).sort_values(['date', 'kick_off', 'match_id', 'period', 'minute', 'event_id'])
    assert df.event_id.is_unique
    manifest['files'] = sorted(hashes, key=lambda x: x['match_id'])
    manifest['matches'] = len(matches)
    manifest['total_shots'] = len(df)
    write_json(manifest_path, manifest)
    df.to_csv(data / 'shots.csv', index=False, encoding='utf-8-sig')
    print(f'Extracted {len(df)} shots', flush=True)


def load_data():
    df = pd.read_csv(ROOT / 'data/shots.csv')
    valid = df.x.between(0, 120) & df.y.between(0, 80) & df.period.between(1, 4)
    df = df.loc[valid].copy()
    df['distance'], df['angle'] = geometry(df.x, df.y)
    # Keep an entire calendar date in one split; no same-day cross-split leakage.
    dates = sorted(df.date.unique())
    cut1, cut2 = dates[int(.60 * len(dates))], dates[int(.80 * len(dates))]
    df['split'] = np.where(df.date < cut1, 'train', np.where(df.date < cut2, 'validation', 'test'))
    return df.reset_index(drop=True), {'validation_start': cut1, 'test_start': cut2,
                                     'invalid_excluded': int((~valid).sum())}


def make_model(C=1., context=True):
    transformers = [('numeric', StandardScaler(), NUM)]
    if context:
        transformers += [('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT),
                         ('binary', 'passthrough', BIN)]
    return Pipeline([('features', ColumnTransformer(transformers, remainder='drop')),
                     ('model', LogisticRegression(C=C, max_iter=3000, random_state=SEED))])


def evaluate(y, p):
    return {'log_loss': float(log_loss(y, p)), 'brier': float(brier_score_loss(y, p)),
            'auc': float(roc_auc_score(y, p)), 'mean_xg': float(np.mean(p)),
            'goal_rate': float(np.mean(y)), 'sum_xg': float(np.sum(p)), 'goals': int(np.sum(y))}


def reliability(y, p, bins=8):
    order = np.argsort(p, kind='stable')
    rows = []
    for indices in np.array_split(order, bins):
        if not len(indices):
            continue
        n, k = len(indices), int(np.asarray(y)[indices].sum())
        rate, z = k/n, 1.96
        center = (rate + z*z/(2*n))/(1+z*z/n)
        half = z*math.sqrt(rate*(1-rate)/n + z*z/(4*n*n))/(1+z*z/n)
        rows.append({'n': n, 'goals': k, 'mean_xg': float(np.asarray(p)[indices].mean()),
                     'goal_rate': rate, 'wilson_low': center-half, 'wilson_high': center+half})
    return pd.DataFrame(rows)


def fit():
    start = time.perf_counter()
    df, split_info = load_data()
    nonpen = df[df.shot_type != 'Penalty'].copy()
    train, val, test = [nonpen[nonpen.split == s] for s in ['train', 'validation', 'test']]
    candidates, fitted = [], {}
    for context in [False, True]:
        for C in [.01, .1, 1., 10.]:
            model = make_model(C, context).fit(train, train.goal)
            p = model.predict_proba(val)[:, 1]
            name = 'geometry' if not context else 'context'
            candidates.append({'model': name, 'C': C, 'validation_log_loss': float(log_loss(val.goal, p))})
            fitted[(name, C)] = model
    selected = {}
    for name in ['geometry', 'context']:
        best = min((r for r in candidates if r['model'] == name), key=lambda r: r['validation_log_loss'])
        selected[name] = best
    # Fix the model using validation only. Test is used for the final audit once.
    main = fitted[('context', selected['context']['C'])]
    geo = fitted[('geometry', selected['geometry']['C'])]
    val_logits = main.decision_function(val).reshape(-1, 1)
    calibrator = LogisticRegression(C=1e6, max_iter=1000).fit(val_logits, val.goal)
    test_logits = main.decision_function(test).reshape(-1, 1)
    p_raw = main.predict_proba(test)[:, 1]
    p_cal = calibrator.predict_proba(test_logits)[:, 1]
    baseline = float(train.goal.mean())
    metrics = {'constant': evaluate(test.goal, np.full(len(test), baseline)),
               'geometry': evaluate(test.goal, geo.predict_proba(test)[:, 1]),
               'context': evaluate(test.goal, p_raw),
               'context_calibrated': evaluate(test.goal, p_cal)}
    output = test.copy()
    output['xg'] = p_raw
    output['xg_calibrated'] = p_cal
    output['xg_geometry'] = geo.predict_proba(test)[:, 1]
    (ROOT / 'results').mkdir(exist_ok=True)
    output.to_csv(ROOT / 'results/test_predictions.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(candidates).to_csv(ROOT / 'results/validation_selection.csv', index=False)
    for name, pred in [('context', p_raw), ('calibrated', p_cal)]:
        reliability(test.goal, pred).to_csv(ROOT / f'results/calibration_{name}.csv', index=False)
    # Resample whole matches to preserve within-match dependence.
    rng = np.random.default_rng(SEED)
    mids = output.match_id.unique()
    groups = {mid: np.flatnonzero(output.match_id.to_numpy() == mid) for mid in mids}
    boot = []
    for _ in range(1000):
        indices = np.concatenate([groups[mid] for mid in rng.choice(mids, size=len(mids), replace=True)])
        y, p = output.goal.to_numpy()[indices], p_raw[indices]
        if len(np.unique(y)) < 2:
            continue
        item = evaluate(y, p)
        item['brier_gain_vs_constant'] = float(brier_score_loss(y, np.full(len(y), baseline))-item['brier'])
        boot.append(item)
    cis = {k: np.quantile([r[k] for r in boot], [.025, .975]).tolist()
           for k in ['log_loss', 'brier', 'auc', 'brier_gain_vs_constant']}
    penalties = df[df.shot_type == 'Penalty']
    ptr, pte = penalties[penalties.split == 'train'], penalties[penalties.split == 'test']
    penalty_p = (ptr.goal.sum()+1)/(len(ptr)+2)
    pen_result = {'train_n': len(ptr), 'train_goals': int(ptr.goal.sum()), 'test_n': len(pte),
                  'test_goals': int(pte.goal.sum()), 'beta_1_1_posterior_mean': float(penalty_p),
                  'test_brier': float(brier_score_loss(pte.goal, np.full(len(pte), penalty_p)))}
    params = main.named_steps['features']
    model = main.named_steps['model']
    num_scaler = params.named_transformers_['numeric']
    enc = params.named_transformers_['category']
    export = {'features': params.get_feature_names_out().tolist(), 'coef': model.coef_[0].tolist(),
              'intercept': float(model.intercept_[0]), 'numeric_mean': num_scaler.mean_.tolist(),
              'numeric_scale': num_scaler.scale_.tolist(), 'categories': [x.tolist() for x in enc.categories_],
              'calibration_slope': float(calibrator.coef_[0,0]),
              'calibration_intercept': float(calibrator.intercept_[0]), 'penalty_probability': float(penalty_p)}
    write_json(ROOT / 'models/browser_model.json', export)
    joblib.dump({'model': main, 'geometry': geo, 'calibrator': calibrator, 'penalty_probability': penalty_p},
                ROOT / 'models/xg_model.joblib')
    summary = {'seed': SEED, 'total_valid_shots': len(df), 'nonpenalty_shots': len(nonpen),
               'penalty_shots': len(penalties), 'split': split_info,
               'split_summary': {s: {'matches': int(df[df.split==s].match_id.nunique()),
                                    'nonpenalty_shots': int(len(nonpen[nonpen.split==s])),
                                    'nonpenalty_goals': int(nonpen[nonpen.split==s].goal.sum()),
                                    'first_date': df[df.split==s].date.min(), 'last_date': df[df.split==s].date.max()}
                                 for s in ['train','validation','test']},
               'selected': selected, 'metrics': metrics, 'match_bootstrap_95ci': cis,
               'penalties': pen_result, 'train_goal_rate': baseline,
               'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__,
                            'numpy': np.__version__, 'pandas': pd.__version__},
               'elapsed_seconds': time.perf_counter()-start}
    write_json(ROOT / 'results/summary.json', summary)
    df.to_csv(ROOT / 'data/shots_features.csv', index=False, encoding='utf-8-sig')
    # Verify the exported browser predictor on every test sample.
    manual = np.full(len(test), export['intercept'])
    offset = 0
    for j, col in enumerate(NUM):
        manual += (test[col].to_numpy()-export['numeric_mean'][j])/export['numeric_scale'][j]*export['coef'][offset]
        offset += 1
    for col, cats in zip(CAT, export['categories']):
        for cat in cats:
            manual += (test[col].to_numpy()==cat)*export['coef'][offset]
            offset += 1
    for col in BIN:
        manual += test[col].to_numpy()*export['coef'][offset]
        offset += 1
    np.testing.assert_allclose(1/(1+np.exp(-manual)), p_raw, atol=1e-12)
    assert set(train.match_id).isdisjoint(test.match_id)
    assert set(train.match_id).isdisjoint(val.match_id)
    assert set(val.match_id).isdisjoint(test.match_id)
    assert train.date.max() < val.date.min() <= val.date.max() < test.date.min()
    assert set(NUM+CAT+BIN).isdisjoint({'goal','statsbomb_xg','end_location','outcome'})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['download','train','all'], default='all', nargs='?')
    args = parser.parse_args()
    if args.command in ['download','all']:
        download()
    if args.command in ['train','all']:
        fit()

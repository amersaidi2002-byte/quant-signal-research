"""Reproducible synthetic time-series research, not a live trading strategy.

Run: python run_benchmark.py
All observations are generated locally. No market data or external APIs.
"""
from pathlib import Path
import json
import platform
import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
SEED = 42
N = 12000
INITIAL_TRAIN = 4000
TEST_BLOCK = 2000
GAP = 1


def simulate():
    rng = np.random.default_rng(SEED)
    phi = np.linspace(.25, .85, 8)
    z = np.zeros((N + 500, 8))
    innovations = rng.normal(size=z.shape)
    for t in range(1, len(z)):
        z[t] = phi * z[t - 1] + np.sqrt(1 - phi ** 2) * innovations[t]
    z = z[500:]
    # Observables at decision time t predict a synthetic next-period return.
    conditional_mean = .0012 * np.tanh(2 * z[:, 0]) + .001 * z[:, 1] * z[:, 2] + .0005 * z[:, 3]
    sigma = np.where(np.arange(N) < 6000, .008, .012)
    y = conditional_mean + sigma * rng.normal(size=N)
    previous_return = np.r_[0., y[:-1]]
    X = pd.DataFrame(z, columns=[f"observable_{j}" for j in range(8)])
    X['previous_return'] = previous_return
    return X, y


def candidates():
    configs = []
    for alpha in [.1, 1., 10., 100.]:
        configs.append(('Ridge', {'alpha': alpha}, make_pipeline(StandardScaler(), Ridge(alpha=alpha))))
    for depth in [3, 6]:
        for leaf in [25, 75]:
            params = dict(n_estimators=100, max_depth=depth, min_samples_leaf=leaf, random_state=SEED, n_jobs=1)
            configs.append(('RandomForest', params, RandomForestRegressor(**params)))
    return configs


def make_model(name, params):
    if name == 'Ridge':
        return make_pipeline(StandardScaler(), Ridge(**params))
    if name == 'RandomForest':
        return RandomForestRegressor(**params)
    return DummyRegressor(strategy='constant', constant=0.)


def circular_block_interval(signal, target, block=20, draws=1000):
    rng = np.random.default_rng(SEED + 1)
    n = len(target)
    results = []
    for _ in range(draws):
        starts = rng.integers(0, n, size=int(np.ceil(n / block)))
        indices = ((starts[:, None] + np.arange(block)) % n).ravel()[:n]
        results.append(float(np.corrcoef(signal[indices], target[indices])[0, 1]))
    return [float(x) for x in np.quantile(results, [.025, .975])]


def main():
    X, y = simulate()
    # Tune once on the initial development period; outer tests never select parameters.
    tuning_rows = []
    selected = {}
    for name, params, estimator in candidates():
        estimator.fit(X.iloc[:2999], y[:2999])
        prediction = estimator.predict(X.iloc[3000:4000])
        mse = float(np.mean((y[3000:4000] - prediction) ** 2))
        tuning_rows.append(dict(model=name, parameters=json.dumps(params, sort_keys=True), validation_mse=mse))
        if name not in selected or mse < selected[name]['validation_mse']:
            selected[name] = dict(parameters=params, validation_mse=mse)
    selected_model = min(selected, key=lambda k: selected[k]['validation_mse'])
    print('Frozen configuration selection:', selected_model, flush=True)
    predictions = []
    folds = []
    names = ['ZeroForecast', 'Ridge', 'RandomForest']
    for fold, start in enumerate(range(INITIAL_TRAIN, N, TEST_BLOCK), 1):
        stop = min(start + TEST_BLOCK, N)
        train_stop = start - GAP
        assert train_stop - 1 < start - 1
        fold_info = dict(fold=fold, train_first=0, train_last=train_stop-1,
                         test_first=start, test_last=stop-1, gap=GAP)
        for name in names:
            params = selected.get(name, {}).get('parameters', {})
            estimator = make_model(name, params)
            estimator.fit(X.iloc[:train_stop], y[:train_stop])
            pred = estimator.predict(X.iloc[start:stop])
            # Size positions using past returns only. All tests use the same rule.
            position = np.tanh(pred / np.std(y[:train_stop], ddof=1))
            predictions.append(pd.DataFrame(dict(index=np.arange(start, stop), fold=fold,
                                                 model=name, target=y[start:stop], prediction=pred,
                                                 position=position)))
        folds.append(fold_info)
        print('Completed outer fold', fold, flush=True)
    long = pd.concat(predictions, ignore_index=True)
    summary, by_fold, costs = [], [], []
    for name, part in long.groupby('model', sort=False):
        part = part.sort_values('index')
        truth, pred = part.target.to_numpy(), part.prediction.to_numpy()
        ic = float(np.corrcoef(pred, truth)[0, 1]) if np.std(pred) else None
        rank_ic = float(spearmanr(pred, truth).statistic) if np.std(pred) else None
        interval = circular_block_interval(pred, truth) if np.std(pred) else None
        summary.append(dict(model=name, observations=len(part), mse=float(np.mean((truth-pred)**2)),
                            mae=float(np.mean(np.abs(truth-pred))), pearson_ic=ic,
                            spearman_ic=rank_ic, ic_95pct_block_bootstrap=interval))
        for fold, data in part.groupby('fold'):
            fold_ic = float(np.corrcoef(data.prediction, data.target)[0, 1]) if np.std(data.prediction) else None
            by_fold.append(dict(model=name, fold=int(fold), mse=float(np.mean((data.target-data.prediction)**2)), pearson_ic=fold_ic))
        pos = part.position.to_numpy()
        turnover = np.abs(np.diff(pos, prepend=0.))
        gross = pos * truth
        for bps in [0, 5, 10]:
            net = gross - bps / 10000. * turnover
            volatility = np.std(net, ddof=1)
            costs.append(dict(model=name, cost_bps_per_unit_turnover=bps,
                              mean_daily_pnl_bps=float(np.mean(net)*10000),
                              annualized_sharpe=float(np.mean(net)/volatility*np.sqrt(252)) if volatility else None,
                              mean_daily_turnover=float(np.mean(turnover))))
    zero_mse = next(x['mse'] for x in summary if x['model']=='ZeroForecast')
    for result in summary:
        result['mse_reduction_vs_zero_pct'] = (1-result['mse']/zero_mse)*100
    payload = dict(study='Synthetic alpha-signal benchmark', simulated_data=True,
                   interpretation='Controlled simulation only; not measured market alpha or a forecast of live returns.',
                   seed=SEED, observations=N, initial_train=INITIAL_TRAIN, test_observations=N-INITIAL_TRAIN,
                   gap=GAP, horizon=1, tuning='Initial chronological 2999/1000 development split with one excluded row.',
                   selected_model_on_development=selected_model, selected=selected, folds=folds,
                   metrics=summary, fold_metrics=by_fold, cost_sensitivity=costs,
                   versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                                 scipy=scipy.__version__, scikit_learn=sklearn.__version__))
    assert long.groupby('model').size().eq(8000).all()
    assert long.groupby(['model', 'index']).size().eq(1).all()
    assert np.isfinite(long[['target', 'prediction', 'position']].to_numpy()).all()
    assert all(f['train_last'] <= f['test_first']-2 for f in folds)
    for result in summary:
        if result['pearson_ic'] is not None:
            assert -1 <= result['pearson_ic'] <= 1
    (ROOT/'results.json').write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    pd.DataFrame(tuning_rows).to_csv(ROOT/'development_search.csv', index=False)
    pd.DataFrame(summary).drop(columns=['ic_95pct_block_bootstrap']).to_csv(ROOT/'metrics.csv', index=False)
    pd.DataFrame(by_fold).to_csv(ROOT/'fold_metrics.csv', index=False)
    pd.DataFrame(costs).to_csv(ROOT/'cost_sensitivity.csv', index=False)
    long.to_csv(ROOT/'out_of_sample_predictions.csv', index=False)
    print(json.dumps(dict(selected_model=selected_model, metrics=summary), indent=2), flush=True)


if __name__ == '__main__':
    main()

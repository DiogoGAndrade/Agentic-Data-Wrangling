"""
Gap analysis: find what cleaning actions can push TIEs to WINs.
Run from Projeto root: python evaluation/gap_analysis.py
Results saved to evaluation/outputs/gap_analysis_results.txt
"""
import warnings; warnings.filterwarnings('ignore')
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd, numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, f1_score, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from engine.config import RANDOM_STATE

EXPORTS = ROOT / 'data' / 'exports'
OUT_PATH = ROOT / 'evaluation' / 'outputs' / 'gap_analysis_results.txt'
OUT = open(OUT_PATH, 'w')

def log(msg):
    print(msg)
    OUT.write(msg + '\n')
    OUT.flush()

class Prep(BaseEstimator, TransformerMixin):
    def __init__(self, clip_iqr=None, num_strat='median', scale=False):
        self.clip_iqr = clip_iqr
        self.num_strat = num_strat
        self.scale = scale
    def fit(self, X, y=None):
        df = pd.DataFrame(X).copy()
        cat = [c for c in df.columns if df[c].dtype == object]
        self._cat = cat
        self._enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        if cat:
            self._enc.fit(df[cat].fillna('__missing__').astype(str))
        num = [c for c in df.columns if c not in cat]
        self._num = num
        self._imp = SimpleImputer(strategy=self.num_strat)
        self._imp.fit(df[num])
        if self.clip_iqr or self.scale:
            df2 = df.copy()
            if cat:
                df2[cat] = self._enc.transform(df2[cat].fillna('__missing__').astype(str))
            df2[num] = self._imp.transform(df2[num])
            if self.clip_iqr:
                self._lo = {}; self._hi = {}
                for col in num:
                    q1, q3 = df2[col].quantile(0.25), df2[col].quantile(0.75)
                    iqr = q3 - q1
                    self._lo[col] = q1 - self.clip_iqr * iqr
                    self._hi[col] = q3 + self.clip_iqr * iqr
                for col in num:
                    df2[col] = df2[col].clip(self._lo[col], self._hi[col])
            if self.scale:
                self._sc = StandardScaler()
                self._sc.fit(df2)
        return self
    def transform(self, X, y=None):
        df = pd.DataFrame(X).copy()
        if self._cat:
            df[self._cat] = self._enc.transform(df[self._cat].fillna('__missing__').astype(str))
        df[self._num] = self._imp.transform(df[self._num])
        if self.clip_iqr:
            for col in self._num:
                df[col] = df[col].clip(self._lo[col], self._hi[col])
        if self.scale:
            return self._sc.transform(df)
        return df.values

# ----------------------------------------------------------------
# support2_clf
# ----------------------------------------------------------------
df_c = pd.read_csv(EXPORTS / 'support2_clf' / 'c0_raw.csv')
X_c = df_c.drop(columns=['hospdead']); y_c = df_c['hospdead'].copy()
cv_clf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
sc_clf = {'m': make_scorer(f1_score, average='macro', zero_division=0)}

log('=== support2_clf / RF  C0=0.869151  sigma=0.008499  WIN>0.877650 ===')
for clip, strat in [(None,'median'),(3.0,'median'),(2.0,'median'),(None,'mean'),(3.0,'mean'),(2.0,'mean')]:
    p = Pipeline([('prep', Prep(clip, strat)),
                  ('m', RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE))])
    s = cross_validate(p, X_c, y_c, cv=cv_clf, scoring=sc_clf)
    v = s['test_m']
    log(f'  clip={clip} strat={strat}: {np.mean(v):.5f}  d={np.mean(v)-0.869151:+.5f}')

log('')
log('=== support2_clf / GBM  C0=0.872889  sigma=0.008325  WIN>0.881214 ===')
for clip, strat in [(None,'median'),(3.0,'median'),(2.0,'median'),(None,'mean')]:
    p = Pipeline([('prep', Prep(clip, strat)),
                  ('m', GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE))])
    s = cross_validate(p, X_c, y_c, cv=cv_clf, scoring=sc_clf)
    v = s['test_m']
    log(f'  clip={clip} strat={strat}: {np.mean(v):.5f}  d={np.mean(v)-0.872889:+.5f}')

log('')
log('=== support2_clf / KNN  C0=0.549873  sigma=0.010344  WIN>0.560217 ===')
for clip, strat, k in [(None,'median',5),(3.0,'median',5),(None,'mean',5),(None,'median',7),(3.0,'median',7)]:
    p = Pipeline([('prep', Prep(clip, strat)),
                  ('m', KNeighborsClassifier(n_neighbors=k))])
    s = cross_validate(p, X_c, y_c, cv=cv_clf, scoring=sc_clf)
    v = s['test_m']
    log(f'  clip={clip} strat={strat} k={k}: {np.mean(v):.5f}  d={np.mean(v)-0.549873:+.5f}')

# ----------------------------------------------------------------
# support2_reg
# ----------------------------------------------------------------
df_r = pd.read_csv(EXPORTS / 'support2_reg' / 'c0_raw.csv')
X_r = df_r.drop(columns=['log_charges']); y_r = df_r['log_charges'].copy()
cv_reg = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
sc_reg = {'m': make_scorer(r2_score)}

log('')
log('=== support2_reg / Ridge  C0=0.726001  sigma=0.013342  WIN>0.739343 ===')
for clip, strat, scale in [(None,'median',False),(None,'mean',False),(None,'mean',True),
                            (3.0,'mean',True),(2.0,'mean',True),(None,'median',True),(3.0,'median',True)]:
    p = Pipeline([('prep', Prep(clip, strat, scale)),
                  ('m', Ridge(random_state=RANDOM_STATE))])
    s = cross_validate(p, X_r, y_r, cv=cv_reg, scoring=sc_reg)
    v = s['test_m']
    log(f'  clip={clip} strat={strat} scale={scale}: {np.mean(v):.5f}  d={np.mean(v)-0.726001:+.5f}')

log('')
log('=== support2_reg / RF  C0=0.944718  sigma=0.006480  WIN>0.951198 ===')
for clip, strat in [(None,'median'),(3.0,'median'),(2.0,'median'),(None,'mean')]:
    p = Pipeline([('prep', Prep(clip, strat)),
                  ('m', RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE))])
    s = cross_validate(p, X_r, y_r, cv=cv_reg, scoring=sc_reg)
    v = s['test_m']
    log(f'  clip={clip} strat={strat}: {np.mean(v):.5f}  d={np.mean(v)-0.944718:+.5f}')

log('')
log('=== support2_reg / GBM  C0=0.940794  sigma=0.006303  WIN>0.947097 ===')
for clip, strat in [(None,'median'),(3.0,'median'),(2.0,'median'),(None,'mean')]:
    p = Pipeline([('prep', Prep(clip, strat)),
                  ('m', GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE))])
    s = cross_validate(p, X_r, y_r, cv=cv_reg, scoring=sc_reg)
    v = s['test_m']
    log(f'  clip={clip} strat={strat}: {np.mean(v):.5f}  d={np.mean(v)-0.940794:+.5f}')

# ----------------------------------------------------------------
# platform
# ----------------------------------------------------------------
df_p = pd.read_csv(EXPORTS / 'platform' / 'c0_raw.csv')
X_p = df_p.drop(columns=['purchased']); y_p = df_p['purchased'].copy()
cv_plt = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

log('')
log('=== platform / KNN  C0=0.501796  sigma=0.017294  WIN>0.519090 ===')
for clip, strat, k in [(None,'median',5),(3.0,'median',5),(3.0,'mean',5),(None,'median',7),(3.0,'median',7),(None,'mean',7)]:
    p = Pipeline([('prep', Prep(clip, strat)),
                  ('m', KNeighborsClassifier(n_neighbors=k))])
    s = cross_validate(p, X_p, y_p, cv=cv_plt, scoring=sc_clf)
    v = s['test_m']
    log(f'  clip={clip} strat={strat} k={k}: {np.mean(v):.5f}  d={np.mean(v)-0.501796:+.5f}')

log('')
log('=== platform / RF  C0=0.491630  sigma=0.010663  WIN>0.502293 ===')
for clip, strat in [(None,'median'),(3.0,'median'),(2.0,'median'),(None,'mean'),(3.0,'mean')]:
    p = Pipeline([('prep', Prep(clip, strat)),
                  ('m', RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE))])
    s = cross_validate(p, X_p, y_p, cv=cv_plt, scoring=sc_clf)
    v = s['test_m']
    log(f'  clip={clip} strat={strat}: {np.mean(v):.5f}  d={np.mean(v)-0.491630:+.5f}')

log('')
log('=== platform / GBM  C0=0.408978  sigma=0.003459  WIN>0.412437 ===')
for clip, strat in [(None,'median'),(3.0,'median'),(2.0,'median'),(None,'mean'),(3.0,'mean'),(1.5,'median')]:
    p = Pipeline([('prep', Prep(clip, strat)),
                  ('m', GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE))])
    s = cross_validate(p, X_p, y_p, cv=cv_plt, scoring=sc_clf)
    v = s['test_m']
    log(f'  clip={clip} strat={strat}: {np.mean(v):.5f}  d={np.mean(v)-0.408978:+.5f}')

log('\nDONE')
OUT.close()
print(f'Results saved to {OUT_PATH}')

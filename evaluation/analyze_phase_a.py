import pandas as pd
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 25)
df = pd.read_csv('evaluation/outputs/MASTER_RESULTS_TABLE.csv')
print(f'Total rows: {len(df)}\n')

# ===== CLASSIFICATION =====
clf = df[df['task_type'] == 'classification']
print('=== CLASSIFICATION — F1-macro per dataset (mean across algos) ===')
piv = clf.pivot_table(index='condition', columns='dataset', values='f1_macro', aggfunc='mean').round(4)
print(piv); print()
print('CLASSIFICATION — overall mean F1-macro:')
print(clf.groupby('condition')['f1_macro'].mean().round(4).sort_values(ascending=False)); print()

# ===== REGRESSION =====
reg = df[df['task_type'] == 'regression']
print('=== REGRESSION — R² per dataset (mean across algos) ===')
piv_r = reg.pivot_table(index='condition', columns='dataset', values='r2', aggfunc='mean').round(4)
print(piv_r); print()
print('REGRESSION — MAE per dataset (mean across algos):')
piv_m = reg.pivot_table(index='condition', columns='dataset', values='mae', aggfunc='mean').round(2)
print(piv_m); print()

# ===== DELTAS vs C1 =====
print('=== DELTA C2_<llm> - C1_manual (positive = LLM beats baseline) ===')
print('Classification F1-macro per dataset:')
c1c = clf[clf['condition']=='C1_manual'].groupby('dataset')['f1_macro'].mean()
for cond in sorted(c for c in clf['condition'].unique() if c.startswith('C2_')):
    sub = clf[clf['condition']==cond].groupby('dataset')['f1_macro'].mean()
    deltas = [(ds, sub[ds] - c1c[ds]) for ds in c1c.index if ds in sub.index]
    avg = sum(d for _, d in deltas) / len(deltas) if deltas else 0
    detail = ' | '.join(f'{ds}={d:+.4f}' for ds, d in deltas)
    print(f'  {cond:25s}  avg={avg:+.4f}   {detail}')

print()
print('Regression R² (life_expectancy):')
c1r = reg[reg['condition']=='C1_manual']['r2'].mean()
for cond in sorted(c for c in reg['condition'].unique() if c.startswith('C2_')):
    c2 = reg[reg['condition']==cond]['r2'].mean()
    print(f'  {cond:25s}  R²={c2:.4f}  delta={c2 - c1r:+.4f}')

# ===== PER ALGORITHM RANKING =====
print()
print('=== Best LLM per dataset/algorithm (winners only, F1 or R²) ===')
for ds in sorted(df['dataset'].unique()):
    sub = df[df['dataset']==ds]
    metric = 'r2' if sub['task_type'].iloc[0]=='regression' else 'f1_macro'
    print(f'\n  {ds} ({metric}):')
    for alg in sorted(sub['model'].unique()):
        sa = sub[sub['model']==alg]
        best = sa.sort_values(metric, ascending=False).head(1)
        c1 = sa[sa['condition']=='C1_manual'][metric].iloc[0] if (sa['condition']=='C1_manual').any() else float('nan')
        b = best.iloc[0]
        delta = b[metric] - c1
        print(f'    {alg:7s}  best={b["condition"]:22s} {metric}={b[metric]:.4f}   (C1={c1:.4f}, Δ={delta:+.4f})')

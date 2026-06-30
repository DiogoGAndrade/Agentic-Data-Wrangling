import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df_raw = pd.read_csv('/sessions/gifted-focused-cannon/mnt/Projeto/Experience with Data Preparation Tasks and Tools_May 25, 2026_21.33.csv', skiprows=[1,2])
df = df_raw[df_raw['Status'] == 'IP Address'].copy().reset_index(drop=True)
N = len(df)

OUT = "/sessions/gifted-focused-cannon/mnt/Projeto/evaluation/outputs/figures"

plt.rcParams.update({
    "font.family":"DejaVu Sans","font.size":9,
    "axes.spines.top":False,"axes.spines.right":False,
})

C_BLUE="#1565C0"; C_GREY="#9e9e9e"; C_GREEN="#2e7d32"; C_ORG="#e65100"

def hbar(ax, series, title, color=C_BLUE):
    vc = series.dropna().value_counts()
    labels = [str(l)[:50] for l in vc.index]
    vals = vc.values
    bars = ax.barh(range(len(vals)), vals, color=color, alpha=0.85)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Responses")
    ax.set_title(title, fontsize=9, pad=6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width()+0.05, bar.get_y()+bar.get_height()/2,
                f" {v} ({v/N*100:.0f}%)", va='center', fontsize=7.5)
    ax.set_xlim(0, max(vals)*1.4)
    ax.grid(axis='x', alpha=0.25, linestyle='--')

# ── Figure Q1: Profile ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
hbar(axes[0], df['QID2'], "Q2. Current situation", C_BLUE)
hbar(axes[1], df['QID3'], "Q3. Primary field", C_GREEN)
hbar(axes[2], df['QID4'], "Q4. Data involvement level", C_ORG)
fig.suptitle("Figure Q1 — Participant Profile (n=25)", fontsize=10, y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT}/figQ1_participant_profile.png", bbox_inches='tight', dpi=200)
plt.close(fig)
print("[OK] Figure Q1")

# ── Figure Q2: Experience & Tools ───────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

fam = df['1._1'].dropna()
axes[0].hist(fam, bins=range(1,12), color=C_BLUE, alpha=0.85, edgecolor='white', align='left')
axes[0].set_xlabel("Score (1=Not familiar, 10=Very familiar)")
axes[0].set_ylabel("Count")
axes[0].set_title("Q1. Familiarity with data preparation")
axes[0].axvline(fam.mean(), color=C_ORG, lw=1.5, ls='--', label="Mean="+str(round(fam.mean(),1)))
axes[0].legend(fontsize=8)
axes[0].grid(axis='y', alpha=0.25, linestyle='--')

order_freq = ['1-5 times','6-19 times','20-40 times','More than 40 times']
vc_freq = df['2.'].value_counts().reindex(order_freq).dropna()
axes[1].bar(range(len(vc_freq)), vc_freq.values, color=C_BLUE, alpha=0.85)
axes[1].set_xticks(range(len(vc_freq)))
axes[1].set_xticklabels([l.replace(' times','') for l in vc_freq.index], rotation=20, ha='right', fontsize=8)
for i,v in enumerate(vc_freq.values):
    axes[1].text(i, v+0.1, str(v)+" ("+str(round(v/N*100))+"%" + ")", ha='center', fontsize=7.5)
axes[1].set_ylabel("Count"); axes[1].set_title("Q2. # data prep tasks performed")
axes[1].grid(axis='y', alpha=0.25, linestyle='--')

order_ai = ['0%','1-25% of the time','26-50% of the time','51-75% of the time','76-100% of the time']
vc_ai = df['6.'].value_counts().reindex(order_ai).dropna()
axes[2].bar(range(len(vc_ai)), vc_ai.values, color=C_GREEN, alpha=0.85)
axes[2].set_xticks(range(len(vc_ai)))
axes[2].set_xticklabels([l.replace(' of the time','') for l in vc_ai.index], rotation=25, ha='right', fontsize=8)
for i,v in enumerate(vc_ai.values):
    axes[2].text(i, v+0.1, str(v), ha='center', fontsize=7.5)
axes[2].set_ylabel("Count"); axes[2].set_title("Q6. % use of AI in data prep")
axes[2].grid(axis='y', alpha=0.25, linestyle='--')

hbar(axes[3], df['7.'], "Q7. AI tool used most", C_GREEN)
fig.suptitle("Figure Q2 — Experience and AI Tool Usage (n=25)", fontsize=10, y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT}/figQ2_experience_tools.png", bbox_inches='tight', dpi=200)
plt.close(fig)
print("[OK] Figure Q2")

# ── Figure Q3: Task difficulty ───────────────────────────────────────────────
tasks = {
    '8._1': 'Filling missing values',
    '8._2': 'Detecting errors & inconsistencies',
    '8._3': 'Identifying & handling outliers',
    '8._4': 'Understanding column semantics',
    '8._5': 'Transforming/encoding data',
    '8._6': 'Feature selection for modelling',
}
diff_order = ['Very Easy','Easy','Neither easy nor diffficult','Difficult','Very Difficult']
diff_colors = ['#1b5e20','#66bb6a','#b0bec5','#e57373','#b71c1c']

fig, ax = plt.subplots(figsize=(10, 4.5))
bottoms = np.zeros(len(tasks))
for level, color in zip(diff_order, diff_colors):
    vals = []
    for col in tasks.keys():
        vc = df[col].value_counts()
        vals.append(int(vc.get(level, 0)))
    ax.barh(range(len(tasks)), vals, left=bottoms, label=level, color=color, alpha=0.9)
    for i, v in enumerate(vals):
        if v > 0:
            txt_color = 'white' if color != '#b0bec5' else '#333'
            ax.text(bottoms[i]+v/2, i, str(v), ha='center', va='center', fontsize=7.5, color=txt_color)
    bottoms += np.array(vals, dtype=float)
ax.set_yticks(range(len(tasks)))
ax.set_yticklabels(list(tasks.values()), fontsize=8.5)
ax.set_xlabel("Number of responses")
ax.set_title("Figure Q3 — Perceived Difficulty of Data Preparation Tasks (Q8, n~20)", fontsize=9)
ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
ax.grid(axis='x', alpha=0.25, linestyle='--')
fig.tight_layout()
fig.savefig(f"{OUT}/figQ3_task_difficulty.png", bbox_inches='tight', dpi=200)
plt.close(fig)
print("[OK] Figure Q3")

# ── Figure Q4: Attitudes ─────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

hbar(axes[0,0], df['9.'], "Q9. Awareness of LLM-based data prep systems", C_BLUE)

conf = df['10._1'].dropna()
axes[0,1].hist(conf, bins=[0.5,1.5,2.5,3.5,4.5,5.5], color=C_ORG, alpha=0.85, edgecolor='white')
axes[0,1].set_xlabel("Score (1=Not confident, 5=Very confident)")
axes[0,1].set_ylabel("Count")
axes[0,1].set_title("Q10. Confidence using LLM-integrated data prep (n=20)")
axes[0,1].axvline(conf.mean(), color=C_BLUE, lw=1.5, ls='--', label="Mean="+str(round(conf.mean(),1)))
axes[0,1].legend(fontsize=8)
axes[0,1].grid(axis='y', alpha=0.25, linestyle='--')

stmts = {
    '11._1': 'LLMs reduce data prep effort',
    '11._2': 'Prefer to retain control',
    '11._3': 'Trust if steps are inspectable',
    '11._4': 'Transparency & reproducibility matter',
}
agree_order = ['Strongly disagree','Disagree','Neither agree nor disagree','Agree','Strongly agree']
agree_colors = ['#b71c1c','#e57373','#b0bec5','#66bb6a','#1b5e20']
ax = axes[1,0]
bottoms = np.zeros(len(stmts))
for level, color in zip(agree_order, agree_colors):
    vals = []
    for col in stmts.keys():
        vc = df[col].value_counts()
        vals.append(int(vc.get(level, 0)))
    ax.barh(range(len(stmts)), vals, left=bottoms, label=level, color=color, alpha=0.9)
    for i, v in enumerate(vals):
        if v > 0:
            txt_color = 'white' if color != '#b0bec5' else '#333'
            ax.text(bottoms[i]+v/2, i, str(v), ha='center', va='center', fontsize=7.5, color=txt_color)
    bottoms += np.array(vals, dtype=float)
ax.set_yticks(range(len(stmts)))
ax.set_yticklabels(list(stmts.values()), fontsize=8.5)
ax.set_xlabel("Number of responses")
ax.set_title("Q11. Attitudes towards automated data preparation (n=20)")
ax.legend(loc='lower right', fontsize=7.5, framealpha=0.9)
ax.grid(axis='x', alpha=0.25, linestyle='--')

hbar(axes[1,1], df['12.'], "Q12. Ideal role of intelligent systems", C_GREEN)

fig.suptitle("Figure Q4 — User Attitudes, Awareness and Preferences (n=25)", fontsize=10, y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT}/figQ4_attitudes.png", bbox_inches='tight', dpi=200)
plt.close(fig)
print("[OK] Figure Q4")

# ── Key stats ─────────────────────────────────────────────────────────────────
print()
print("=== KEY STATS ===")
masters = (df['QID2'] == "Master's student").sum()
print("Masters students:", masters)
print("Working professionally:", (df['QID2'] == "Working professionally").sum())
print("Data Science:", (df['QID3'] == "Data Science / Analytics").sum())
fam2 = df['1._1'].dropna()
print("Familiarity mean:", round(fam2.mean(),1), "SD:", round(fam2.std(),1), "n:", len(fam2))
print("More than 40 tasks:", (df['2.'] == 'More than 40 times').sum())
print("ChatGPT:", (df['7.'] == 'ChatGPT').sum())
print("Q9 aware+used:", (df['9.'] == 'Yes, and I have used such systems').sum())
print("Q9 aware+not used:", (df['9.'] == 'Yes, but I have never used such systems').sum())
print("Q9 not aware:", (df['9.'] == 'No, I was not aware of this possibility').sum())
conf2 = df['10._1'].dropna()
print("Q10 confidence mean:", round(conf2.mean(),1), "SD:", round(conf2.std(),1))
q11_1_pos = df['11._1'].isin(['Agree','Strongly agree']).sum()
q11_2_pos = df['11._2'].isin(['Agree','Strongly agree']).sum()
q11_3_pos = df['11._3'].isin(['Agree','Strongly agree']).sum()
q11_4_pos = df['11._4'].isin(['Agree','Strongly agree']).sum()
print("Q11.1 Agree+: ", q11_1_pos, "/", df['11._1'].dropna().shape[0])
print("Q11.2 Agree+ (prefer control):", q11_2_pos, "/", df['11._2'].dropna().shape[0])
print("Q11.3 Agree+ (trust if inspectable):", q11_3_pos, "/", df['11._3'].dropna().shape[0])
print("Q11.4 Agree+ (transparency):", q11_4_pos, "/", df['11._4'].dropna().shape[0])
q12_ctrl = (df['12.'] == 'Automatically perform tasks, while allowing user inspection and control').sum()
print("Q12 auto+control:", q12_ctrl, "/", df['12.'].dropna().shape[0])

import matplotlib.pyplot as plt
import seaborn as sns

from common import *
from sec_2a import *

set_seed(0)
df = load_data()
df.head()

# Pairwise scatter plot
sns.set(style="ticks", color_codes=True)
sns.pairplot(df.iloc[:, 3:-1],
             plot_kws=dict(s=5, edgecolor="none", linewidth=0, alpha=0.2));

# Box plot
fig = plt.figure(figsize=(15, 7))
fig.subplots_adjust(hspace=0.4, wspace=0.4, bottom=0.25)

df_labeled = df.loc[df["label"] != 0]
for i, col in enumerate(df_labeled.columns[3:-1], 1):
    ax = fig.add_subplot(2, 4, i)
    sns.boxplot(x="label", y=col, data=df_labeled, ax=ax)
    ax.set_xticklabels(["no cloud", "cloud"])

plt.show()

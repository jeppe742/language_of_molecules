from wandb import Api
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18, 'legend.fontsize': 14})

api = Api()

runs = api.runs("jeppe742/language-of-molecules-graph", {'$and':
                                                         [
                                                             {'config.model': 'Transformer'},
                                                             {'config.num_layers': 4},
                                                             {'tags': {"$ne": "old"}}
                                                         ]
                                                         }
                )


attention_types = {0: 'Graph attention', 2: 'Graph edge attention', 1: 'Full graph attention'}
df = None
for run in tqdm(runs):
    df_temp = run.history()
    df_temp['Attention'] = run.config['edge_encoding'] if isinstance(run.config['edge_encoding'], int) else run.config['edge_encoding']['value']
    df_temp['dataset'] = run.config['dataset']
    df_temp['bond order'] = run.config['bond_order'] if isinstance(run.config['bond_order'], int) else run.config['bond_order']['value']
    df_temp['Attention'] = df_temp.Attention.map(attention_types)

    if df is None:
        df = df_temp
    else:
        df = df.append(df_temp, sort=True)

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()

axes = [ax1, ax2, ax3, ax4]
figs = [fig1, fig2, fig3, fig4]

df[(df.dataset == 'qm9') & (df['bond order'])].groupby(['_step', 'Attention']).mean()['val_perplexity_1'].unstack().plot(ax=axes[0])
df[(df.dataset == 'zinc') & (df['bond order'])].groupby(['_step', 'Attention']).mean()['val_perplexity_1'].unstack().plot(ax=axes[1])
df[(df.dataset == 'qm9') & (df['bond order'] == False)].groupby(['_step', 'Attention']).mean()['val_perplexity_1'].unstack().plot(ax=axes[2])
df[(df.dataset == 'zinc') & (df['bond order'] == False)].groupby(['_step', 'Attention']).mean()['val_perplexity_1'].unstack().plot(ax=axes[3])

axes[0].set_title('Bond transformer on QM9')
axes[1].set_title('Bond transformer on ZINC')
axes[2].set_title('Binary transformer on QM9')
axes[3].set_title('Binary transformer on ZINC')

for i, (fig, ax) in enumerate(zip(figs, axes)):
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation perplexity')
    l = ax.get_legend()
    l.set_title('')
    fig.savefig(f"misc/graph_attention{i+1}.pdf", bbox_inches="tight")

plt.show()

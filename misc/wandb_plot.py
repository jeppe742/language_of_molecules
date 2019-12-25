from wandb import Api
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

api = Api()
runs = api.runs("jeppe742/language-of-molecules-graph", {'config.model': 'Transformer'})


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

fig, axes = plt.subplots(2, 2)


df[(df.dataset == 'qm9') & (df['bond order'])].groupby(['_step', 'Attention']).mean()['val_perplexity_1'].unstack().plot(ax=axes[0, 0])
df[(df.dataset == 'zinc') & (df['bond order'])].groupby(['_step', 'Attention']).mean()['val_perplexity_1'].unstack().plot(ax=axes[0, 1])
df[(df.dataset == 'qm9') & (df['bond order'] == False)].groupby(['_step', 'Attention']).mean()['val_perplexity_1'].unstack().plot(ax=axes[1, 0])
df[(df.dataset == 'zinc') & (df['bond order'] == False)].groupby(['_step', 'Attention']).mean()['val_perplexity_1'].unstack().plot(ax=axes[1, 1])

axes[0, 0].set_title('Bond transformer on QM9')
axes[0, 1].set_title('Bond transformer on ZINC')
axes[1, 0].set_title('Binary transformer on QM9')
axes[1, 1].set_title('Binary transformer on ZINC')

for ax in axes.reshape(-1):
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation perplexity')
    l = ax.get_legend()
    l.set_title('')

plt.show()

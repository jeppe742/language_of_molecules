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
                                                             {'config.edge_encoding': 1},
                                                             {'config.dataset': 'qm9'},
                                                             {'tags': {"$ne": "old"}}
                                                         ]
                                                         }
                )


df = None
for run in tqdm(runs):
    df_temp = run.history()
    #df_temp['Attention'] = run.config['edge_encoding'] if isinstance(run.config['edge_encoding'], int) else run.config['edge_encoding']['value']
    df_temp['dataset'] = run.config['dataset']
    df_temp['epsilon_greedy'] = run.config['epsilon_greedy'] if isinstance(run.config['epsilon_greedy'], float) else run.config['epsilon_greedy']['value']
    df_temp['bond order'] = run.config['bond_order'] if isinstance(run.config['bond_order'], int) else run.config['bond_order']['value']
    df_temp['min_idx'] = df_temp.val_perplexity_1.idxmin()
    #df_temp['Attention'] = df_temp.Attention.map(attention_types)

    if df is None:
        df = df_temp
    else:
        df = df.append(df_temp, sort=True)

df_2 = df[df._step == df.min_idx][['val_perplexity_1', 'val_perplexity_2', 'val_perplexity_3', 'val_perplexity_4', 'val_perplexity_5']].T
df_2.index = [1, 2, 3, 4, 5]
df_2.columns = ['Bond transformer - $\epsilon=0$', 'Binary transformer - $\epsilon=0$', 'Bond transformer - $\epsilon=0.2$', 'Binary transformer - $\epsilon=0.2$']
df_2.plot()
plt.xlabel('$n_{masks}$')
plt.xticks([1, 2, 3, 4, 5])
#plt.yticks([1, 20, 40, 60])
plt.title('QM9')
plt.ylabel('Validation perplexity')

plt.savefig(f"misc/epsilon_greedy_qm9.pdf", bbox_inches="tight")
plt.show()

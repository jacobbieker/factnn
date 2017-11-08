from operator import itemgetter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os


main_path = sys.argv[1]



folders = [os.path.join(main_path, folder) for folder in os.listdir(main_path)]


comparison_names = []
heatmap_data = []
comparison_auc = []
for folder in folders:
    architecture =  folder.split('_')[-1]
    c = architecture.count('c')
    f = architecture.count('f')
    
    name_new = str(c)+'c_'+str(f)+'f'
    comparison_names.append(name_new)
    
    
    df = pd.read_csv(os.path.join(folder, architecture+'_Hyperparameter.csv'))
    heatmap_data.append([architecture, df.sort_values(by='Auc', ascending=False).head(10)['Auc'].mean()])
    comparison_auc.append(df.sort_values(by='Auc', ascending=False).head(10)['Auc'].values.tolist())
    
    
matrix = np.zeros((6, 5))
for arch in heatmap_data:
    c = arch[0].count('c')-1
    f = arch[0].count('f')-1
    matrix[c, f] = arch[1]

font_big = 18
font_small = 14

plt.figure(figsize=(7.5,8))
plt.contourf(matrix, 100, cmap='jet')
plt.gca().invert_yaxis()
cbar = plt.colorbar()
cbar.set_ticks(np.arange(0.5, 1.0, 0.01), update_ticks=True)
cbar.ax.set_ylabel('Auc', fontsize=font_small)

plt.yticks(range(6), range(1,7), fontsize=font_small)
plt.xticks(range(5), range(1,6), fontsize=font_small)

plt.title('Architectures', fontsize=font_big)
plt.xlabel('# Fully Connected Layers', fontsize=font_small)
plt.ylabel('# Convolution Layers', fontsize=font_small)

for i in range(6):
    plt.plot([0,4], [i, i], 'k')
for i in range(5):
    plt.plot([i,i], [0, 5], 'k')

plt.tight_layout()
plt.savefig('build/Top10_Auc_Architecture_Comparison.pdf')



import seaborn as sns

df_plot = pd.DataFrame(np.array(comparison_auc).T, columns=comparison_names)
df_plot = df_plot[df_plot.max().sort_values(ascending=False).index]

df_swarm = pd.melt(df_plot)
df_swarm.columns = ['Architectures', 'Auc']
sns.set(style='darkgrid', font='DejaVu Sans')
plt.title('Comparing the Top 10 Aucs of all Architectures', fontsize=15)
sns.violinplot(x="Architectures", y="Auc", data=df_swarm, inner=None, color=".9")
sns.swarmplot(x="Architectures", y="Auc", data=df_swarm)
plt.xticks(rotation=60)
plt.grid(axis='x')
plt.tight_layout()
plt.savefig('build/Top10_Auc_Architecture_Heatmap.pdf')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from src.config import CLUSTER_FIGURES_DIR, BOXPLOT_FIGURES_DIR

def gerar_graficos_caracteristicas(df_stats, periodo, k_escolhido):
    """ Gera gráficos de barras para as características dos clusters. """
    pasta = os.path.join(CLUSTER_FIGURES_DIR, f'Caracteristicas_K{k_escolhido}')
    os.makedirs(pasta, exist_ok=True)
    
    df_plot = df_stats.copy()
    df_plot.index = [f"C{idx}" for idx in df_plot.index]
    
    # 1. Idades
    cols_idade = ['% < 1 ano', '% 1 a 5 anos', '% 60+ anos']
    if all(c in df_plot.columns for c in cols_idade):
        df_plot[cols_idade].plot(kind='bar', figsize=(10, 6), colormap='Set2', edgecolor='black')
        plt.title(f'Perfil Etario por Cluster - {periodo}')
        plt.ylabel('Percentagem (%)')
        plt.ylim(0, 100)
        plt.xticks(rotation=0)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(pasta, f'{periodo}_Idades.png'), dpi=300)
        plt.close()

    # 2. Gravidade
    cols_gravidade = ['% UTI', '% Suporte Invasivo', '% Obito']
    if all(c in df_plot.columns for c in cols_gravidade):
        df_plot[cols_gravidade].plot(kind='bar', figsize=(10, 6), colormap='Reds', edgecolor='black')
        plt.title(f'Indicadores de Gravidade - {periodo}')
        plt.ylabel('Percentagem (%)')
        plt.ylim(0, 100)
        plt.xticks(rotation=0)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(pasta, f'{periodo}_Gravidade.png'), dpi=300)
        plt.close()

def gerar_boxplot_com_siglas(df_dados, metrica, titulo, eixo_y_label, nome_arquivo_saida):
    """ Desenha boxplots com as siglas das UFs. """
    if df_dados.empty: return
    
    k_val = df_dados['K'].iloc[0] if 'K' in df_dados.columns else "X"
    pasta = os.path.join(BOXPLOT_FIGURES_DIR, f'Boxplots_UFs_K{k_val}')
    os.makedirs(pasta, exist_ok=True)
    
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")
    ordem = sorted(df_dados['Cluster'].unique())
    
    ax = sns.boxplot(data=df_dados, x='Cluster', y=metrica, order=ordem, color='lightgray', showfliers=False)
    cores = sns.color_palette('Set1', n_colors=len(ordem))
    
    for i, cid in enumerate(ordem):
        df_c = df_dados[df_dados['Cluster'] == cid].dropna(subset=[metrica])
        if df_c.empty: continue
        
        num_pontos = len(df_c)
        x_off = np.linspace(-0.28, 0.28, num_pontos) if num_pontos > 1 else [0]
        
        for (idx, row), off in zip(df_c.iterrows(), x_off):
            uf = row['SG_UF_NOT'] if 'SG_UF_NOT' in row else row.get('UF', '??')
            ax.plot(i + off, row[metrica], marker='o', color=cores[i], markersize=7, markeredgecolor='black', alpha=0.9)
            ax.text(i + off + 0.03, row[metrica], uf, fontsize=9, verticalalignment='center')
            
    plt.title(titulo, fontsize=15, fontweight='bold')
    plt.ylabel(eixo_y_label)
    plt.xticks(ticks=range(len(ordem)), labels=[f"C{c}" for c in ordem])
    plt.tight_layout()
    plt.savefig(os.path.join(pasta, nome_arquivo_saida), dpi=300)
    plt.close()

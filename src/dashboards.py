import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import gc
from src.config import DASHBOARD_FIGURES_DIR, MAP_CODE_TO_UF
from src.data_processing import try_parse_date

def plot_percentage_helper(ax, df, col, title, palette, order=None, horizontal=False):
    """ Função auxiliar para criar gráficos de barras com porcentagem no topo. """
    if col not in df.columns: 
        ax.set_title(f"{title} (Col não encontrada)")
        return
    
    counts = df[col].value_counts()
    if order is not None and len(order) > 0:
        counts = counts.reindex(order).fillna(0)
    
    total = counts.sum()
    if total == 0:
        ax.set_title(f"{title} (Sem dados)")
        return

    percs = (counts / total * 100)
    
    if horizontal:
        sns.barplot(x=counts.values, y=counts.index, ax=ax, palette=palette, hue=counts.index, legend=False)
        for i, (count, perc) in enumerate(zip(counts.values, percs.values)):
            ax.text(count, i, f' {int(count)} ({perc:.1f}%)', va='center')
    else:
        sns.barplot(x=counts.index, y=counts.values, ax=ax, palette=palette, hue=counts.index, legend=False)
        for i, (count, perc) in enumerate(zip(counts.values, percs.values)):
            ax.text(i, count, f'{int(count)}\n({perc:.1f}%)', ha='center', va='bottom')
    
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel("Nº de Casos")
    ax.set_xlabel("")

def run_simple_dashboards(filtered_path):
    print(">>> Gerando Dashboards Visuais (Fidelidade Total)...", flush=True)
    if not os.path.exists(filtered_path): return
    
    df = pd.read_csv(filtered_path, sep=';', low_memory=False)
    # Normalizar nomes de colunas para maiúsculas
    df.columns = [c.strip().upper() for c in df.columns]
    
    df['DT_SIN_PRI'] = try_parse_date(df['DT_SIN_PRI'])
    df = df.dropna(subset=['DT_SIN_PRI'])
    
    os.makedirs(DASHBOARD_FIGURES_DIR, exist_ok=True)
    
    # 1. Preparação de Categorias com Nomes Amigáveis
    df['SEXO'] = df['CS_SEXO'].replace({'M': 'Masc', 'F': 'Fem', 'I': 'Ign'})
    
    # Mapeamento de Evolução
    df['EVOL_NOME'] = df['EVOLUCAO'].astype(str).str.replace('.0', '', regex=False).replace({
        '1': 'Cura', '2': 'Óbito', '3': 'Óbito Outro', '9': 'Ignorado', 'nan': 'Ignorado'
    })
    
    # Mapeamento de UTI
    df['UTI_NOME'] = df['UTI'].astype(str).str.replace('.0', '', regex=False).replace({
        '1': 'Sim', '2': 'Não', '9': 'Ignorado', 'nan': 'Ignorado'
    })
    
    # Mapeamento de Suporte Ventilatório
    df['SUPORT_NOME'] = df['SUPORT_VEN'].astype(str).str.replace('.0', '', regex=False).replace({
        '1': 'Sim Inv', '2': 'Sim NInv', '3': 'Não', '9': 'Ignorado', 'nan': 'Ignorado'
    })
    
    # Mapeamento de Raça
    df['RACA_NOME'] = df['CS_RACA'].astype(str).str.replace('.0', '', regex=False).replace({
        '1': 'Branca', '2': 'Preta', '3': 'Amarela', '4': 'Parda', '5': 'Indígena', '9': 'Ignorado', 'nan': 'Ignorado'
    })
    
    def categorizar_idade_dash(idade):
        if pd.isna(idade): return "Ignorado"
        if idade <= 0.5: return "<=6m"
        if idade <= 1.0: return "6m-1a"
        if idade <= 5.0: return "1-5a"
        if idade <= 12.0: return "5-12a"
        if idade <= 19.0: return "13-19a"
        if idade <= 59.0: return "20-59a"
        return "60+a"
    
    df['FAIXA_ETARIA_DASH'] = df['IDADE_ANOS'].apply(categorizar_idade_dash)
    ordem_idade = ["<=6m", "6m-1a", "1-5a", "5-12a", "13-19a", "20-59a", "60+a"]

    # 2. DASHBOARD 4x2
    fig, axes = plt.subplots(4, 2, figsize=(20, 28))
    axes = axes.flatten()
    
    configs = [
        {'col': 'SG_UF_NOT', 'title': 'Top 10 UFs', 'pal': 'viridis', 'order': df['SG_UF_NOT'].value_counts().head(10).index},
        {'col': 'SEXO', 'title': 'Distribuição por Sexo', 'pal': 'Set2', 'order': ['Masc', 'Fem', 'Ign']},
        {'col': 'EVOL_NOME', 'title': 'Evolução Clínica', 'pal': 'RdYlGn', 'order': ['Cura', 'Óbito', 'Óbito Outro', 'Ignorado']},
        {'col': 'UTI_NOME', 'title': 'Uso de UTI', 'pal': 'Set3', 'order': ['Sim', 'Não', 'Ignorado']},
        {'col': 'SUPORT_NOME', 'title': 'Suporte Ventilatório', 'pal': 'Paired', 'order': ['Sim Inv', 'Sim NInv', 'Não', 'Ignorado']},
        {'col': 'FAIXA_ETARIA_DASH', 'title': 'Faixa Etária', 'pal': 'coolwarm', 'order': ordem_idade},
        {'col': 'RACA_NOME', 'title': 'Distribuição por Raça', 'pal': 'Pastel1', 'order': ['Branca', 'Preta', 'Amarela', 'Parda', 'Indígena', 'Ignorado']},
        {'col': 'YEAR', 'title': 'Casos por Ano', 'pal': 'Set1', 'order': sorted(df['YEAR'].unique())}
    ]

    for i, cfg in enumerate(configs):
        plot_percentage_helper(axes[i], df, cfg['col'], cfg['title'], cfg['pal'], order=cfg.get('order'))

    plt.tight_layout()
    plt.savefig(os.path.join(DASHBOARD_FIGURES_DIR, 'dashboard_analise_geral.png'), dpi=300)
    plt.close()
    
    # 3. SÉRIES TEMPORAIS
    df['ANO_DT'] = df['DT_SIN_PRI'].dt.year
    df['SEMANA'] = df['DT_SIN_PRI'].dt.strftime('%U').astype(int) + 1
    df.loc[df['SEMANA'] > 52, 'SEMANA'] = 52

    # SE GERAL
    plt.figure(figsize=(15, 8))
    for ano in sorted(df['ANO_DT'].unique()):
        data_ano = df[df['ANO_DT'] == ano].groupby('SEMANA').size().reindex(range(1, 53), fill_value=0)
        plt.plot(data_ano.index, data_ano.values, label=str(ano), marker='o', markersize=4, alpha=0.7)
    plt.title("Série Temporal Geral de VSR por Semana Epidemiológica (2015-2025)", fontsize=15, fontweight='bold')
    plt.xlabel("Semana Epidemiológica")
    plt.ylabel("Nº de Casos")
    plt.legend(title="Ano", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(DASHBOARD_FIGURES_DIR, 'serie_temporal_geral.png'), dpi=300)
    plt.close()

    # SE POR IDADE
    plt.figure(figsize=(15, 8))
    for faixa in ordem_idade:
        data_faixa = df[df['FAIXA_ETARIA_DASH'] == faixa].groupby('SEMANA').size().reindex(range(1, 53), fill_value=0)
        plt.plot(data_faixa.index, data_faixa.values, label=faixa, marker='s', markersize=4, alpha=0.8)
    plt.title("Série Temporal por Faixa Etária (Total Acumulado)", fontsize=15, fontweight='bold')
    plt.xlabel("Semana Epidemiológica")
    plt.ylabel("Nº de Casos")
    plt.legend(title="Faixa Etária")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(DASHBOARD_FIGURES_DIR, 'serie_temporal_idade.png'), dpi=300)
    plt.close()

    print(f"✅ Dashboards e Séries Temporais salvos em {DASHBOARD_FIGURES_DIR}")
    gc.collect()

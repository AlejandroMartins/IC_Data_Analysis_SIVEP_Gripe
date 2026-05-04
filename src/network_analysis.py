import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os
import numpy as np
from src.config import FIGURES_DIR, MAP_CODE_TO_UF
from src.data_processing import try_parse_date

def run_network_analysis(filtered_csv_path):
    """
    Cria grafos bipartidos conectando UFs a Semanas Epidemiológicas com melhorias visuais:
    - Filtro de Top 3 picos por UF.
    - Cores em vez de números (Heatmap Style).
    - Ordenação Geográfica/Temporal.
    - Remoção de rótulos centrais.
    """
    print(">>> Iniciando Análise de Grafos Bipartidos Otimizada...", flush=True)
    if not os.path.exists(filtered_csv_path):
        print(f"❌ Arquivo {filtered_csv_path} não encontrado.")
        return

    df = pd.read_csv(filtered_csv_path, sep=';', low_memory=False)
    df.columns = [c.strip().upper() for c in df.columns]
    
    df['DT_SIN_PRI'] = try_parse_date(df['DT_SIN_PRI'])
    df = df.dropna(subset=['DT_SIN_PRI'])
    
    if 'YEAR' in df.columns:
        df['ANO_ANALISE'] = df['YEAR']
    elif 'ANO' in df.columns:
        df['ANO_ANALISE'] = df['ANO']
    else:
        df['ANO_ANALISE'] = df['DT_SIN_PRI'].dt.year
        
    df['SEMANA'] = df['DT_SIN_PRI'].dt.strftime('%U').astype(int) + 1
    df.loc[df['SEMANA'] > 52, 'SEMANA'] = 52
    
    cenarios = {
        'GERAL': range(2015, 2026),
        'PRE_PANDEMIA': range(2015, 2020),
        'PANDEMIA': range(2020, 2023),
        'POS_PANDEMIA': range(2023, 2026)
    }

    output_dir = os.path.join(FIGURES_DIR, "network")
    os.makedirs(output_dir, exist_ok=True)

    for nome_c, anos in cenarios.items():
        print(f"  -> Processando cenário: {nome_c}")
        df_c = df[df['ANO_ANALISE'].isin(anos)].copy()
        
        if df_c.empty:
            print(f"    ⚠️ Nenhum dado encontrado para os anos {list(anos)} neste cenário.")
            continue

        # Agrupamento e Média
        counts = df_c.groupby(['ANO_ANALISE', 'SG_UF_NOT', 'SEMANA']).size().reset_index(name='casos')
        avg_counts = counts.groupby(['SG_UF_NOT', 'SEMANA'])['casos'].mean().reset_index()
        
        # --- MELHORIA 1: FILTRO DE TOP 3 PICOS POR UF ---
        # Seleciona apenas as 3 semanas de maior pico para cada UF
        links = avg_counts.sort_values(['SG_UF_NOT', 'casos'], ascending=[True, False])
        links = links.groupby('SG_UF_NOT').head(3)

        if links.empty:
            print(f"    ⚠️ Sem dados suficientes para gerar o grafo em {nome_c}.")
            continue

        B = nx.Graph()
        
        # Adicionando arestas e coletando pesos para o heatmap
        weights = []
        ufs_reais = set()
        semanas_reais = set()

        for _, row in links.iterrows():
            uf_str = str(row['SG_UF_NOT']).replace('.0', '')
            uf_nome = MAP_CODE_TO_UF.get(uf_str, uf_str)
            semana = int(row['SEMANA'])
            peso = round(row['casos'], 1)
            
            B.add_edge(uf_nome, semana, weight=peso)
            ufs_reais.add(uf_nome)
            semanas_reais.add(semana)
            weights.append(peso)
        
        # --- MELHORIA 3: ORDENAÇÃO ---
        ufs_plot = sorted(list(ufs_reais))
        semanas_plot = sorted(list(semanas_reais))

        # Configuração do Plot
        fig, ax = plt.subplots(figsize=(18, 12))
        
        # Posições (Bipartido)
        pos = {}
        # UFs à esquerda (x=0)
        pos.update((node, (0, i)) for i, node in enumerate(ufs_plot))
        # Semanas à direita (x=1)
        # Ajuste de escala vertical para as semanas para alinhar melhor
        v_scale = len(ufs_plot) / len(semanas_plot) if len(semanas_plot) > 0 else 1
        pos.update((node, (1, i * v_scale)) for i, node in enumerate(semanas_plot))

        # --- MELHORIA 2: CORES EM VEZ DE NÚMEROS (HEATMAP) ---
        edges = B.edges(data=True)
        edge_weights = [d['weight'] for u, v, d in edges]
        
        # Normalização para o colormap
        norm = mcolors.Normalize(vmin=min(edge_weights), vmax=max(edge_weights))
        cmap = cm.YlOrRd  # Amarelo para Vermelho (Estilo Heatmap)
        
        # Desenhar Nós
        nx.draw_networkx_nodes(B, pos, nodelist=ufs_plot, node_color='lightsteelblue', 
                               node_size=1500, alpha=0.9, label='UFs', ax=ax)
        nx.draw_networkx_nodes(B, pos, nodelist=semanas_plot, node_color='navajowhite', 
                               node_size=1000, alpha=0.9, label='Semanas', ax=ax)
        
        # Desenhar Arestas com Cores e Espessura Variável
        # --- MELHORIA 4: REMOÇÃO DE RÓTULOS CENTRAIS ---
        # (Simplesmente não chamamos nx.draw_networkx_edge_labels)
        edge_colors = [cmap(norm(w)) for w in edge_weights]
        # Espessura proporcional ao peso (ajustada para visibilidade)
        edge_widths = [1 + 4 * (norm(w)) for w in edge_weights] 
        
        nx.draw_networkx_edges(B, pos, edgelist=edges, width=edge_widths, 
                               edge_color=edge_colors, alpha=0.7, ax=ax)
        
        # Labels dos nós
        nx.draw_networkx_labels(B, pos, font_size=10, font_weight='bold', ax=ax)

        # Adicionar Barra de Cores (Colorbar)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label('Média de Casos (Intensidade do Pico)', fontsize=12, fontweight='bold')

        plt.title(f"Grafo Bipartido VSR: Top 3 Picos por UF vs Semanas ({nome_c})\n(Cores e Espessura indicam volume de casos)", 
                  fontsize=16, fontweight='bold', pad=20)
        
        # Legenda customizada
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Unidades Federativas (UF)',
                   markerfacecolor='lightsteelblue', markersize=15),
            Line2D([0], [0], marker='o', color='w', label='Semanas Epidemiológicas',
                   markerfacecolor='navajowhite', markersize=12)
        ]
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)

        plt.axis('off')
        plt.tight_layout()
        
        file_name = f"grafo_bipartido_clean_{nome_c.lower()}.png"
        plt.savefig(os.path.join(output_dir, file_name), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✅ Grafo limpo salvo: {file_name}")

    print(f"✅ Análise de grafos concluída com sucesso. Resultados em: {output_dir}")
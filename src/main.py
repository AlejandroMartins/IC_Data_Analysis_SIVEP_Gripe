import argparse
import os
import shutil
import sys
import pandas as pd
import gc

# Adiciona o diretório atual ao sys.path
sys.path.append(os.getcwd())

def limpar_cache_python():
    print(">>> Limpando cache do Python (__pycache__)...", flush=True)
    for root, dirs, files in os.walk(".", topdown=False):
        for name in dirs:
            if name == "__pycache__":
                try:
                    shutil.rmtree(os.path.join(root, name))
                except:
                    pass

from src.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, CLUSTER_RESULTS_DIR, METRICS_RESULTS_DIR,
    CLUSTER_FIGURES_DIR, BOXPLOT_FIGURES_DIR, DASHBOARD_FIGURES_DIR, 
    FILTERED_VSR_DATA, DADOS_CLUSTERING_PRE, DADOS_CLUSTERING_PAN, 
    DADOS_CLUSTERING_POS, FEATURES_BY_YEAR_UF_CSV, CLUSTERING_DATA_PATHS
)
from src.data_processing import processar_dados_vsr, preparar_dados_para_clustering
from src.clustering_analysis import executar_clustering
from src.extract_features import extrair_caracteristicas_ano_uf
from src.dashboards import run_simple_dashboards
from src.metrics_calculation import (
    analisar_caracteristicas_clusters, analisar_metricas_uf, 
    calcular_e_consolidar_metricas
)
from src.visualization import gerar_graficos_caracteristicas, gerar_boxplot_com_siglas
from src.network_analysis import run_network_analysis

def setup_directories():
    dirs = [
        RAW_DATA_DIR, PROCESSED_DATA_DIR, CLUSTER_RESULTS_DIR, METRICS_RESULTS_DIR,
        CLUSTER_FIGURES_DIR, BOXPLOT_FIGURES_DIR, DASHBOARD_FIGURES_DIR
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="Projeto VSR - UFOP")
    parser.add_argument('step', nargs='?', default='all', 
                        choices=['all', 'clean', 'process_raw', 'prepare_clustering', 
                                 'run_clustering', 'extract_features', 'run_dashboards',
                                 'analyze_clusters', 'analyze_uf_metrics', 'run_network'],
                        help="Etapa a ser executada.")
    parser.add_argument('--metric', default='dtw', choices=['dtw', 'euclidean'],
                        help="Métrica de distância para o clustering (padrão: dtw).")
    
    args = parser.parse_args()
    limpar_cache_python()
    
    if args.step == 'clean':
        if os.path.exists(PROCESSED_DATA_DIR): shutil.rmtree(PROCESSED_DATA_DIR)
        if os.path.exists("reports"): shutil.rmtree("reports")
        if os.path.exists("data/results"): shutil.rmtree("data/results")
        print("✅ Limpeza concluída!")
        return

    setup_directories()
    k_valores = [4, 5, 6]
    periodos = ['PRE_PANDEMIA', 'PANDEMIA', 'POS_PANDEMIA']

    if args.step in ['all', 'process_raw']:
        processar_dados_vsr(RAW_DATA_DIR, FILTERED_VSR_DATA)

    if args.step in ['all', 'prepare_clustering']:
        if os.path.exists(FILTERED_VSR_DATA):
            df_filtered = pd.read_csv(FILTERED_VSR_DATA, sep=';', low_memory=False)
            preparar_dados_para_clustering(df_filtered, DADOS_CLUSTERING_PRE, 
                                           DADOS_CLUSTERING_PAN, DADOS_CLUSTERING_POS)
        else:
            print("❌ Execute 'process_raw' primeiro.")

    if args.step in ['all', 'run_clustering']:
        executar_clustering(CLUSTERING_DATA_PATHS, k_valores, metric=args.metric)

    if args.step in ['all', 'extract_features']:
        if os.path.exists(FILTERED_VSR_DATA):
            extrair_caracteristicas_ano_uf(FILTERED_VSR_DATA, FEATURES_BY_YEAR_UF_CSV)

    if args.step in ['all', 'analyze_clusters']:
        if os.path.exists(FILTERED_VSR_DATA):
            # Analisa características para todos os K para garantir gráficos de todos
            for k in k_valores:
                analisar_caracteristicas_clusters(FILTERED_VSR_DATA, k, periodos, gerar_graficos_caracteristicas, metric=args.metric)
            
            # Consolida métricas de onda
            configs_onda = [{"periodo": p, "dados": CLUSTERING_DATA_PATHS[p]} for p in periodos]
            calcular_e_consolidar_metricas(configs_onda, k_valores)

    if args.step in ['all', 'analyze_uf_metrics']:
        if os.path.exists(FILTERED_VSR_DATA):
            # Gera os boxplots por UF para TODOS os valores de K (4, 5, 6)
            for k in k_valores:
                print(f">>> Gerando boxplots para K={k}...")
                analisar_metricas_uf(FILTERED_VSR_DATA, k, periodos, gerar_boxplot_com_siglas)

    if args.step in ['all', 'run_dashboards']:
        if os.path.exists(FILTERED_VSR_DATA):
            run_simple_dashboards(FILTERED_VSR_DATA)

    if args.step in ['all', 'run_network']:
        if os.path.exists(FILTERED_VSR_DATA):
            run_network_analysis(FILTERED_VSR_DATA)

    print("\n🏁 Execução concluída!")

if __name__ == "__main__":
    main()

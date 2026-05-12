import os
import numpy as np

# Semente aleatória para reprodutibilidade
SEED = 42
np.random.seed(SEED)

# Mapeamento de códigos de UF para siglas
MAP_CODE_TO_UF = {
    '11': 'RO', '12': 'AC', '13': 'AM', '14': 'RR', '15': 'PA', '16': 'AP', '17': 'TO',
    '21': 'MA', '22': 'PI', '23': 'CE', '24': 'RN', '25': 'PB', '26': 'PE', '27': 'AL', '28': 'SE', '29': 'BA',
    '31': 'MG', '32': 'ES', '33': 'RJ', '35': 'SP',
    '41': 'PR', '42': 'SC', '43': 'RS',
    '50': 'MS', '51': 'MT', '52': 'GO', '53': 'DF'
}

# Índices de corte para períodos (semanas contínuas desde 2015)
IDX_FIM_PRE = 269  # Semana 10 de 2020
IDX_FIM_PAN = 415  # Semana 52 de 2022

# Colunas necessárias para o dashboard e análises (conforme VSR.py original)
COLUNAS_DASHBOARD = [
    'DT_SIN_PRI', 'DT_INTERNA', 'DT_EVOLUCA',
    'TP_IDADE', 'NU_IDADE_N', 'NU_IDADE',
    'SUPORT_VEN', 'VENTILATION',
    'UTI', 'ICU',
    'EVOLUCAO',
    'SG_UF_NOT', 'REGION',
    'CS_SEXO', 'SEX',
    'CS_RACA', 'RACE', "SEM_NOT",
    'CS_ESCOL_N', 'PCR_VSR', 'AN_VSR', 'PCR_OUTRO', 'CLASSI_FIN', 'RES_VSR'
]

# Configurações de Pastas
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
METRICS_RESULTS_DIR = os.path.join(RESULTS_DIR, "metrics")
CLUSTER_RESULTS_DIR = os.path.join(RESULTS_DIR, "clustering")

REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
CLUSTER_FIGURES_DIR = os.path.join(FIGURES_DIR, "clustering")
BOXPLOT_FIGURES_DIR = os.path.join(FIGURES_DIR, "boxplots")
DASHBOARD_FIGURES_DIR = os.path.join(FIGURES_DIR, "dashboards")

# Caminhos de Arquivos Específicos
FILTERED_VSR_DATA = os.path.join(PROCESSED_DATA_DIR, "VSR_2015_2025_FILTRADO.csv")

# Variáveis que o main.py tenta importar
DADOS_CLUSTERING_PRE = os.path.join(PROCESSED_DATA_DIR, "dados_clustering_pre.csv")
DADOS_CLUSTERING_PAN = os.path.join(PROCESSED_DATA_DIR, "dados_clustering_pan.csv")
DADOS_CLUSTERING_POS = os.path.join(PROCESSED_DATA_DIR, "dados_clustering_pos.csv")

RESULTADO_K_PERIODO_CSV = os.path.join(CLUSTER_RESULTS_DIR, "resultado_k{}_{}.csv")
FEATURES_BY_YEAR_UF_CSV = os.path.join(METRICS_RESULTS_DIR, "CARACTERISTICAS_POR_ANO_UF.csv")

# Dicionário para o clustering_analysis.py
CLUSTERING_DATA_PATHS = {
    'PRE_PANDEMIA': DADOS_CLUSTERING_PRE,
    'PANDEMIA': DADOS_CLUSTERING_PAN,
    'POS_PANDEMIA': DADOS_CLUSTERING_POS
}

# Constantes adicionais para métricas e tabelas analíticas
CHARACTERISTICS_CLUSTERS_CSV = os.path.join(METRICS_RESULTS_DIR, "caracteristicas_clusters_k{}_{}.csv")
METRICS_PER_UF_CSV = os.path.join(METRICS_RESULTS_DIR, "metricas_por_uf_k{}_{}.csv")
GENERAL_UFS_ANALYTICAL_TABLE = os.path.join(METRICS_RESULTS_DIR, "tabela_analitica_geral_ufs_{}.csv")
GENERAL_CLUSTERS_SUMMARY_TABLE = os.path.join(METRICS_RESULTS_DIR, "resumo_geral_clusters_{}.csv")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import gc

from tslearn.utils import to_time_series_dataset
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.metrics import cdist_dtw
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score, pairwise_distances

from src.config import SEED, CLUSTER_FIGURES_DIR, RESULTADO_K_PERIODO_CSV

# Garantir semente aleatória global
np.random.seed(SEED)

def carregar_dados_clustering(caminho):
    """ Carrega dados e aplica normalização Z-score por série. """
    print(f">>> Lendo {caminho}...", flush=True)
    if not os.path.exists(caminho):
        print(f"❌ Arquivo não encontrado: {caminho}", flush=True)
        return None, None
    df = pd.read_csv(caminho, index_col=0)
    if df.empty:
        print(f"⚠️ Arquivo vazio: {caminho}", flush=True)
        return None, None
    
    # Preencher NaNs com 0 antes de normalizar
    data_values = df.fillna(0).values.astype(np.float32)
    
    # NORMALIZAÇÃO Z-SCORE (Média 0, Variância 1)
    scaler = TimeSeriesScalerMeanVariance()
    X_bruto = to_time_series_dataset(data_values)
    X_norm = scaler.fit_transform(X_bruto)
    X_norm = np.nan_to_num(X_norm)
    
    return df, X_norm

def calcular_matriz_distancia(X_ts, metric="dtw"):
    """ Calcula a matriz de distância usando DTW ou Euclidiana. """
    print(f" ... Calculando matriz de distância ({metric})...", flush=True)
    if metric.lower() == "dtw":
        dist_matrix = cdist_dtw(X_ts, n_jobs=1)
    else:
        # Para euclidiana, redimensionamos de volta para (n_series, n_times)
        X_flat = X_ts.reshape(X_ts.shape[0], -1)
        dist_matrix = pairwise_distances(X_flat, metric="euclidean")
        
    dist_matrix = np.nan_to_num(dist_matrix, posinf=1e6, neginf=0)
    return dist_matrix

def rodar_clustering(df, X_tslearn, dist_matrix, n_clusters, nome_periodo, metric="dtw"):
    """ Executa o K-Medoids para um valor de K e período específico. """
    n_samples = len(df)
    if n_clusters >= n_samples:
        print(f" ⚠️ Pulando K={n_clusters} para {nome_periodo}: Amostras insuficientes.", flush=True)
        return -1

    print(f" ... Clustering K={n_clusters} ({metric}) em {nome_periodo}...", flush=True)
    model = KMedoids(n_clusters=n_clusters,
                     metric="precomputed",
                     method="pam",
                     init="k-medoids++",
                     max_iter=300,
                     random_state=SEED)
    
    try:
        y_pred = model.fit_predict(dist_matrix)
        silhouette = silhouette_score(dist_matrix, y_pred, metric="precomputed")
    except Exception as e:
        print(f" ❌ Erro no clustering: {e}", flush=True)
        return -1

    # Salvar resultados
    resultado = pd.DataFrame({"UF": df.index, "Cluster": y_pred})
    csv_saida = RESULTADO_K_PERIODO_CSV.format(n_clusters, nome_periodo, metric)
    os.makedirs(os.path.dirname(csv_saida), exist_ok=True)
    resultado.to_csv(csv_saida, index=False)

    # Gráfico de Clusters
    plt.figure(figsize=(15, 8))
    cols = (n_clusters + 1) // 2
    for yi in range(n_clusters):
        plt.subplot(2, cols, yi + 1)
        cluster_data = X_tslearn[y_pred == yi]
        for xx in cluster_data:
            plt.plot(xx.ravel(), "k-", alpha=.2)
        
        medoid_idx = model.medoid_indices_[yi]
        plt.plot(X_tslearn[medoid_idx].ravel(), "r-", linewidth=2)
        plt.title(f"Cluster {yi} (n={len(cluster_data)})")
        plt.grid(True, alpha=0.3)
    
    plt.suptitle(f"K-Medoids PAM ({metric}) - {nome_periodo} (K={n_clusters}) | Sil: {silhouette:.4f}")
    plt.tight_layout()
    figure_path = os.path.join(CLUSTER_FIGURES_DIR,metric.lower(),f"cluster_k{n_clusters}_{nome_periodo}_{metric}.png")
    os.makedirs(os.path.dirname(figure_path), exist_ok=True)
    plt.savefig(figure_path, dpi=300)
    plt.close()
    
    gc.collect()
    return silhouette

def executar_clustering(periodos_data_paths, k_valores, metric="dtw"):
    """ Orquestra o clustering para todos os cenários. """
    print(f"=== INICIANDO CLUSTERING (Métrica: {metric.upper()}) ===", flush=True)
    for periodo, caminho in periodos_data_paths.items():
        df, X_ts = carregar_dados_clustering(caminho)
        if df is not None:
            matriz_distancia = calcular_matriz_distancia(X_ts, metric=metric)
            for k in k_valores:
                rodar_clustering(df, X_ts, matriz_distancia, k, periodo, metric=metric)
            del matriz_distancia, df, X_ts
            gc.collect()
    print("\n🏁 Clustering finalizado!", flush=True)

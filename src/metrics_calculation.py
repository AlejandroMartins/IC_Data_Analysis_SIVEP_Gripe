import pandas as pd
import numpy as np
import os
import gc
from src.config import (
    MAP_CODE_TO_UF, METRICS_RESULTS_DIR, 
    CHARACTERISTICS_CLUSTERS_CSV, METRICS_PER_UF_CSV, 
    GENERAL_UFS_ANALYTICAL_TABLE, GENERAL_CLUSTERS_SUMMARY_TABLE,
    RESULTADO_K_PERIODO_CSV
)
from src.data_processing import try_parse_date, classificar_periodo, calcular_idade_unificada_row

def pre_processar_dados_pacientes(caminho):
    """ Processa dados brutos para cálculo de métricas clínicas com fidelidade total. """
    print(">>> Processando dados brutos para métricas...", flush=True)
    if not os.path.exists(caminho): return pd.DataFrame()
    
    try:
        df = pd.read_csv(caminho, sep=";", encoding="utf-8", low_memory=False)
    except:
        df = pd.read_csv(caminho, sep=";", encoding="latin-1", low_memory=False)

    df.columns = [str(col).upper() for col in df.columns]
    df["DT_SIN_PRI"] = try_parse_date(df["DT_SIN_PRI"])
    df = df.dropna(subset=["DT_SIN_PRI"])
    df["ANO"] = df["DT_SIN_PRI"].dt.year.astype(int)
    df["SEMANA"] = df["DT_SIN_PRI"].dt.strftime('%U').astype(int) + 1
    df.loc[df["SEMANA"] > 52, "SEMANA"] = 52
    
    # Adiciona a coluna de ano original para a função de idade
    if 'YEAR' not in df.columns:
        df['year'] = df['ANO']
    else:
        df['year'] = df['YEAR']

    # LÓGICA DE IDADE UNIFICADA DO USUÁRIO
    df['IDADE_ANOS'] = df.apply(calcular_idade_unificada_row, axis=1)
    df['IDADE_ANOS'] = pd.to_numeric(df['IDADE_ANOS'], errors='coerce')

    df["TIME_IDX"] = (df["ANO"] - 2015) * 52 + (df["SEMANA"] - 1)
    df["PERIODO"] = df["TIME_IDX"].apply(classificar_periodo)
    
    df["SG_UF_NOT"] = df["SG_UF_NOT"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    df["SG_UF_NOT"] = df["SG_UF_NOT"].replace(MAP_CODE_TO_UF)

    df["RACA_CAT"] = "Outras/Ign"
    df.loc[df["CS_RACA"].astype(str).isin(["1", "1.0"]), "RACA_CAT"] = "Branca"
    df.loc[df["CS_RACA"].astype(str).isin(["2", "2.0", "4", "4.0"]), "RACA_CAT"] = "Preta/Parda"

    for col in ["UTI", "SUPORT_VEN", "EVOLUCAO"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    return df

def calcular_estatisticas_cluster(grupo):
    """ Calcula estatísticas descritivas para um cluster. """
    total = len(grupo)
    if total == 0: return pd.Series()
    
    perc_masc = (grupo["CS_SEXO"] == "M").sum() / total * 100
    perc_lt_1 = (grupo["IDADE_ANOS"] < 1).sum() / total * 100
    perc_1_a_5 = ((grupo["IDADE_ANOS"] >= 1) & (grupo["IDADE_ANOS"] <= 5)).sum() / total * 100
    perc_60_plus = (grupo["IDADE_ANOS"] >= 60).sum() / total * 100
    
    # Lógica de denominador válido do usuário
    total_uti = grupo['UTI'].isin(['1', '2']).sum()
    perc_uti = (grupo["UTI"] == "1").sum() / total_uti * 100 if total_uti > 0 else 0
    
    total_sup = grupo['SUPORT_VEN'].isin(['1', '2', '3']).sum()
    perc_sup_inv = (grupo["SUPORT_VEN"] == "1").sum() / total_sup * 100 if total_sup > 0 else 0
    
    total_evol = grupo['EVOLUCAO'].isin(['1', '2', '3']).sum()
    perc_obito = (grupo["EVOLUCAO"].isin(["2", "3"])).sum() / total_evol * 100 if total_evol > 0 else 0
    
    return pd.Series({
        "Total_Pacientes": total, "% Masculino": round(perc_masc, 2), "% < 1 ano": round(perc_lt_1, 2),
        "% 1 a 5 anos": round(perc_1_a_5, 2), "% 60+ anos": round(perc_60_plus, 2),
        "% UTI": round(perc_uti, 2), "% Suporte Invasivo": round(perc_sup_inv, 2), "% Obito": round(perc_obito, 2)
    })

def analisar_caracteristicas_clusters(arquivo_bruto, k_escolhido, periodos, gerar_graficos_func, metric="dtw"):
    print(f"=== ANALISE DE CARACTERISTICAS (K={k_escolhido}) ===", flush=True)
    df_pacientes = pre_processar_dados_pacientes(arquivo_bruto)
    if df_pacientes.empty: return

    lista_estat = []
    for periodo in periodos:
        arquivo_ids = RESULTADO_K_PERIODO_CSV.format(k_escolhido, periodo, metric)
        if not os.path.exists(arquivo_ids): continue
        df_ids = pd.read_csv(arquivo_ids)
        df_p = df_pacientes[df_pacientes["PERIODO"] == periodo].copy()
        mapa = dict(zip(df_ids["UF"], df_ids["Cluster"]))
        df_p["Cluster"] = df_p["SG_UF_NOT"].map(mapa)
        df_p = df_p.dropna(subset=["Cluster"])
        df_p["Cluster"] = df_p["Cluster"].astype(int)

        estat = df_p.groupby("Cluster").apply(calcular_estatisticas_cluster)
        estat["Estados_do_Cluster"] = df_p.groupby("Cluster")["SG_UF_NOT"].apply(lambda x: ", ".join(sorted(x.unique())))
        estat["PERIODO"] = periodo
        lista_estat.append(estat)
        gerar_graficos_func(estat, periodo, k_escolhido, metric=metric)

    if lista_estat:
        df_f = pd.concat(lista_estat)
        os.makedirs(METRICS_RESULTS_DIR, exist_ok=True)
        df_f.to_csv(CHARACTERISTICS_CLUSTERS_CSV.format(k_escolhido, metric), sep=";", decimal=",", encoding="utf-8-sig")

def calcular_metricas_por_estado(serie):
    v = serie.values
    n_anos = len(v) // 52
    if n_anos == 0: return {"Pico_Medio_Semana": 0, "Duracao_Media_Semanas": 0, "Amplitude_Media": 0}
    matriz = v[:n_anos*52].reshape(n_anos, 52)
    picos, duracoes, amplitudes = [], [], []
    for ano_dados in matriz:
        p_val = np.max(ano_dados)
        if p_val == 0: continue
        picos.append(np.argmax(ano_dados) + 1)
        duracoes.append(np.sum(ano_dados > 0.4 * p_val))
        amplitudes.append(p_val - np.min(ano_dados))
    return {
        "Pico_Medio_Semana": np.mean(picos) if picos else 0, 
        "Duracao_Media_Semanas": np.mean(duracoes) if duracoes else 0, 
        "Amplitude_Media": np.mean(amplitudes) if amplitudes else 0
    }

def calcular_e_consolidar_metricas(configs, k_lista,metric="dtw"):
    print("=== CONSOLIDANDO MÉTRICAS DE ONDA ===", flush=True)
    dados_ufs, dados_clusters = [], []
    for cfg in configs:
        periodo, arquivo = cfg["periodo"], cfg["dados"]
        if not os.path.exists(arquivo): continue
        df_s = pd.read_csv(arquivo, index_col=0)
        for k in k_lista:
            arquivo_c = RESULTADO_K_PERIODO_CSV.format(k, periodo, metric)
            if not os.path.exists(arquivo_c): continue
            df_c = pd.read_csv(arquivo_c)
            metricas_deste_cenario = []
            for _, row in df_c.iterrows():
                uf, cid = row["UF"], row["Cluster"]
                if uf in df_s.index:
                    m = calcular_metricas_por_estado(df_s.loc[uf])
                    reg = {"Periodo": periodo, "K": k, "UF": uf, "Cluster": cid, 
                           "Pico_Medio_Semana": round(m["Pico_Medio_Semana"], 2),
                           "Duracao_Media_Semanas": round(m["Duracao_Media_Semanas"], 2),
                           "Amplitude_Media": round(m["Amplitude_Media"], 4)}
                    dados_ufs.append(reg)
                    metricas_deste_cenario.append(reg)
            
            df_temp = pd.DataFrame(metricas_deste_cenario)
            if not df_temp.empty:
                grupo = df_temp.groupby('Cluster')[['Pico_Medio_Semana', 'Duracao_Media_Semanas', 'Amplitude_Media']].mean()
                contagem = df_temp['Cluster'].value_counts().sort_index()
                lista_ufs = df_temp.groupby('Cluster')['UF'].apply(lambda x: ', '.join(x))
                for cid in grupo.index:
                    dados_clusters.append({
                        'Periodo': periodo, 'K': k, 'Cluster': cid, 'Qtd_Estados': contagem[cid],
                        'UFs_no_Cluster': lista_ufs[cid],
                        'Media_Pico_Semana': round(grupo.loc[cid, 'Pico_Medio_Semana'], 2),
                        'Media_Duracao_Semanas': round(grupo.loc[cid, 'Duracao_Media_Semanas'], 2),
                        'Media_Amplitude': round(grupo.loc[cid, 'Amplitude_Media'], 4)
                    })
    
    if dados_ufs:
        os.makedirs(METRICS_RESULTS_DIR, exist_ok=True)
        pd.DataFrame(dados_ufs).to_csv(GENERAL_UFS_ANALYTICAL_TABLE, index=False, sep=";", decimal=",", encoding="utf-8-sig")
    if dados_clusters:
        pd.DataFrame(dados_clusters).to_csv(GENERAL_CLUSTERS_SUMMARY_TABLE.format(metric), index=False, sep=";", decimal=",", encoding="utf-8-sig")

def analisar_metricas_uf(arquivo_bruto, k_escolhido, periodos, gerar_boxplot_func,metric="dtw"):
    print(f"=== ANALISE METRICAS POR UF (K={k_escolhido}) ===", flush=True)
    df_pacientes = pre_processar_dados_pacientes(arquivo_bruto)
    if df_pacientes.empty: return
    lista_m = []
    for periodo in periodos:
        arquivo_ids = RESULTADO_K_PERIODO_CSV.format(k_escolhido, periodo, metric)
        if not os.path.exists(arquivo_ids): continue
        df_ids = pd.read_csv(arquivo_ids)
        df_p = df_pacientes[df_pacientes["PERIODO"] == periodo].copy()
        mapa = dict(zip(df_ids["UF"], df_ids["Cluster"]))
        df_p["Cluster"] = df_p["SG_UF_NOT"].map(mapa)
        df_p = df_p.dropna(subset=["Cluster"])
        df_p["Cluster"] = df_p["Cluster"].astype(int)

        m_uf = df_p.groupby(["SG_UF_NOT", "Cluster"]).apply(lambda x: pd.Series({
            "Mediana_Idade": x["IDADE_ANOS"].median(),
            "%_UTI": (x["UTI"] == "1").sum() / x['UTI'].isin(['1','2']).sum() * 100 if x['UTI'].isin(['1','2']).sum() > 0 else 0,
            "%_Suporte_Invasivo": (x["SUPORT_VEN"] == "1").sum() / x['SUPORT_VEN'].isin(['1','2','3']).sum() * 100 if x['SUPORT_VEN'].isin(['1','2','3']).sum() > 0 else 0,
            "%_Obito": (x["EVOLUCAO"].isin(["2", "3"])).sum() / x['EVOLUCAO'].isin(['1','2','3']).sum() * 100 if x['EVOLUCAO'].isin(['1','2','3']).sum() > 0 else 0
        })).reset_index()
        m_uf["Periodo"] = periodo
        m_uf["K"] = k_escolhido
        lista_m.append(m_uf)
        if not m_uf.empty:
            for metrica in ["Mediana_Idade", "%_UTI", "%_Suporte_Invasivo", "%_Obito"]:
                gerar_boxplot_func(m_uf, metrica, f"{metrica} - {periodo}", metrica, f"boxplot_{metrica}_{periodo}.png")

    if lista_m:
        pd.concat(lista_m).to_csv(METRICS_PER_UF_CSV.format(k_escolhido, metric), index=False, sep=";", decimal=",", encoding="utf-8-sig")

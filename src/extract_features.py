import pandas as pd
import numpy as np
import os
import gc
from src.config import MAP_CODE_TO_UF
from src.data_processing import try_parse_date

def calcular_metricas_grupo_usuario(group):
    """
    Recebe todos os pacientes de UMA UF em UM ANO e retorna um resumo das métricas.
    Lógica de idade e métricas clínicas idêntica ao código original do usuário.
    """
    total_pacientes = len(group)
    if total_pacientes == 0:
        return pd.Series()

    # 1. MÉTRICAS DA ONDA (Pico, Duração, Amplitude)
    casos_por_semana = group.groupby('SEMANA').size()
    if not casos_por_semana.empty:
        amplitude = casos_por_semana.max()
        pico = casos_por_semana.idxmax()
        duracao = len(casos_por_semana[casos_por_semana > 0])
    else:
        amplitude = pico = duracao = 0

    # 2. MÉTRICAS DEMOGRÁFICAS (Idade)
    idade_media = group['IDADE_ANOS'].mean()
    idade_mediana = group['IDADE_ANOS'].median()

    # 3. FAIXAS ETÁRIAS (Porcentagens sobre o TOTAL de pacientes do grupo)
    perc_le_6m = (group['IDADE_ANOS'] <= 0.5).sum() / total_pacientes * 100
    perc_6m_1a = ((group['IDADE_ANOS'] > 0.5) & (group['IDADE_ANOS'] <= 1.0)).sum() / total_pacientes * 100
    perc_1_5a = ((group['IDADE_ANOS'] > 1.0) & (group['IDADE_ANOS'] <= 5.0)).sum() / total_pacientes * 100
    perc_5_12a = ((group['IDADE_ANOS'] > 5.0) & (group['IDADE_ANOS'] <= 12.0)).sum() / total_pacientes * 100
    perc_13_19a = ((group['IDADE_ANOS'] > 12.0) & (group['IDADE_ANOS'] <= 19.0)).sum() / total_pacientes * 100
    perc_20_59a = ((group['IDADE_ANOS'] > 19.0) & (group['IDADE_ANOS'] <= 59.0)).sum() / total_pacientes * 100
    perc_60p = (group['IDADE_ANOS'] >= 60.0).sum() / total_pacientes * 100

    # 4. MÉTRICAS CLÍNICAS (UTI, Suporte, Óbito)
    # Filtra apenas os valores válidos para o denominador (1 e 2 para UTI, etc.)
    total_com_uti = group['UTI'].isin(['1', '2']).sum()
    perc_uti = (group['UTI'] == '1').sum() / total_com_uti * 100 if total_com_uti > 0 else np.nan

    total_com_sup = group['SUPORT_VEN'].isin(['1', '2', '3']).sum()
    perc_sup_inv = (group['SUPORT_VEN'] == '1').sum() / total_com_sup * 100 if total_com_sup > 0 else np.nan

    total_com_evol = group['EVOLUCAO'].isin(['1', '2', '3']).sum()
    perc_obito = (group['EVOLUCAO'].isin(['2', '3'])).sum() / total_com_evol * 100 if total_com_evol > 0 else np.nan

    # 5. MÉTRICAS DE RAÇA/COR
    total_com_raca = group['CS_RACA'].isin(['1', '2', '3', '4', '5']).sum()
    perc_branca = (group['CS_RACA'] == '1').sum() / total_com_raca * 100 if total_com_raca > 0 else np.nan
    perc_preta_parda = group['CS_RACA'].isin(['2', '4']).sum() / total_com_raca * 100 if total_com_raca > 0 else np.nan

    return pd.Series({
        'Total_Pacientes': total_pacientes,
        'Pico_Semana': pico,
        'Duracao_Semanas': duracao,
        'Amplitude_Max': amplitude,
        'Idade_Media': idade_media,
        'Idade_Mediana': idade_mediana,
        '%_Ate_6_Meses': perc_le_6m,
        '%_6_Meses_a_1_Ano': perc_6m_1a,
        '%_1_a_5_Anos': perc_1_5a,
        '%_5_a_12_Anos': perc_5_12a,
        '%_13_a_19_Anos': perc_13_19a,
        '%_20_a_59_Anos': perc_20_59a,
        '%_60_Mais_Anos': perc_60p,
        '%_UTI': perc_uti,
        '%_Suporte_Invasivo': perc_sup_inv,
        '%_Obito': perc_obito,
        '%_Raca_Branca': perc_branca,
        '%_Raca_Preta_Parda': perc_preta_parda
    })

def extrair_caracteristicas_ano_uf(input_path, output_path):
    print(">>> Extraindo características por Ano e UF (Fidelidade Total)...", flush=True)
    if not os.path.exists(input_path):
        print(f"❌ Erro: Arquivo {input_path} não encontrado.")
        return

    try:
        df = pd.read_csv(input_path, sep=';', low_memory=False)
    except:
        df = pd.read_csv(input_path, sep=';', encoding='latin-1', low_memory=False)

    df['DT_SIN_PRI'] = try_parse_date(df['DT_SIN_PRI'])
    df = df.dropna(subset=['DT_SIN_PRI'])
    df['ANO_DADO'] = df['DT_SIN_PRI'].dt.year.astype(int)
    df['IDADE_ANOS'] = pd.to_numeric(df['IDADE_ANOS'], errors='coerce')
    
    df['SEMANA'] = df['DT_SIN_PRI'].dt.strftime('%U').astype(int) + 1
    df.loc[df['SEMANA'] > 52, 'SEMANA'] = 52
    
    df['SG_UF_NOT'] = df['SG_UF_NOT'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    df['SG_UF_NOT'] = df['SG_UF_NOT'].replace(MAP_CODE_TO_UF)
    
    for col in ['UTI', 'SUPORT_VEN', 'EVOLUCAO', 'CS_RACA']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()

    print("   -> Agregando dados por ANO e UF...", flush=True)
    # Agrupamos por year (coluna fixa do processamento) e UF
    df_agregado = df.groupby(['year', 'SG_UF_NOT']).apply(calcular_metricas_grupo_usuario).reset_index()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_agregado.to_csv(output_path, index=False, sep=';', decimal=',', encoding='utf-8-sig')
    
    print(f"✅ Base criada com sucesso em: {output_path}")
    del df, df_agregado
    gc.collect()

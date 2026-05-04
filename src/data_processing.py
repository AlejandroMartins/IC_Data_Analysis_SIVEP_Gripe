import pandas as pd
import numpy as np
import os
import gc
from statsmodels.tsa.seasonal import STL
from src.config import MAP_CODE_TO_UF, COLUNAS_DASHBOARD

def try_parse_date(series):
    """ Tenta converter datas de forma robusta. """
    s1 = pd.to_datetime(series, errors='coerce', dayfirst=True, format='%d/%m/%Y')
    s2 = pd.to_datetime(series, errors='coerce', dayfirst=False, format='%Y-%m-%d')
    return s1.fillna(s2)

def parse_age_old_format(val):
    """ Processa o formato antigo de idade. """
    try:
        s = str(int(float(val)))
        if not s.isdigit() or len(s) < 2: return np.nan
        unidade, numero = int(s[0]), int(s[1:])
        if unidade == 1: return numero / 8760.0
        if unidade == 2: return numero / 365.0
        if unidade == 3: return numero / 12.0
        if unidade == 4: return float(numero)
    except: return np.nan
    return np.nan

def parse_age_new_format(val):
    """ Processa o formato novo de idade. """
    try: return float(val)
    except: return np.nan

def calcular_idade_unificada_row(row):
    """ Lógica de idade unificada. """
    year = row.get('year')
    age_val = row.get('NU_IDADE_N', row.get('NU_IDADE', row.get('nu_idade_n')))
    if pd.isna(year) or pd.isna(age_val): return np.nan
    s_age_val = str(age_val).replace(',', '.')
    try:
        y_int = int(float(year))
        if y_int <= 2018:
            return parse_age_old_format(s_age_val)
        else:
            return parse_age_new_format(s_age_val)
    except:
        return np.nan

def processar_dados_vsr(raw_dir, output_path, chunk_size=50000):
    """ Lê arquivos brutos em chunks para economizar RAM. """
    print(f">>> Processando arquivos brutos (Chunks de {chunk_size})...", flush=True)
    all_filtered_dataframes = []
    
    if not os.path.exists(raw_dir):
        print(f"❌ Erro: Diretório {raw_dir} não encontrado.")
        return None
        
    files = sorted([f for f in os.listdir(raw_dir) if f.endswith('.csv')])
    
    for f in files:
        try:
            ano_s = ''.join(filter(str.isdigit, f))
            ano = int(ano_s)
            if ano < 100: ano += 2000
        except: continue
            
        print(f" ... Lendo {f} (Ano {ano})", flush=True)
        path = os.path.join(raw_dir, f)
        
        try:
            sample = pd.read_csv(path, sep=';', nrows=1, encoding='ISO-8859-1')
        except:
            sample = pd.read_csv(path, sep=';', nrows=1, encoding='utf-8')
        
        available_cols = [c for c in COLUNAS_DASHBOARD if c.upper() in [str(x).upper() for x in sample.columns]]
        for essential in ['RES_VSR', 'PCR_VSR', 'AN_VSR', 'CLASSI_FIN', 'SG_UF_NOT', 'DT_SIN_PRI', 'NU_IDADE_N', 'NU_IDADE', 'PCR_OUTRO']:
            if essential.upper() in [str(x).upper() for x in sample.columns] and essential not in available_cols:
                available_cols.append(essential)

        try:
            reader = pd.read_csv(path, sep=';', encoding='ISO-8859-1', usecols=available_cols, chunksize=chunk_size, low_memory=False, on_bad_lines='skip')
        except:
            reader = pd.read_csv(path, sep=';', encoding='utf-8', usecols=available_cols, chunksize=chunk_size, low_memory=False, on_bad_lines='skip')

        total_vsr_year = 0
        for chunk in reader:
            chunk.columns = [str(col).upper() for col in chunk.columns]
            chunk['year'] = ano
            
            if ano <= 2018:
                res_vsr = pd.to_numeric(chunk.get('RES_VSR'), errors='coerce') == 1
                pcr_outro = chunk.get('PCR_OUTRO', pd.Series("", index=chunk.index)).astype(str).str.contains('VSR|SINCICIAL|SINCIAL', case=False, na=False)
                cond_vsr = res_vsr | pcr_outro
            else:
                pcr = pd.to_numeric(chunk.get('PCR_VSR'), errors='coerce') == 1
                an = pd.to_numeric(chunk.get('AN_VSR'), errors='coerce') == 1
                cond_vsr = pcr | an
            
            df_chunk_filtrado = chunk[cond_vsr].copy()
            if not df_chunk_filtrado.empty:
                df_chunk_filtrado['IDADE_ANOS'] = df_chunk_filtrado.apply(calcular_idade_unificada_row, axis=1)
                all_filtered_dataframes.append(df_chunk_filtrado)
                total_vsr_year += len(df_chunk_filtrado)

        print(f"   ✅ {total_vsr_year} casos VSR encontrados em {ano}.", flush=True)
        gc.collect()

    if all_filtered_dataframes:
        full_df = pd.concat(all_filtered_dataframes, ignore_index=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        full_df.to_csv(output_path, index=False, sep=';', encoding='utf-8-sig')
        print(f"\n✅ Arquivo consolidado salvo em {output_path}", flush=True)
        return full_df
    return None

def preparar_dados_para_clustering(filtered_df, output_pre, output_pan, output_pos):
    """ Prepara matrizes para clustering com blindagem contra TypeError. """
    print(">>> Preparando matrizes para clustering (STL Global)...", flush=True)
    
    # 1. Garantia absoluta de tipo para SG_UF_NOT
    # Primeiro, converte tudo para string, trata nulos e limpa espaços
    filtered_df['SG_UF_NOT'] = filtered_df['SG_UF_NOT'].fillna('MISSING').astype(str).str.strip()
    
    # Remove qualquer valor que ainda pareça nulo ou seja 'nan' (como string)
    filtered_df = filtered_df[~filtered_df['SG_UF_NOT'].isin(['nan', 'None', 'MISSING', ''])]
    
    # Mapeia códigos para siglas
    filtered_df['SG_UF_NOT'] = filtered_df['SG_UF_NOT'].replace(MAP_CODE_TO_UF)
    
    # 2. Tratamento de Datas
    filtered_df['DT_SIN_PRI'] = try_parse_date(filtered_df['DT_SIN_PRI'])
    filtered_df = filtered_df.dropna(subset=['DT_SIN_PRI'])
    
    filtered_df['ANO'] = filtered_df['DT_SIN_PRI'].dt.year.astype(int)
    filtered_df['SEMANA'] = filtered_df['DT_SIN_PRI'].dt.strftime('%U').astype(int) + 1
    filtered_df.loc[filtered_df['SEMANA'] > 52, 'SEMANA'] = 52
    
    # 3. Ordenação Segura (Blindagem contra TypeError)
    # Extraímos os valores únicos, garantimos que são strings e ordenamos
    ufs_unicas = [str(x) for x in filtered_df['SG_UF_NOT'].unique() if pd.notna(x)]
    ufs = sorted(ufs_unicas)
    
    time_indices = range(0, (2025-2015+1)*52)
    matriz_bruta = pd.DataFrame(0, index=ufs, columns=time_indices)
    
    for (ano, sem, uf), count in filtered_df.groupby(['ANO', 'SEMANA', 'SG_UF_NOT']).size().items():
        idx = (ano - 2015) * 52 + (sem - 1)
        # Garantimos que o UF está na matriz (já que filtramos nulos antes)
        if idx in matriz_bruta.columns and str(uf) in matriz_bruta.index:
            matriz_bruta.loc[str(uf), idx] = count
    
    print(">>> Aplicando STL Global...", flush=True)
    df_sazonal = pd.DataFrame(index=matriz_bruta.index, columns=matriz_bruta.columns)
    for uf in matriz_bruta.index:
        serie = matriz_bruta.loc[uf].values
        if np.all(serie == 0):
            df_sazonal.loc[uf] = 0
            continue
        try:
            res = STL(serie, period=52, seasonal=13, robust=True).fit()
            df_sazonal.loc[uf] = res.seasonal
        except:
            df_sazonal.loc[uf] = 0
    
    from src.config import IDX_FIM_PRE, IDX_FIM_PAN
    sazonal_pre = df_sazonal.iloc[:, :IDX_FIM_PRE + 1]
    sazonal_pan = df_sazonal.iloc[:, IDX_FIM_PRE + 1 : IDX_FIM_PAN + 1]
    sazonal_pos = df_sazonal.iloc[:, IDX_FIM_PAN + 1 :]
    
    sazonal_pre.to_csv(output_pre)
    sazonal_pan.to_csv(output_pan)
    sazonal_pos.to_csv(output_pos)
    print("✅ Matrizes de sazonalidade salvas.", flush=True)
    gc.collect()

def classificar_periodo(time_idx):
    """ Classifica o índice temporal nos períodos definidos no projeto. """
    from src.config import IDX_FIM_PRE, IDX_FIM_PAN
    if time_idx <= IDX_FIM_PRE:
        return 'PRE_PANDEMIA'
    elif time_idx <= IDX_FIM_PAN:
        return 'PANDEMIA'
    else:
        return 'POS_PANDEMIA'

# Sistema de Análise Epidemiológica de VSR (2015-2025)

Este projeto é uma plataforma modular desenvolvida em Python para o processamento, análise e visualização de dados do Vírus Sincicial Respiratório (VSR) baseada nos registros do SIVEP-Gripe. O sistema foi projetado para ser eficiente em memória e flexível em suas metodologias analíticas.

---

## 📁 1. Estrutura do Projeto

```text
.
├── README.md                # Este guia completo
├── .gitignore               # Proteção para não subir arquivos pesados ao Git
├── RELATORIO_SISTEMA_VSR.md # Documentação técnica detalhada dos módulos
├── src/                     # Código fonte modular
│   ├── main.py              # Orquestrador central (CLI)
│   ├── config.py            # Configurações globais e caminhos
│   ├── data_processing.py   # Limpeza de dados e decomposição sazonal (STL)
│   ├── clustering_analysis.py # Algoritmos de agrupamento (K-Medoids)
│   ├── network_analysis.py  # Análise de grafos bipartidos (NetworkX)
│   ├── dashboards.py        # Dashboards 4x2 e Séries Temporais
│   ├── metrics_calculation.py # Cálculos clínicos e estatísticas de onda
│   └── visualization.py     # Plotagem de boxplots e perfis de cluster
├── data/
│   ├── raw/                 # Local para os CSVs brutos (INFLUDxx.csv)
│   ├── processed/           # Dados filtrados e matrizes de séries temporais
│   └── results/             # Tabelas CSV com resultados analíticos
└── reports/
    └── figures/             # Gráficos gerados (Dashboards, Redes, Boxplots)
```

---

## 🛠️ 2. Instalação e Configuração

1. **Clone o Repositório**:
   ```bash
   git clone https://github.com/AlejandroMartins/IC_Data_Analysis_SIVEP_Gripe.git
   ```

2. **Crie o Ambiente Virtual**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # No Windows use: .\venv\Scripts\activate
   ```

3. **Instale as Dependências**:
   ```bash
   pip install -r requirements.txt
   ```


4. **Prepare os Dados**:
   Coloque seus arquivos `INFLUD15.csv` até `INFLUD25.csv` dentro da pasta `data/raw/`.

---

## 🚀 3. Como Executar (Comandos CLI)

O sistema é operado através do arquivo `src/main.py`. Você pode rodar etapas individuais ou o fluxo completo.

### 🔄 Fluxo Completo
Executa todas as etapas (processamento, clustering, métricas, dashboards e redes) em sequência:
```bash
python3 -m src.main all
```

### 🧹 Limpeza de Resultados
Remove todas as pastas de resultados e gráficos para iniciar uma nova análise do zero:
```bash
python3 -m src.main clean
```

### 📂 Processamento de Dados
Lê os arquivos brutos, filtra casos positivos de VSR e unifica a lógica de idade (pré/pós 2019):
```bash
python3 -m src.main process_raw
```

### 📊 Clustering de Séries Temporais
Agrupa as UFs por comportamento sazonal. Você pode escolher a métrica de distância:
- **DTW (Padrão)**: Ideal para capturar o formato da onda ignorando atrasos temporais.
- **Euclidiana**: Comparação rígida ponto a ponto.
```bash
python3 -m src.main run_clustering --metric dtw
# OU
python3 -m src.main run_clustering --metric euclidean
```

### 📈 Dashboards e Séries Temporais
Gera o dashboard 4x2 (Perfil Demográfico/Clínico) e séries temporais por SE e Faixa Etária:
```bash
python3 -m src.main run_dashboards
```

### 🕸️ Análise de Grafos (Redes)
Gera grafos bipartidos conectando UFs às Semanas Epidemiológicas de pico para 4 cenários (Geral, Pré, Pandemia e Pós):
```bash
python3 -m src.main run_network
```

### 📦 Boxplots por UF
Gera boxplots de métricas clínicas (UTI, Óbito, Idade) para todos os valores de K (4, 5 e 6):
```bash
python3 -m src.main analyze_uf_metrics
```

---

## 🧠 4. Detalhes Técnicos Importantes

- **Mapeamento de Nomes**: Os gráficos não exibem códigos (1, 2, 3), mas sim nomes reais ("Branca", "Cura", "Sim Inv"), facilitando a leitura científica.
- **Eficiência**: O uso de `chunks` permite processar arquivos de vários gigabytes sem travar o computador.
- **Gitignore**: O arquivo `.gitignore` já está configurado para **não subir** os arquivos CSV pesados, mantendo seu repositório leve e profissional.

---
*Este projeto foi desenvolvido como parte de uma Iniciação Científica na UFOP.*

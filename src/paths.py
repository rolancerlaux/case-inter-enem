from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "DADOS"
DICT_DIR = DATA_DIR / "DICIONÁRIO"
REPORTS_DIR = ROOT_DIR / "reports"
FIGS_DIR = REPORTS_DIR / "figs"

MICRODADOS = RAW_DIR / "MICRODADOS_ENEM_2023.csv"
ITENS_PROVA = RAW_DIR / "ITENS_PROVA_2023.csv"

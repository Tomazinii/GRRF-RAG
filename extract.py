import os
import csv
import logging
from tqdm import tqdm
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import AcceleratorOptions

# Configurações de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# Caminho de entrada e saída
pdf_folder = "./all"
output_csv = "auditoria.csv"

# Configurar o uso do CUDA:1
accelerator_options = AcceleratorOptions(
    num_threads=8,
    device="cuda:1"  # Explicitamente usar CUDA:1
)

# Pipeline com OCR e estrutura de tabela ativada
pipeline_options = PdfPipelineOptions()
pipeline_options.accelerator_options = accelerator_options
pipeline_options.do_ocr = True
pipeline_options.do_table_structure = True
pipeline_options.table_structure_options.do_cell_matching = True

# Ativar profiling
settings.debug.profile_pipeline_timings = True

# Criar o conversor com configurações customizadas
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options
        )
    }
)

# Coleta de PDFs
pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]
logging.info(f"{len(pdf_files)} arquivos PDF encontrados.")

# Lista para armazenar resultados
rows = []

# Processamento com barra de progresso
for filename in tqdm(pdf_files, desc="Convertendo PDFs"):
    file_path = os.path.join(pdf_folder, filename)
    document_id = os.path.splitext(filename)[0]
    try:
        logging.info(f"Iniciando: {filename}")
        result = converter.convert(Path(file_path))
        markdown = result.document.export_to_markdown()
        timing = result.timings["pipeline_total"].times[0]
        rows.append({"document_id": document_id, "content": markdown})
        logging.info(f"Finalizado: {filename} em {timing:.2f} segundos")
    except Exception as e:
        logging.error(f"Erro ao processar {filename}: {e}")

# Escrita no CSV
try:
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["document_id", "content"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logging.info(f"Exportação completa: {output_csv}")
except Exception as e:
    logging.error(f"Erro ao salvar CSV: {e}")

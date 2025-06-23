import random
import traceback
from langchain_google_vertexai import VertexAI
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import OutputParserException
from langchain.output_parsers import ResponseSchema
import pandas as pd
import json
import threading
from concurrent.futures import ThreadPoolExecutor

# 🔹 Configuração do modelo VertexAI
model = VertexAI(model_name="gemini-2.0-flash")

# 🔹 Leitura do CSV e preparação dos dados
arquivo_csv = "your-file"
df = pd.read_csv(arquivo_csv)[:500]
docs = df.to_dict(orient="records")

# Seleção da amostra
# if len(docs) >= 200:
#     amostra = random.sample(docs, 38)
# else:
#     raise ValueError("A lista não contém elementos suficientes.")

# Preparação dos documentos
lista = []
for row in docs:
    conteudo = f"""
    # Dados do Processo Judicial
    Número do Processo: {row['numero_processo']}
    Descrição do Pedido: {row['descricao_pedido']}
    Número do Registro: {row['numero_registro']}
    Documento: {row['numero_doc_atual']}/{row['numero_doc_total']}
    Identificação: {row['identificacao']}
    Relator: {row['relator']}
    Data de Julgamento: {row['data_julgamento']}
    Órgão Julgador: {row['orgao_julgador']}
    
    # Ementa:
    {row['ementa']}
    
    # Acordão:
    {row['acordao_text']}
    
    # Tese:
    {row['tese']}
    """.strip()
    
    lista.append({
        "content": conteudo,
        # "doc_id": f"proc_{row['numero_processo']}" if pd.notna(row['numero_processo']) else "null"
    })

# 🔹 Schema de resposta para pares de queries
response_schemas = [
    ResponseSchema(
        name="query_principal", 
        description="Primeira pergunta jurídica específica e prática baseada no caso"
    ),
    ResponseSchema(
        name="query_alternativa", 
        description="Segunda pergunta jurídica similar à primeira, mas com enfoque ou formulação diferente"
    ),
    ResponseSchema(
        name="resposta_consolidada", 
        description="Resposta única que atende ambas as perguntas, baseada na jurisprudência"
    ),
    ResponseSchema(
        name="trecho_referencia", 
        description="Trecho específico do documento que fundamenta a resposta"
    ),
    ResponseSchema(
        name="area_juridica", 
        description="Área do direito envolvida (ex: Civil, Penal, Trabalhista, etc.)"
    ),
    ResponseSchema(
        name="palavras_chave", 
        description="Lista de 3-5 palavras-chave jurídicas relevantes"
    )
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = parser.get_format_instructions()

# 🔹 Prompt profissional para geração de queries jurídicas
prompt_template = PromptTemplate(
    template="""
Você é um especialista em jurisprudência brasileira. Com base no processo judicial fornecido, 
gere um PAR de perguntas jurídicas realistas que um advogado, juiz ou estudante de direito 
faria sobre este caso.

DIRETRIZES IMPORTANTES:
1. As perguntas devem ser SIMILARES mas NÃO IDÊNTICAS
2. Devem refletir situações práticas do cotidiano jurídico
3. Evite perguntas óbvias ou que não seriam feitas na prática
4. Foque nos aspectos mais relevantes da decisão
5. Uma pergunta pode ser mais específica e outra mais geral
6. Considere diferentes perspectivas (autor, réu, tribunal)

TIPOS DE PERGUNTAS VÁLIDAS:
- Interpretação de dispositivos legais
- Aplicação de precedentes
- Requisitos para determinado direito
- Consequências jurídicas de atos
- Procedimentos processuais
- Critérios para decisões
- Elementos caracterizadores

EVITE PERGUNTAS COMO:
- "Qual foi a decisão?" (muito óbvia)
- "Quem ganhou o processo?" (simplista demais)
- "Quando foi julgado?" (informação básica)

DOCUMENTO JUDICIAL:
{jurisprudencia}

INSTRUÇÕES DE FORMATO:
{format_instructions}

Gere o par de queries considerando que elas serão usadas para treinar um sistema 
de busca jurídica inteligente.
    """,
    input_variables=["jurisprudencia"],
    partial_variables={"format_instructions": format_instructions}
)

# 🔹 Controle de concorrência e buffer
lock = threading.Lock()
output_list = []
processed_count = 0

# 🔹 Função para processar cada documento
def process_document(idx, documento):
    global processed_count
    
    try:
        # Formatação do prompt
        formatted_prompt = prompt_template.format(
            jurisprudencia=documento["content"]
        )
        
        # Invocação do modelo
        response = model.invoke(formatted_prompt)
        
        # Parse da resposta
        parsed_output = parser.parse(response)
        # parsed_output["doc_id"] = documento["doc_id"]
        parsed_output["documento_index"] = idx
        
        # Thread-safe addition ao buffer
        with lock:
            output_list.append(parsed_output)
            processed_count += 1
            
        print(f"✔ Documento {idx + 1}/{len(lista)} processado. Par de queries gerado.")
        
    except OutputParserException as e:
        print(f"❌ Erro de parsing no documento {idx}: {str(e)[:100]}...")
        # Tenta recuperar parcialmente
        try:
            with lock:
                output_list.append({
                    "doc_id": documento["doc_id"],
                    "documento_index": idx,
                    "erro": "Falha no parsing",
                    "resposta_bruta": str(response)[:500] if 'response' in locals() else "N/A"
                })
        except:
            pass
            
    except Exception as e:
        print(f"❌ Erro geral no documento {idx}: {str(e)}")

# 🔹 Processamento paralelo com controle de erro
print(f"🚀 Iniciando processamento de {len(lista)} documentos...")

try:
    with ThreadPoolExecutor(max_workers=3) as executor:  # Reduzido para evitar rate limiting
        futures = [
            executor.submit(process_document, idx, documento) 
            for idx, documento in enumerate(lista)
        ]
        
        # Aguarda conclusão de todas as threads
        for future in futures:
            try:
                future.result(timeout=120)  # Timeout de 2 minutos por documento
            except Exception as e:
                print(f"❌ Timeout ou erro na thread: {e}")

except Exception as e:
    print(f"❌ Erro crítico no processamento paralelo: {e}")
    traceback.print_exc()

# 🔹 Salvamento dos resultados
output_filename = "queries_juridicas_pares.json"

try:
    with open(output_filename, "w", encoding="utf-8") as json_file:
        json.dump(output_list, json_file, ensure_ascii=False, indent=2)
    
    print(f"✅ Processamento concluído!")
    print(f"📊 Total processado: {processed_count}/{len(lista)} documentos")
    print(f"💾 Resultados salvos em: {output_filename}")
    
    # Estatísticas básicas
    if output_list:
        successful = len([item for item in output_list if "erro" not in item])
        print(f"📈 Sucessos: {successful}, Erros: {len(output_list) - successful}")
        
        # Amostra do resultado
        if successful > 0:
            print("\n🔍 Exemplo de resultado gerado:")
            for item in output_list:
                if "erro" not in item:
                    print(f"Query 1: {item.get('query_principal', 'N/A')[:80]}...")
                    print(f"Query 2: {item.get('query_alternativa', 'N/A')[:80]}...")
                    print(f"Área: {item.get('area_juridica', 'N/A')}")
                    break
    
except Exception as e:
    print(f"❌ Erro ao salvar arquivo: {e}")
    # Backup em caso de erro
    try:
        with open("backup_queries.json", "w", encoding="utf-8") as backup_file:
            json.dump(output_list, backup_file, ensure_ascii=False)
        print("💾 Backup salvo como 'backup_queries.json'")
    except:
        print("❌ Falha também no backup")

print("🏁 Execução finalizada.")
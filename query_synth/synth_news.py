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
arquivo_csv = "your-file"  # Substitua pelo caminho do seu arquivo de notícias
df = pd.read_csv(arquivo_csv)[:500]  # Ajuste a quantidade conforme necessário
docs = df.to_dict(orient="records")

# Preparação dos documentos
lista = []
for row in docs:
    conteudo = f"""
    # Dados da Notícia
    Título: {row['title']}
    Data: {row['date']}
    Categoria: {row['category']}
    Subcategoria: {row['subcategory']}
    
    # Conteúdo da Notícia:
    {row['text']}
    """.strip()
    
    lista.append({
        "content": conteudo,
        "title": row['title'],
        "category": row['category'],
        "subcategory": row['subcategory']
    })

# 🔹 Schema de resposta para queries de notícias
response_schemas = [
    ResponseSchema(
        name="query_principal", 
        description="Pergunta específica e prática baseada no conteúdo da notícia que um leitor interessado faria"
    ),
    ResponseSchema(
        name="resposta_consolidada", 
        description="Resposta clara e objetiva baseada no conteúdo da notícia"
    ),
    ResponseSchema(
        name="trecho_referencia", 
        description="Trecho específico da notícia que fundamenta a resposta"
    ),
    ResponseSchema(
        name="categoria_tematica", 
        description="Tema principal da notícia (ex: Política, Economia, Saúde, Tecnologia, etc.)"
    ),
    ResponseSchema(
        name="palavras_chave", 
        description="Lista de 3-5 palavras-chave relevantes da notícia"
    )
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = parser.get_format_instructions()

# 🔹 Prompt profissional para geração de queries de notícias
prompt_template = PromptTemplate(
    template="""
Você é um especialista em análise de notícias e informação jornalística. Com base na notícia fornecida, 
gere uma pergunta realista que um leitor interessado no assunto faria sobre esta notícia.

DIRETRIZES IMPORTANTES:
1. A pergunta deve ser ESPECÍFICA e PRÁTICA
2. Deve refletir curiosidades genuínas que leitores teriam
3. Evite perguntas óbvias que já estão claramente respondidas no título
4. Foque nos aspectos mais relevantes e interessantes da notícia
5. Considere diferentes perspectivas (causas, consequências, contexto, detalhes)

TIPOS DE PERGUNTAS VÁLIDAS:
- Contextualização de eventos
- Consequências ou impactos
- Causas ou motivações
- Detalhes específicos mencionados
- Relações com outros eventos
- Implicações futuras
- Aspectos técnicos explicados
- Dados e estatísticas mencionados

EVITE PERGUNTAS COMO:
- "Qual é o título da notícia?" (muito óbvio)
- "Quando aconteceu?" (se está claro no texto)
- "O que aconteceu?" (muito genérico)

NOTÍCIA:
{noticia}

INSTRUÇÕES DE FORMATO:
{format_instructions}

Gere a query considerando que ela será usada para treinar um sistema 
de busca de notícias inteligente.
    """,
    input_variables=["noticia"],
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
            noticia=documento["content"]
        )
        
        # Invocação do modelo
        response = model.invoke(formatted_prompt)
        
        # Parse da resposta
        parsed_output = parser.parse(response)
        parsed_output["documento_index"] = idx
        parsed_output["titulo_noticia"] = documento["title"]
        parsed_output["categoria_original"] = documento["category"]
        parsed_output["subcategoria_original"] = documento["subcategory"]
        
        # Thread-safe addition ao buffer
        with lock:
            output_list.append(parsed_output)
            processed_count += 1
            
        print(f"✔ Notícia {idx + 1}/{len(lista)} processada. Query gerada.")
        
    except OutputParserException as e:
        print(f"❌ Erro de parsing na notícia {idx}: {str(e)[:100]}...")
        # Tenta recuperar parcialmente
        try:
            with lock:
                output_list.append({
                    "documento_index": idx,
                    "titulo_noticia": documento["title"],
                    "erro": "Falha no parsing",
                    "resposta_bruta": str(response)[:500] if 'response' in locals() else "N/A"
                })
        except:
            pass
            
    except Exception as e:
        print(f"❌ Erro geral na notícia {idx}: {str(e)}")

# 🔹 Processamento paralelo com controle de erro
print(f"🚀 Iniciando processamento de {len(lista)} notícias...")

try:
    with ThreadPoolExecutor(max_workers=10) as executor:  # Reduzido para evitar rate limiting
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
output_filename = "queries_noticias.json"

try:
    with open(output_filename, "w", encoding="utf-8") as json_file:
        json.dump(output_list, json_file, ensure_ascii=False, indent=2)
    
    print(f"✅ Processamento concluído!")
    print(f"📊 Total processado: {processed_count}/{len(lista)} notícias")
    print(f"💾 Resultados salvos em: {output_filename}")
    
    # Estatísticas básicas
    if output_list:
        successful = len([item for item in output_list if "erro" not in item])
        print(f"📈 Sucessos: {successful}, Erros: {len(output_list) - successful}")
        
        # Estatísticas por categoria
        categorias = {}
        for item in output_list:
            if "erro" not in item and "categoria_original" in item:
                cat = item["categoria_original"]
                categorias[cat] = categorias.get(cat, 0) + 1
        
        if categorias:
            print("\n📊 Distribuição por categoria:")
            for cat, count in sorted(categorias.items(), key=lambda x: x[1], reverse=True):
                print(f"  {cat}: {count} notícias")
        
        # Amostra do resultado
        if successful > 0:
            print("\n🔍 Exemplo de resultado gerado:")
            for item in output_list:
                if "erro" not in item:
                    print(f"Título: {item.get('titulo_noticia', 'N/A')[:60]}...")
                    print(f"Query: {item.get('query_principal', 'N/A')[:80]}...")
                    print(f"Categoria: {item.get('categoria_tematica', 'N/A')}")
                    break
    
except Exception as e:
    print(f"❌ Erro ao salvar arquivo: {e}")
    # Backup em caso de erro
    try:
        with open("backup_queries_noticias.json", "w", encoding="utf-8") as backup_file:
            json.dump(output_list, backup_file, ensure_ascii=False)
        print("💾 Backup salvo como 'backup_queries_noticias.json'")
    except:
        print("❌ Falha também no backup")

print("🏁 Execução finalizada.")
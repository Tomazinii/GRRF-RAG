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
arquivo_csv = "your-file-name"  # Substitua pelo caminho do seu arquivo de auditoria
df = pd.read_csv(arquivo_csv, quotechar='"', doublequote=True)[:100]  # Ajuste a quantidade conforme necessário
docs = df.to_dict(orient="records")

print("len", len(df))
# Preparação dos documentos
lista = []
for row in docs:
    # Usando apenas os campos disponíveis: document_id e content
    conteudo = f"""
    # Documento de Auditoria - TCM Goiás
    ID do Documento: {row['document_id']}
    
    # Conteúdo do Relatório de Auditoria:
    {row['content']}
    """.strip()
    
    lista.append({
        "content": conteudo,
        "document_id": row['document_id'],
        "original_content": row['content']  # Mantém o conteúdo original para referência
    })

# 🔹 Schema de resposta para queries de auditoria
response_schemas = [
    ResponseSchema(
        name="query_principal", 
        description="Pergunta específica sobre controle interno, conformidade ou achados de auditoria que um auditor, gestor público ou controlador faria"
    ),
    ResponseSchema(
        name="resposta_consolidada", 
        description="Resposta técnica baseada nos achados, recomendações e constatações da auditoria"
    ),
    ResponseSchema(
        name="trecho_referencia", 
        description="OBRIGATÓRIO: Trecho ESPECÍFICO e LITERAL extraído do conteúdo do documento que fundamenta diretamente a resposta. NUNCA deixe vazio. Deve conter pelo menos 20 palavras do texto original."
    ),
    ResponseSchema(
        name="area_controle", 
        description="Área de controle envolvida (ex: Financeiro, Licitações, Pessoal, Patrimônio, etc.)"
    ),
    ResponseSchema(
        name="tipo_irregularidade", 
        description="Tipo de irregularidade ou aspecto verificado (se aplicável)"
    ),
    ResponseSchema(
        name="palavras_chave", 
        description="Lista de 3-5 palavras-chave técnicas de auditoria e controle público"
    )
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = parser.get_format_instructions()

# 🔹 Prompt profissional para geração de queries de auditoria
prompt_template = PromptTemplate(
    template="""
Você é um especialista em auditoria governamental e controle interno do setor público brasileiro, 
com foco no Tribunal de Contas dos Municípios de Goiás (TCM-GO). Com base no relatório de auditoria fornecido, 
gere uma pergunta técnica realista que um auditor, controlador interno, gestor público ou membro do TCM faria 
sobre este processo de auditoria.

⚠️ REGRA CRÍTICA: O campo "trecho_referencia" é OBRIGATÓRIO e NUNCA pode estar vazio, N/A ou similar. 
SEMPRE extraia um trecho literal específico do documento fornecido que fundamente sua resposta.

DIRETRIZES IMPORTANTES:
1. A pergunta deve ser TÉCNICA e ESPECÍFICA ao contexto de controle público
2. Deve refletir preocupações reais de auditoria e compliance
3. Foque em aspectos de conformidade, eficiência, eficácia ou legalidade
4. SEMPRE identifique um trecho específico do texto para referenciar
5. O trecho de referência deve ter pelo menos 20 palavras do documento original

TIPOS DE PERGUNTAS VÁLIDAS:
- Conformidade com normas e legislação
- Eficiência de controles internos
- Adequação de procedimentos licitatórios
- Gestão de recursos públicos
- Cumprimento de recomendações anteriores
- Riscos identificados e mitigação
- Aspectos de governança pública
- Indicadores de desempenho
- Procedimentos de prestação de contas
- Responsabilização de gestores

EVITE PERGUNTAS COMO:
- "Qual foi o resultado da auditoria?" (muito genérico)
- "Quando foi realizada?" (informação básica)
- "Quem é o responsável?" (dado factual simples)

CONTEXTO ESPECÍFICO TCM-GO:
- Foque em aspectos municipais de Goiás
- Considere legislação específica de controle externo
- Enfatize aspectos de fiscalização municipal
- Considere impactos na gestão pública local

IMPORTANTE SOBRE TRECHO_REFERENCIA:
- SEMPRE copie literalmente um trecho do documento fornecido
- O trecho deve ser relevante para a pergunta e resposta
- Nunca invente ou parafraseie o trecho
- Se o documento for muito longo, escolha a parte mais relevante
- Mínimo de 20 palavras, máximo de 100 palavras

RELATÓRIO DE AUDITORIA:
{auditoria}

INSTRUÇÕES DE FORMATO:
{format_instructions}

Gere a query considerando que ela será usada para treinar um sistema 
de busca especializado em auditoria e controle público.

LEMBRE-SE: O trecho_referencia NUNCA pode estar vazio ou N/A!
    """,
    input_variables=["auditoria"],
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
            auditoria=documento["content"]
        )
        
        # Invocação do modelo
        response = model.invoke(formatted_prompt)
        
        # Parse da resposta
        parsed_output = parser.parse(response)
        
        # Validação crítica: trecho_referencia não pode estar vazio
        if not parsed_output.get("trecho_referencia") or parsed_output.get("trecho_referencia").strip() in ["", "N/A", "null", "None"]:
            # Tenta extrair um trecho do conteúdo original
            content_lines = documento["original_content"].split('\n')
            # Pega as primeiras linhas significativas (não vazias)
            meaningful_lines = [line.strip() for line in content_lines if line.strip() and len(line.strip()) > 10]
            if meaningful_lines:
                # Pega até 3 linhas ou até 100 palavras
                fallback_trecho = ' '.join(meaningful_lines[:3])[:500]
                parsed_output["trecho_referencia"] = fallback_trecho
                print(f"⚠️  Trecho de referência corrigido automaticamente para documento {idx}")
            else:
                # Se ainda assim não conseguir, usa o início do conteúdo
                parsed_output["trecho_referencia"] = documento["original_content"][:200] + "..."
                print(f"⚠️  Trecho de referência extraído do início do documento {idx}")
        
        parsed_output["documento_index"] = idx
        parsed_output["document_id"] = documento["document_id"]
        
        # Thread-safe addition ao buffer
        with lock:
            output_list.append(parsed_output)
            processed_count += 1
            
        print(f"✔ Auditoria {idx + 1}/{len(lista)} processada. Query gerada.")
        
    except OutputParserException as e:
        print(f"❌ Erro de parsing na auditoria {idx}: {str(e)[:100]}...")
        # Tenta recuperar parcialmente
        try:
            with lock:
                output_list.append({
                    "documento_index": idx,
                    "document_id": documento["document_id"],
                    "erro": "Falha no parsing",
                    "resposta_bruta": str(response)[:500] if 'response' in locals() else "N/A"
                })
        except:
            pass
            
    except Exception as e:
        print(f"❌ Erro geral na auditoria {idx}: {str(e)}")

# 🔹 Processamento paralelo com controle de erro
print(f"🚀 Iniciando processamento de {len(lista)} relatórios de auditoria...")

try:
    with ThreadPoolExecutor(max_workers=8) as executor:  # Reduzido para documentos mais complexos
        futures = [
            executor.submit(process_document, idx, documento) 
            for idx, documento in enumerate(lista)
        ]
        
        # Aguarda conclusão de todas as threads
        for future in futures:
            try:
                future.result(timeout=150)  # Timeout maior para análises complexas
            except Exception as e:
                print(f"❌ Timeout ou erro na thread: {e}")

except Exception as e:
    print(f"❌ Erro crítico no processamento paralelo: {e}")
    traceback.print_exc()

# 🔹 Salvamento dos resultados
output_filename = "queries_auditoria_tcm_goias_8.json"

try:
    with open(output_filename, "w", encoding="utf-8") as json_file:
        json.dump(output_list, json_file, ensure_ascii=False, indent=2)
    
    print(f"✅ Processamento concluído!")
    print(f"📊 Total processado: {processed_count}/{len(lista)} auditorias")
    print(f"💾 Resultados salvos em: {output_filename}")
    
    # Estatísticas básicas
    if output_list:
        successful = len([item for item in output_list if "erro" not in item])
        print(f"📈 Sucessos: {successful}, Erros: {len(output_list) - successful}")
        
        # Estatísticas por área de controle
        areas_controle = {}
        trechos_vazios = 0
        
        for item in output_list:
            if "erro" not in item:
                # Área de controle
                area = item.get("area_controle", "N/A")
                areas_controle[area] = areas_controle.get(area, 0) + 1
                
                # Verificação de trechos vazios
                trecho = item.get("trecho_referencia", "")
                if not trecho or trecho.strip() in ["", "N/A", "null", "None"]:
                    trechos_vazios += 1
        
        if areas_controle:
            print("\n🎯 Distribuição por área de controle:")
            for area, count in sorted(areas_controle.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {area}: {count} ocorrências")
        
        # Alerta sobre trechos vazios
        if trechos_vazios > 0:
            print(f"\n⚠️  ATENÇÃO: {trechos_vazios} documentos ainda com trecho_referencia vazio!")
        else:
            print(f"\n✅ Todos os {successful} documentos têm trecho_referencia preenchido!")
        
        # Amostra do resultado
        if successful > 0:
            print("\n🔍 Exemplo de resultado gerado:")
            for item in output_list:
                if "erro" not in item:
                    print(f"Document ID: {item.get('document_id', 'N/A')}")
                    print(f"Query: {item.get('query_principal', 'N/A')[:80]}...")
                    print(f"Área: {item.get('area_controle', 'N/A')}")
                    print(f"Trecho (primeiras 60 chars): {item.get('trecho_referencia', 'N/A')[:60]}...")
                    break
    
except Exception as e:
    print(f"❌ Erro ao salvar arquivo: {e}")
    # Backup em caso de erro
    try:
        with open("backup_queries_tcm_goias.json", "w", encoding="utf-8") as backup_file:
            json.dump(output_list, backup_file, ensure_ascii=False)
        print("💾 Backup salvo como 'backup_queries_tcm_goias.json'")
    except:
        print("❌ Falha também no backup")

print("🏁 Execução finalizada.")

# 🔹 Validação final de qualidade
print("\n" + "="*60)
print("🔍 VALIDAÇÃO FINAL DE QUALIDADE")
print("="*60)

if output_list:
    total_items = len([item for item in output_list if "erro" not in item])
    items_sem_trecho = len([item for item in output_list if "erro" not in item and 
                           (not item.get("trecho_referencia") or 
                            item.get("trecho_referencia", "").strip() in ["", "N/A", "null", "None"])])
    
    print(f"📊 Total de documentos processados com sucesso: {total_items}")
    print(f"⚠️  Documentos com trecho_referencia vazio: {items_sem_trecho}")
    print(f"✅ Taxa de sucesso com trecho preenchido: {((total_items - items_sem_trecho) / total_items * 100):.1f}%" if total_items > 0 else "N/A")
    
    if items_sem_trecho == 0:
        print("🎉 PERFEITO! Todos os documentos têm trecho_referencia preenchido!")
    else:
        print("❌ ATENÇÃO: Alguns trechos ainda estão vazios. Verifique a qualidade dos dados de entrada.")

print("="*60)
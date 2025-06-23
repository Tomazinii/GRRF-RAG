"""
Sistema de Avaliação de Estratégias de Recuperação de Documentos Jurídicos

Este módulo implementa um sistema para avaliar GRRF
de documentos jurídicos usando embeddings, reranking e métricas de avaliação.
"""

import getpass
import os



from datetime import datetime
import json
import logging
import sys
import threading

from langchain_huggingface import HuggingFaceEmbeddings

from collections import defaultdict
import time
from typing import Dict, List, Optional, Any
from uuid import uuid4
import numpy as np

from FlagEmbedding import FlagReranker
# from langchain_google_vertexai import VertexAIEmbeddings

from app_pgvector import PgvectoryVectorStore
from evaluator import evaluate_configuration

from tqdm import tqdm
# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class RetrievalEvaluationSystem:
    
    # Constantes de configuração
    CHUNK_METHODS = ["512", "1024", "2048", "markdown"]
    TOP_K_RETRIEVAL = 20
    POSITIVE_THRESHOLD = 0.0
    
    def __init__(self, vector_table_name):
        """Inicializa o sistema com todas as configurações necessárias."""
        self.VECTOR_TABLE_NAME = vector_table_name  # agora é por instância
        self.vector_store = self._initialize_vector_store()
        self.reranker = self._initialize_reranker()
        self.embeddings_model = self._initialize_embeddings_model()
        
        
        logger.info("Sistema de avaliação inicializado com sucesso")
    
    def _initialize_vector_store(self) -> PgvectoryVectorStore:
        """Inicializa o armazenamento de vetores PostgreSQL."""
        logger.info("Inicializando conexão com vector store")
        
        return PgvectoryVectorStore(
            db_host=os.getenv("HOST", "your-host-db"),
            db_port=os.getenv("PORT", 5433),
            db_user=os.getenv("USER", "myuser"),
            db_password=os.getenv("PASSWORD", "mypassword"),
            db_name=os.getenv("DATABASE", "mydatabase"),
        )
    
    def _initialize_reranker(self) -> FlagReranker:
        """Inicializa o modelo de reranking com detecção automática de GPU disponível."""
        logger.info("Inicializando modelo de reranking")

        try:
            import torch

            # Verifica GPUs visíveis no ambiente
            visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            logger.info(f"CUDA_VISIBLE_DEVICES: {visible_gpus or '[não definido]'}")

            if torch.cuda.is_available():
                # Dentro do processo, a primeira GPU visível sempre será cuda:0
                logger.info("Usando '7' (primeira GPU visível no ambiente)")
                device = "cuda:7"
            else:
                logger.warning("Nenhuma GPU visível. Usando CPU.")
                device = "cpu"

            return FlagReranker(
                'BAAI/bge-reranker-v2-m3',
                use_fp16=(device != "cpu"),
                devices=["cuda:7"],
                batch_size=4
            )
        except Exception as e:
            logger.error(f"Erro ao inicializar reranker: {e}. Forçando uso de CPU.")
            return FlagReranker(
                'BAAI/bge-reranker-v2-m3',
                use_fp16=False,
                devices=["cpu"],
                batch_size=4
            )
    
    def _initialize_embeddings_model(self) -> HuggingFaceEmbeddings:
        """Inicializa o modelo de embeddings do Google Vertex AI."""
        logger.info("Inicializando modelo de embeddings")
        
        return HuggingFaceEmbeddings(
                model_name="BAAI/bge-m3",
                model_kwargs={
                    "device": "cuda:7",
                },
            )
    
    def _retrieve_chunks_by_method(self, query_embedding: List[float], method: str) -> List[Dict]:
        """
        Recupera chunks usando um método específico de chunking.
        
        Args:
            query_embedding: Embedding da query
            method: Método de chunking a ser usado
            
        Returns:
            Lista de chunks recuperados
        """
        try:
            retrieved = self.vector_store.retriever(
                query_embedding=query_embedding,
                vector_table_name=self.VECTOR_TABLE_NAME,
                top_k=self.TOP_K_RETRIEVAL,
                filter=method,
            )
            
            # Adiciona informação do método usado
            for chunk in retrieved:
                chunk["chunk_method"] = method
                
            logger.debug(f"Recuperados {len(retrieved)} chunks usando método {method}")
            return retrieved
            
        except Exception as e:
            logger.error(f"Erro ao recuperar chunks com método {method}: {e}")
            return []
    
    def _apply_reranking(self, query_text: str, chunks: List[Dict]) -> List[Dict]:
        """
        Aplica reranking aos chunks recuperados.
        
        Args:
            query_text: Texto da query original
            chunks: Lista de chunks para reranking
            
        Returns:
            Chunks com scores de reranking adicionados
        """
        if not chunks:
            return chunks
            
        try:
            pairs = [[query_text, chunk["content"]] for chunk in chunks]
            scores = self.reranker.compute_score(pairs)
            
            # Adiciona scores aos chunks
            for i, chunk in enumerate(chunks):
                chunk["rerank_score"] = scores[i]
                
            logger.debug(f"Reranking aplicado a {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Erro durante reranking: {e}")
            # Retorna chunks sem score em caso de erro
            for chunk in chunks:
                chunk["rerank_score"] = 0.0
            return chunks
    
    def _select_best_method_by_positive_mean(self, chunks: List[Dict]) -> Optional[str]:
        """
        Seleciona o melhor método baseado na média dos scores positivos.
        
        Args:
            chunks: Lista de todos os chunks com scores
            
        Returns:
            Nome do melhor método ou None se nenhum método válido
        """
        method_scores = defaultdict(list)
        
        # Agrupa scores por método, considerando apenas scores positivos
        for chunk in chunks:
            if chunk["rerank_score"] > self.POSITIVE_THRESHOLD:
                method_scores[chunk["chunk_method"]].append(chunk["rerank_score"])
        
        if not method_scores:
            logger.warning("Nenhum método com scores positivos encontrado")
            return None
        
        # Seleciona método com maior média
        best_method = max(method_scores, key=lambda m: np.mean(method_scores[m]))
        best_mean = np.mean(method_scores[best_method])
        
        logger.debug(f"Melhor método: {best_method} (média: {best_mean:.4f})")
        return best_method
    
    
    
    def count_tokens(self, text: str, model: str = "cl100k_base") -> int:
        """
        Conta tokens de um texto com base no tokenizer real usado por modelos da OpenAI.

        Args:
            text: Texto de entrada
            model: Nome do tokenizer (default: cl100k_base)

        Returns:
            Número de tokens
        """
        import tiktoken
        
        encoding = tiktoken.get_encoding(model)
        return len(encoding.encode(text))
                
        
        
    def select_by_rrf_fusion(
        self,
        query_embed: List[float],
        query_text: str,
        eval_data: Dict,
        top_k: int = 10,
        k_rrf: int = 60
    ) -> Optional[Dict]:
        """
        Estratégia baseada em fusão de rankings (Reciprocal Rank Fusion).

        Args:
            query_embed: Embedding da query
            query_text: Texto da query
            eval_data: Dados para avaliação
            top_k: Número de chunks a selecionar
            k_rrf: Parâmetro de suavização para RRF

        Returns:
            Dicionário com os resultados da avaliação
        """
        from collections import defaultdict
        import time

        def reciprocal_rank_fusion(rankings, k=60):
            rrf_scores = defaultdict(float)
            for rank_list in rankings:
                for position, chunk_id in enumerate(rank_list):
                    rrf_scores[chunk_id] += 1 / (k + position + 1)
            return dict(sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True))

        logger.info(f"Executando estratégia 'rrf_fusion' para query: {query_text[:50]}...")
        start_time = time.time()

        # Passo 1: Recupera chunks por método
        all_chunks = []
        method_rankings = []
        chunk_id_map = {}
        unique_id_counter = 0

        for method in tqdm(self.CHUNK_METHODS, desc="Recuperando chunks", leave=False):
            chunks = self._retrieve_chunks_by_method(query_embed, method)
            logger.debug(f"Método {method}: {len(chunks)} chunks recuperados")
            if not chunks:
                continue

            # Aplica reranking
            reranked = self._apply_reranking(query_text, chunks)

            # Ordena por score
            reranked_sorted = sorted(reranked, key=lambda x: x["rerank_score"], reverse=True)

            # Atribui um ID único por chunk
            ranking_list = []
            for chunk in reranked_sorted:
                chunk_key = f"{method}_{unique_id_counter}"
                chunk["chunk_id"] = chunk_key
                chunk_id_map[chunk_key] = chunk
                ranking_list.append(chunk_key)
                unique_id_counter += 1

            all_chunks.extend(reranked_sorted)
            method_rankings.append(ranking_list)

        if not all_chunks:
            logger.warning("Nenhum chunk recuperado para RRF")
            return None

        # Passo 2: Aplica RRF
        fused_scores = reciprocal_rank_fusion(method_rankings, k=k_rrf)
        top_chunk_ids = list(fused_scores.keys())[:top_k]
        selected_chunks = [chunk_id_map[cid] for cid in top_chunk_ids]

        end_time = time.time()
        elapsed_time = round(end_time - start_time, 2)

        try:
            results = evaluate_configuration(
                query=eval_data["query_principal"],
                response=eval_data["resposta_consolidada"],
                reference_contexts=eval_data["trecho_referencia"],
                retrieved_contexts=selected_chunks
            )

            strategy_output = {
                "strategy": "select_by_rrf_fusion",
                "ragas": results["ragas"],
                "execution_metadata": {
                    "response_time_seconds": elapsed_time,
                },
                "total_chunks_evaluated": len(all_chunks),
                "strategy_characteristics": {
                    "fusion_type": "RRF",
                    "selected_chunks_count": len(selected_chunks),
                    "reranker": {
                        "chunks_rank": sorted(
                            [
                                {
                                    "chunk_id": chunk["chunk_id"],
                                    "chunk_method": chunk["chunk_method"],
                                    "score": round(chunk["rerank_score"], 2),
                                    "chunk_text": chunk["content"][:100]
                                }
                                for chunk in selected_chunks
                            ],
                            key=lambda x: x["score"],
                            reverse=True
                        )
                    }
                }
            }

            return strategy_output

        except Exception as e:
            logger.error(f"Erro durante avaliação com RRF: {e}")
            return None

    def naive(
        self,
        query_embed: List[float],
        query_text: str,
        eval_data: Dict
    ) -> Optional[Dict]:
        """
        Estratégia naive: recupera os chunks de cada método de chunking, sem reranking,
        e avalia diretamente com a métrica do evaluator.

        Args:
            query_embed: Embedding da query
            query_text: Texto da query
            eval_data: Dados contendo query, resposta, trechos de referência

        Returns:
            strategy_output com os resultados por método
        """
        import time
        from collections import defaultdict

        logger.info(f"Executando estratégia 'naive' para query: {query_text[:50]}...")

        method_results = []
        total_chunks = 0

        for method in tqdm(self.CHUNK_METHODS, desc="Recuperando chunks", leave=False):
            start_time = time.time()
            
            chunks = self._retrieve_chunks_by_method(query_embed, method)
            logger.debug(f"Método {method}: {len(chunks)} chunks recuperados")
            total_chunks += len(chunks)

            if not chunks:
                logger.warning(f"Nenhum chunk retornado para o método {method}")
                continue

            try:
                end_time = time.time()
                elapsed_time = round(end_time - start_time, 2)
                
                # Avaliação direta
                result = evaluate_configuration(
                    query=eval_data["query_principal"],
                    response=eval_data["resposta_consolidada"],
                    reference_contexts=eval_data["trecho_referencia"],
                    retrieved_contexts=chunks
                )

                method_results.append({
                    "chunking_method": method,
                    "execution_metadata": {
                        "response_time_seconds": elapsed_time,
                    },
                    "ragas": result["ragas"],
                    "retrieved_chunks_count": len(chunks),
                    "chunks_preview": [
                        {
                            "chunk_text": chunk["content"][:100],
                            "chunk_id": i
                        }
                        for i, chunk in enumerate(chunks[:5])  # limita a 5 para visualizar
                    ]
                })

            except Exception as e:
                logger.error(f"Erro durante avaliação naive para método {method}: {e}")
                continue


        if not method_results:
            logger.warning("Nenhum resultado gerado na estratégia naive")
            return None

        # Output final
        strategy_output = {
            "strategy": "naive",
            "total_chunks_evaluated": total_chunks,
            "strategy_characteristics": {
                "per_chunking_method_results": method_results
            }
        }

        return strategy_output
    
    
    
    def _load_query_dataset(self, file_path: str) -> List[Dict]:
        """
        Carrega dataset de queries do arquivo JSON.
        
        Args:
            file_path: Caminho para o arquivo de queries
            
        Returns:
            Lista de dados de queries
        """
        try:
            logger.info(f"Carregando dataset de queries: {file_path}")
            
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            query_data = data.get("data", [])
            logger.info(f"Carregadas {len(query_data)} queries")
            
            return query_data
            
        except FileNotFoundError:
            logger.error(f"Arquivo não encontrado: {file_path}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Erro ao decodificar JSON: {e}")
            return []
        except Exception as e:
            logger.error(f"Erro inesperado ao carregar dataset: {e}")
            return []
    
    def _save_results(self, results: List[Dict], output_prefix: str = "jurisprudencia") -> str:
        """
        Salva resultados em arquivo JSON.
        
        Args:
            results: Lista de resultados para salvar
            output_prefix: Prefixo do nome do arquivo
            
        Returns:
            Nome do arquivo gerado
        """
        try:
            final_output = {"results": results}
            file_id = str(uuid4())
            filename = f"{output_prefix}_{file_id}.json"
            
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(final_output, f, ensure_ascii=False, indent=4)
                
            logger.info(f"Resultados salvos em: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Erro ao salvar resultados: {e}")
            return ""
        
        
        
    def process_single_query(self, query_data, idx, total_queries):
        """
        Versão mais verbosa do process_single_query com logs detalhados
        """
        query = query_data["query_principal"]
        
        # Log de início
        print(f"\n[{idx}/{total_queries}] Iniciando: {query[:50]}...")
        
        # Gera embedding
        try:
            print(f"[{idx}] Gerando embedding...")
            query_embed = self.embeddings_model.embed_query(query)
            print(f"[{idx}] ✅ Embedding gerado")
        except Exception as e:
            print(f"[{idx}] ❌ Erro no embedding: {e}")
            return None
        
        # Estratégias disponíveis
        strategies = {
            "naive": self.naive,
            "select_by_rrf_fusion": self.select_by_rrf_fusion,
            "select_by_density": self.select_by_density,
            "select_weighted_voting": self.select_weighted_voting,
            "select_top_rerank_mmr": self.select_top_rerank_mmr,
            "positive_mean": self.select_by_positive_mean,
        }
        
        configs = []
        total_strategies = len(strategies)
        
        # Progress bar para estratégias
        strategy_pbar = tqdm(strategies.items(), 
                            desc=f"Query {idx} - Estratégias", 
                            leave=False,
                            position=1)
        
        for strategy_name, strategy_func in strategy_pbar:
            strategy_pbar.set_description(f"Query {idx} - {strategy_name}")
            
            try:
                print(f"[{idx}] Executando {strategy_name}...")
                result = strategy_func(
                    query_embed=query_embed,
                    query_text=query,
                    eval_data=query_data,
                )
                
                if result:
                    result["reranker_method"] = strategy_name
                    configs.append(result)
                    import gc
                    import torch
                    gc.collect()
                    torch.cuda.empty_cache()
                    print(f"[{idx}] ✅ {strategy_name} concluído")
                else:
                    print(f"[{idx}] ⚠️ {strategy_name} falhou")
                    
            except Exception as e:
                print(f"[{idx}] ❌ Erro em {strategy_name}: {e}")
                continue
        
        strategy_pbar.close()
        
     
        # Resultado final
        if configs:
            print(f"[{idx}] ✅ Query concluída com {len(configs)} estratégias")
            return {
                "query_text": query_data.get("query_principal"),
                "query_id": idx,
                "strategies": configs,
            }
        else:
            print(f"[{idx}] ❌ Nenhuma estratégia funcionou")
            return None
        
    def run_evaluation_parallel(self, query_file_path: str, max_workers: int = 3) -> str:
        """
        Execução paralela usando o padrão do código antigo que funcionava.
        """
        import concurrent.futures
        from functools import partial
        
        logger.info("Iniciando avaliação MULTIPROCESS (padrão código antigo)")
        
        # Carrega dataset
        query_dataset = self._load_query_dataset(query_file_path)
        if not query_dataset:
            logger.error("Falha ao carregar dataset. Abortando avaliação.")
            return ""
        
        total_queries = len(query_dataset)
        logger.info(f"Usando {max_workers} processos para {total_queries} queries")
        
        # Função parcial (igual ao código antigo)
        func = partial(process_single_query_global, self.VECTOR_TABLE_NAME)
        
        results = []
        
        # Progress bar principal
        main_pbar = tqdm(total=total_queries, desc="Processando queries", unit="query")
        
        try:
            # ProcessPoolExecutor (igual ao código antigo)
            with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
                # Submete todas as queries (igual ao código antigo)
                futures = [
                    executor.submit(func, data, idx, total_queries) 
                    for idx, data in enumerate(query_dataset, 1)
                ]
                
                # Processa resultados conforme completam (igual ao código antigo)
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                            main_pbar.set_postfix({
                                'Sucesso': len(results),
                                'Taxa': f"{len(results)/len(results)*100:.1f}%"
                            })
                    except Exception as e:
                        logger.error(f"Erro ao processar query: {e}")
                    finally:
                        main_pbar.update(1)
        
        finally:
            main_pbar.close()
        
        success_rate = len(results) / total_queries * 100
        logger.info(f"Avaliação multiprocess concluída! {len(results)}/{total_queries} queries processadas ({success_rate:.1f}% sucesso)")
        
        return self._save_results(results)



from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing

def execute_case(case_name, query_path, max_workers=4):
    """Executa um caso específico de avaliação com paralelização de queries"""
    logger.info(f"Iniciando caso: {case_name}")
    
    try:
        system = RetrievalEvaluationSystem(vector_table_name=case_name)
        # Use o método paralelo em vez do sequencial
        output_file = system.run_evaluation_parallel(query_path, max_workers=5)
        
        if output_file:
            logger.info(f"[{case_name}] Avaliação concluída! Resultados: {output_file}")
            return case_name, output_file, None
        else:
            logger.warning(f"[{case_name}] Nenhum resultado gerado.")
            return case_name, None, "Nenhum resultado gerado"
    except Exception as e:
        logger.error(f"Erro no caso '{case_name}': {e}")
        return case_name, None, str(e)





def process_single_query_global(vector_table_name, query_data, idx, total_queries):
    """
    Função GLOBAL para processar uma query (seguindo padrão do código antigo).
    Cria instância do sistema DENTRO do processo worker.
    """
    try:
        # Cria instância DENTRO do worker (igual ao código antigo)
        system = RetrievalEvaluationSystem(vector_table_name=vector_table_name)
        
        query = query_data["query_principal"]
        print(f"[Worker] Processando query {idx}/{total_queries}: {query[:50]}...")
        
        # Gera embedding
        query_embed = system.embeddings_model.embed_query(query)
        
        # Estratégias disponíveis
        strategies = {
            "naive": system.naive,
            "select_by_rrf_fusion": system.select_by_rrf_fusion,
        }
        
        configs = []
        
        # Executa cada estratégia
        for strategy_name, strategy_func in strategies.items():
            try:
                result = strategy_func(
                    query_embed=query_embed,
                    query_text=query,
                    eval_data=query_data,
                )
                
                if result:
                    result["reranker_method"] = strategy_name
                    configs.append(result)
                    
            except Exception as e:
                print(f"[Worker] Query {idx} - Erro na estratégia {strategy_name}: {e}")
                continue
        
        # Resultado final
        if configs:
            print(f"[Worker] ✅ Query {idx} concluída com {len(configs)} estratégias")
            return {
                "query_text": query_data.get("query_principal"),
                "query_id": idx,
                "strategies": configs,
            }
        else:
            print(f"[Worker] ❌ Query {idx} falhou - nenhuma estratégia funcionou")
            return None
            
    except Exception as e:
        print(f"[Worker] ❌ Erro fatal na query {idx}: {e}")
        return None



def main():
    """Executa os três casos em paralelo"""
    cases = {
        "case-name": "filepath-name.json",
    }

    logger.info("=== Iniciando execução paralela dos casos ===")
    
    # Usar ProcessPoolExecutor para verdadeiro paralelismo
    with ProcessPoolExecutor(max_workers=min(len(cases), multiprocessing.cpu_count())) as executor:
        # Submete todos os casos para execução
        future_to_case = {
        executor.submit(execute_case, case_name, query_path, max_workers=3): case_name 
        for case_name, query_path in cases.items()
    }
            
        # Processa resultados conforme completam
        for future in as_completed(future_to_case):
            case_name = future_to_case[future]
            try:
                case_name, output_file, error = future.result()
                if output_file:
                    logger.info(f"✅ [{case_name}] Concluído com sucesso: {output_file}")
                else:
                    logger.error(f"❌ [{case_name}] Falhou: {error}")
            except Exception as e:
                logger.error(f"❌ [{case_name}] Erro inesperado: {e}")
    
    logger.info("=== Execução paralela finalizada ===")

import multiprocessing as mp
if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
        main()
    
    except KeyboardInterrupt:
        print("Encerrado pelo usuário.")
        sys.exit(0)
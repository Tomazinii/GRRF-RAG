#!/usr/bin/env python3
"""
Sistema de Processamento de Chunks para Jurisprudência
Responsável por processar documentos jurídicos, criar chunks e indexá-los em vector store.
"""

import json
import os
import logging
from time import sleep
from typing import List, Dict, Any, Union
from uuid import uuid4
from dataclasses import dataclass
from abc import ABC, abstractmethod

from langchain_huggingface import HuggingFaceEmbeddings


from app_pgvector import PgvectoryVectorStore
import pandas as pd
from chunk_functions import markdown_chunker, sliding_window_chunker
from chunks import load_data_jurisprudencia, load_data_auditoria, load_data_noticias


# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ChunkConfig:
    """Configurações para processamento de chunks"""
    dataset_path: str
    vector_table_name: str
    batch_size: int = 50

class DatabaseConfig:
    """Configurações do banco de dados"""
    
    @staticmethod
    def get_vector_store() -> PgvectoryVectorStore:
        """Cria e retorna uma instância do vector store configurada"""
        return PgvectoryVectorStore(
            db_host=os.getenv("HOST", "your-host-db"),
            db_port=os.getenv("PORT", 5433),
            db_user=os.getenv("USER", "myuser"),
            db_password=os.getenv("PASSWORD", "mypassword"),
            db_name=os.getenv("DATABASE", "mydatabase"),
        )


class EmbeddingConfig:
    """Configurações do modelo de embedding"""
    
    @staticmethod
    def get_model(gpu_device: str) -> HuggingFaceEmbeddings:
        return HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={
                "device": gpu_device  
            },
        )


class BaseChunker(ABC):
    """Classe base abstrata para estratégias de chunking"""
    
    def __init__(self, dataset_path: str):
        """
        Inicializa o chunker com o caminho do dataset
        
        Args:
            dataset_path: Caminho para o arquivo de dados
        """
        self.dataset_path = dataset_path
        logger.info(f"Inicializando chunker com dataset: {dataset_path}")
    
    @abstractmethod
    def execute(self, chunk_size: Union[int, str]) -> List[Dict[str, Any]]:
        """
        Executa o processo de chunking
        
        Args:
            chunk_size: Tamanho do chunk ou tipo de chunking
            
        Returns:
            Lista de chunks processados
        """
        pass
    
    def _load_documents(self) -> List[Dict[str, Any]]:
        """Carrega os documentos do dataset"""
        try:
            if self.dataset_path == "./jurisprudencia.csv":
                documents = load_data_jurisprudencia(self.dataset_path)
            elif self.dataset_path == "./auditoria.csv":
                documents = load_data_auditoria(self.dataset_path)
            elif self.dataset_path == "./noticias.csv":
                documents = load_data_noticias(self.dataset_path)
                
            logger.info(f"Carregados {len(documents)} documentos do dataset")
            return documents
        except Exception as e:
            logger.error(f"Erro ao carregar documentos: {e}")
            raise


class SlidingWindowChunker(BaseChunker):
    """Implementação de chunking com janela deslizante"""
    
    def execute(self, chunk_size: int) -> List[Dict[str, Any]]:
        """
        Executa chunking com janela deslizante
        
        Args:
            chunk_size: Tamanho do chunk em caracteres
            
        Returns:
            Lista de chunks processados
        """
        logger.info(f"Iniciando chunking com janela deslizante - tamanho: {chunk_size}")
        
        documents = self._load_documents()
        chunks = []
        
        for i, document in enumerate(documents):
            try:
                texts = sliding_window_chunker(
                    text=document["content"],
                    chunk_size=chunk_size,
                )
                
                document_chunks = [
                    {
                        "content": text,
                        "chunk_type": str(chunk_size),
                        "document_id": document["document_id"]
                    }
                    for text in texts
                ]
                
                chunks.extend(document_chunks)
                
                if (i + 1) % 50 == 0:  # Log a cada 50 documentos
                    logger.info(f"Processados {i + 1}/{len(documents)} documentos")
                    
            except Exception as e:
                logger.warning(f"Erro ao processar documento {document.get('document_id', 'unknown')}: {e}")
                continue
        
        logger.info(f"Chunking concluído: {len(chunks)} chunks gerados")
        return chunks


class MarkdownChunker(BaseChunker):
    """Implementação de chunking baseado em markdown"""
    
    def execute(self, chunk_size: str = "markdown") -> List[Dict[str, Any]]:
        """
        Executa chunking baseado em estrutura markdown
        
        Args:
            chunk_size: Tipo de chunking (sempre "markdown" para esta classe)
            
        Returns:
            Lista de chunks processados
        """
        logger.info("Iniciando chunking baseado em markdown")
        
        documents = self._load_documents()
        chunks = []
        
        for i, document in enumerate(documents):
            try:
                texts = markdown_chunker(text=document["content"])
                
                document_chunks = [
                    {
                        "content": text.page_content,
                        "chunk_type": chunk_size,
                        "document_id": document["document_id"]
                    }
                    for text in texts
                ]
                
                chunks.extend(document_chunks)
                
                if (i + 1) % 50 == 0:  # Log a cada 50 documentos
                    logger.info(f"Processados {i + 1}/{len(documents)} documentos")
                    
            except Exception as e:
                logger.warning(f"Erro ao processar documento {document.get('document_id', 'unknown')}: {e}")
                continue
        
        logger.info(f"Chunking markdown concluído: {len(chunks)} chunks gerados")
        return chunks


class ChunkerFactory:
    """Factory para criação de chunkers"""
    
    @staticmethod
    def create_chunker(chunk_method: Union[int, str], dataset_path: str) -> BaseChunker:
        """
        Cria o chunker apropriado baseado no método especificado
        
        Args:
            chunk_method: Método de chunking (int para sliding window, "markdown" para markdown)
            dataset_path: Caminho para o dataset
            
        Returns:
            Instância do chunker apropriado
        """
        if chunk_method == "markdown":
            return MarkdownChunker(dataset_path)
        elif isinstance(chunk_method, int):
            return SlidingWindowChunker(dataset_path)
        else:
            raise ValueError(f"Método de chunking não suportado: {chunk_method}")

import torch

class EmbeddingProcessor:
    """Responsável pelo processamento de embeddings"""
    def __init__(self, model: HuggingFaceEmbeddings):
        self.model = model
        logger.info("EmbeddingProcessor inicializado")

    def process_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 50) -> List[Dict[str, Any]]:
        logger.info(f"Iniciando processamento de embeddings para {len(chunks)} chunks")

        processed_chunks = []
        total_batches = (len(chunks) + batch_size - 1) // batch_size

        i = 0
        while i < len(chunks):
            current_batch_size = batch_size
            success = False
            while not success and current_batch_size >= 1:
                try:
                    batch_chunks = chunks[i:i + current_batch_size]
                    texts = [chunk["content"] for chunk in batch_chunks]

                    embeddings = self.model.embed_documents(texts=texts)

                    for chunk, embedding in zip(batch_chunks, embeddings):
                        chunk["embedding"] = embedding
                        processed_chunks.append(chunk)

                    torch.cuda.empty_cache()
                    batch_num = (i // batch_size) + 1
                    logger.info(f"Batch {batch_num}/{total_batches} processado ({len(batch_chunks)} chunks)")
                    i += current_batch_size
                    success = True

                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        logger.warning(
                            f"Erro de memória no batch iniciando em {i}. Reduzindo batch_size de {current_batch_size} para {current_batch_size // 2}"
                        )
                        torch.cuda.empty_cache()
                        current_batch_size = current_batch_size // 2
                    else:
                        logger.error(f"Erro inesperado ao processar batch: {e}")
                        break

        logger.info("Processamento de embeddings concluído")
        return [chunk for chunk in processed_chunks if "embedding" in chunk]


class VectorStoreIndexer:
    """Responsável pela indexação no vector store"""
    
    def __init__(self, vector_store: PgvectoryVectorStore):
        """
        Inicializa o indexador
        
        Args:
            vector_store: Instância do vector store configurada
        """
        self.vector_store = vector_store
        logger.info("VectorStoreIndexer inicializado")
    
    def index_chunks(self, chunks: List[Dict[str, Any]], table_name: str) -> None:
        """
        Indexa chunks no vector store
        
        Args:
            chunks: Lista de chunks com embeddings
            table_name: Nome da tabela para indexação
        """
        logger.info(f"Iniciando indexação de {len(chunks)} chunks na tabela '{table_name}'")
        
        try:
            self.vector_store.indexer(
                vector_table_name=table_name,
                documents_embedded=chunks,
            )
            logger.info("Indexação concluída com sucesso")
            
        except Exception as e:
            logger.error(f"Erro durante indexação: {e}")
            raise


class ChunkProcessingPipeline:
    """Pipeline principal para processamento de chunks"""
    
    def __init__(self, config: ChunkConfig,  gpu_device: str):
        """
        Inicializa o pipeline
        
        Args:
            config: Configurações do pipeline
        """
        self.config = config
        self.vector_store = DatabaseConfig.get_vector_store()
        self.embedding_model = EmbeddingConfig.get_model(gpu_device=gpu_device)
        self.embedding_processor = EmbeddingProcessor(self.embedding_model)
        self.indexer = VectorStoreIndexer(self.vector_store)
        logger.info(f"Pipeline inicializado com GPU {gpu_device}")
        logger.info("Pipeline inicializado com sucesso")
    
    def process_method(self, chunk_method: Union[int, str]) -> None:
        """
        Processa um método de chunking específico
        
        Args:
            chunk_method: Método de chunking a ser processado
        """
        logger.info(f"Iniciando processamento para método: {chunk_method}")
        
        try:
            # Cria chunker apropriado
            chunker = ChunkerFactory.create_chunker(chunk_method, self.config.dataset_path)
            
            # Executa chunking
            chunks = chunker.execute(chunk_method)
            
            if not chunks:
                logger.warning(f"Nenhum chunk gerado para método {chunk_method}")
                return
            
            # Processa embeddings
            embedded_chunks = self.embedding_processor.process_chunks(
                chunks, self.config.batch_size
            )
            
            # Indexa no vector store
            self.indexer.index_chunks(embedded_chunks, self.config.vector_table_name)
            
            logger.info(f"Método {chunk_method} processado com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao processar método {chunk_method}: {e}")
            raise
    
    def run_all_methods(self, methods: List[Union[int, str]]) -> None:
        """
        Executa o pipeline para todos os métodos especificados
        
        Args:
            methods: Lista de métodos de chunking para processar
        """
        logger.info(f"Iniciando pipeline para {len(methods)} métodos: {methods}")
        
        for i, method in enumerate(methods, 1):
            logger.info(f"Processando método {i}/{len(methods)}: {method}")
            
            try:
                self.process_method(method)
                logger.info(f"Método {method} concluído com sucesso")
                
            except Exception as e:
                logger.error(f"Falha no processamento do método {method}: {e}")
                # Continua com o próximo método
                continue
        
        logger.info("Pipeline concluído")



from multiprocessing import Process

def run_case(case_name, dataset_path, table_name, gpu_device, methods):
    logger.info(f"Iniciando processo para {case_name} na GPU {gpu_device}")
    config = ChunkConfig(dataset_path=dataset_path, vector_table_name=table_name, batch_size=50)
    pipeline = ChunkProcessingPipeline(config, gpu_device)
    pipeline.run_all_methods(methods)

def main():
    cases = {
        "case-name": ("./jurisprudencia.csv", "jurisprudencia"),
        "case-name": ("./noticias.csv", "noticias"),
        "case-name": ("./auditoria.csv", "auditoria"),
    }
    gpu_devices = ["cuda:1", "cuda:6"]
    methods = [512, 1024, 2048, "markdown"]

    processes = []
    for (case_name, (dataset_path, table_name)), gpu_device in zip(cases.items(), gpu_devices):
        p = Process(target=run_case, args=(case_name, dataset_path, table_name, gpu_device, methods))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    logger.info("Todos os cases foram processados")

if __name__ == "__main__":
    main()
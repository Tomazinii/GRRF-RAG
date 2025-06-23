import json
from typing import List
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.extras import execute_values





class PgvectoryVectorStore:
    
    def __init__(self ,db_name: str, db_port: int, db_user: str, db_password: str, db_host: str) -> None:
        self.db_name = db_name
        self.db_user = db_user
        self.db_password = db_password
        self.db_host = db_host
        self.db_port = db_port
        
    def _connect(self):
        """Estabelece a conexão com o banco de dados PostgreSQL."""
        return psycopg2.connect(
            dbname=self.db_name,
            user=self.db_user,
            password=self.db_password,
            host=self.db_host,
            port=self.db_port
        )
        
    def retriever(self, query_embedding: list, vector_table_name, similarity_threshold=0.3, top_k=10, filter=None):
        """
        Recupera conteúdos do banco de dados com base na similaridade do embedding.
        
        :param query_embedding: Vetor de consulta (embedding).
        :param colum_name: Nome da coluna de vetor no banco.
        :param similarity_threshold: Limiar mínimo de similaridade (float entre 0 e 1).
        :param top_k: Número máximo de correspondências desejadas.
        :param vector_table_name: Nome da tabela vetorial.
        :param filter: Valor opcional de chunk_type para filtrar os resultados.
        :return: Lista de dicionários contendo os conteúdos correspondentes.
        """

        filter_clause = ""
        filter_params = ()

        if filter:
            filter_clause = "AND chunk_type = %s"
            filter_params = (filter,)

        query = f"""
        WITH vector_matches AS (
            SELECT 
                content, document_id, chunk_type, embedding,
                1 - (embedding<=> %s::vector) AS similarity
            FROM {vector_table_name}
            WHERE 1 - (embedding <=> %s::vector) > %s
            {filter_clause}
            ORDER BY similarity DESC
            LIMIT %s
        )
        SELECT content, document_id, chunk_type, embedding, similarity FROM vector_matches
        """

        conn = None
        matches = []

        try:
            conn = self._connect()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Montar parâmetros dinamicamente
            params = (query_embedding, query_embedding, similarity_threshold) + filter_params + (top_k,)

            cursor.execute(query, params)
            results = cursor.fetchall()

            if not results:
                raise Exception("Did not find any results. Adjust the query parameters.")

            for r in results:
                chunk = {
                    "content": r.get("content"),
                    "document_id": r.get("document_id"),
                    "chunk_type": r.get("chunk_type"),
                    "embedding": json.loads(r.get("embedding")) if isinstance(r.get("embedding"), str) else r.get("embedding"),
                }
                matches.append(chunk)

        except Exception as e:
            print(f"Erro ao recuperar documentos: {e}")

        finally:
            if conn:
                cursor.close()
                conn.close()

        return matches

    

    
    def indexer(self, documents_embedded: list[dict], vector_table_name):
        
        insert_query = f"""
        INSERT INTO {vector_table_name} (
            content,
            document_id,
            chunk_type,
            embedding
            )
        VALUES %s
        """
    
        values = []
        if documents_embedded:
            for doc in documents_embedded:
                row = [
                    doc.get("content"),
                    doc.get("document_id"),
                    doc.get("chunk_type"),
                    doc.get("embedding")
                ]
                    
                values.append(tuple(row))
                
        try:
            conn = self._connect()
            
            cursor = conn.cursor()
            
            execute_values(cursor, insert_query, values)
            
            conn.commit()
            print(f"registros inseridos com sucesso na tabela {vector_table_name}.")
            cursor.close()
            conn.close()
            
            
        except Exception as e:
            print(f"Erro ao inserir documentos: {e}")
        finally:
            if conn:
                cursor.close()
                conn.close()

    def create_table(self, vector_table_name: str):
        """
        Cria uma nova tabela PostgreSQL para armazenar documentos e embeddings.
        """
        create_extension_query = "CREATE EXTENSION IF NOT EXISTS vector;"
        create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {vector_table_name} (
                content TEXT NOT NULL,
                document_id VARCHAR(255),
                chunk_type TEXT NOT NULL,
                embedding vector(1024) NOT NULL
            );
            """
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute(create_extension_query)
        cursor.execute(f"DROP TABLE IF EXISTS {vector_table_name}")
        
        cursor.execute(create_table_query)
        conn.commit()

        cursor.close()
        conn.close()
        
        print("Tabela criada com sucesso.")
        




import os
from app_pgvector import PgvectoryVectorStore


vector_store = PgvectoryVectorStore(
    db_host=os.getenv("HOST", "200.137.197.252"),
    db_port=os.getenv("PORT", 5433),
    db_user=os.getenv("USER", "myuser"),
    db_password=os.getenv("PASSWORD", "mypassword"),
    db_name=os.getenv("DATABASE", "mydatabase"),
)

tables = [ "jurisprudencia",
    "noticias",
    "auditoria"
    ]
for table in tables:
    print(f"Table: {table}")
    vector_store.create_table(
    vector_table_name=table
    )
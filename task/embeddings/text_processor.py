from enum import StrEnum
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor

from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.utils.text import chunk_text


class SearchMode(StrEnum):
    EUCLIDIAN_DISTANCE = "euclidean"  # Euclidean distance (<->)
    COSINE_DISTANCE = "cosine"  # Cosine distance (<=>)


class TextProcessor:
    """Processor for text documents that handles chunking, embedding, storing, and retrieval"""

    def __init__(self, embeddings_client: DialEmbeddingsClient, db_config: dict):
        self.embeddings_client = embeddings_client
        self.db_config = db_config

    def _get_connection(self):
        """Get database connection"""
        return psycopg2.connect(
            host=self.db_config['host'],
            port=self.db_config['port'],
            database=self.db_config['database'],
            user=self.db_config['user'],
            password=self.db_config['password']
        )

    #TODO:
    # provide method `process_text_file` that will:
    #   - apply file name, chunk size, overlap, dimensions and whether the table should be truncated
    #   - truncate table with vectors if needed
    #   - load content from file and generate chunks (in `utils.text` present `chunk_text` that will help do that)
    #   - generate embeddings from chunks
    #   - save (insert) embeddings and chunks to DB
    #       hint 1: embeddings should be saved as string list
    #       hint 2: embeddings string list should be cast to vector ({embeddings}::vector)
    def process_text_file(self,
                          file_name: str,
                          chunk_size: int,
                          overlap: int,
                          dimensions: int,
                          truncate_table: bool):
        conn = self._get_connection()
        cursor = conn.cursor()

        if truncate_table:
            cursor.execute("TRUNCATE TABLE vectors")

        file_path = Path(__file__).parent.parent / file_name
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            chunks = chunk_text(content, chunk_size, overlap)
            embeddings = self.embeddings_client.get_embeddings(chunks, dimensions=1536)
            for i in range(len(chunks)):
                vector_string = str(embeddings.get(i))
                cursor.execute("INSERT INTO vectors (document_name, text, embedding) VALUES (%s, %s, %s::vector)",
                    (file_name, chunks[i], vector_string))
            conn.commit()

        cursor.close()
        conn.close()



    #TODO:
    # provide method `search` that will:
    #   - apply search mode, user request, top k for search, min score threshold and dimensions
    #   - generate embeddings from user request
    #   - search in DB relevant context
    #     hint 1: to search it in DB you need to create just regular select query
    #     hint 2: Euclidean distance `<->`, Cosine distance `<=>`
    #     hint 3: You need to extract `text` from `vectors` table
    #     hint 4: You need to filter distance in WHERE clause
    #     hint 5: To get top k use `limit`

    def search(self,
               search_mode: SearchMode,
               user_request: str,
               top_k: int,
               score_threshold: float,
               dimensions: int) -> list[str]:

        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if score_threshold < 0 or score_threshold > 1:
            raise ValueError("score_threshold must be in [0.0..., 0.99...] range")

        embedding = self.embeddings_client.get_embeddings([user_request], dimensions=dimensions)
        vector_string = str(embedding[0])

        if search_mode == SearchMode.COSINE_DISTANCE:
            max_distance = 1.0 - score_threshold
        else:
            max_distance = float('inf') if score_threshold == 0 else (1.0 / score_threshold) - 1.0

        retrieved_chunks = []
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # TODO:
                #  1. Execute query:
                #       - query: Use `_get_search_query(search_mode)` method. Please take a look at query!
                #       - vars: (vector_string, vector_string, max_distance, top_k)
                #  2. Fetch all results with `cursor` into `results`
                cursor.execute(self._get_search_query(search_mode), (vector_string, vector_string, max_distance, top_k))

                results = cursor.fetchall()

                for row in results:
                    if search_mode == SearchMode.COSINE_DISTANCE:
                        similarity = 1.0 - row['distance']
                    else:
                        similarity = 1.0 / (1.0 + row['distance'])

                    print(f"---Similarity score: {similarity:.2f}---")
                    print(f"Data: {row['text']}\n")
                    retrieved_chunks.append(row['text'])

        return retrieved_chunks

    def _get_search_query(self, search_mode: SearchMode) -> str:
        return """SELECT text, embedding {mode} %s::vector AS distance
                  FROM vectors
                  WHERE embedding {mode} %s::vector <= %s
                  ORDER BY distance
                  LIMIT %s""".format(mode='<->' if search_mode == SearchMode.EUCLIDIAN_DISTANCE else '<=>')
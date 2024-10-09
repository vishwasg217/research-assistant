from vector_database import VectorDB
from engine import Engine

if __name__ == "__main__":
    vector_db = VectorDB(collection_name="Paper")
    engine = Engine(vector_db=vector_db)

    question = "How do GPT models compare to BERT models for classification tasks?"
    response = engine.query(question=question, level="intermediate")
    print(response)
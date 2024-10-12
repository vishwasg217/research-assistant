from typing import Literal
import weaviate
from weaviate.classes.query import MetadataQuery
from weaviate.collections.classes.filters import _Filters
import os
from dotenv import load_dotenv
from .pydantic_classes import Document


openai_key = os.getenv("OPENAI_API_KEY")
headers = {
    "X-OpenAI-Api-Key": openai_key,
}
client = weaviate.connect_to_local(
    headers=headers
)
response = client.collections.list_all(simple=False)

print(response['Paper'].properties)

client.close()
import datetime
import chromadb
import traceback
import pandas as pd
from chromadb.utils import embedding_functions

from model_configurations import get_model_configuration

gpt_emb_version = 'text-embedding-ada-002'
gpt_emb_config = get_model_configuration(gpt_emb_version)

dbpath = "./"

filename = 'COA_OpenData.csv'
df = pd.read_csv(filename)

chroma_client = chromadb.PersistentClient(path=dbpath)

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key = gpt_emb_config['api_key'],
    api_base = gpt_emb_config['api_base'],
    api_type = gpt_emb_config['openai_type'],
    api_version = gpt_emb_config['api_version'],
    deployment_id = gpt_emb_config['deployment_name']
)

collection = chroma_client.get_or_create_collection(
    name="TRAVEL",
    metadata={"hnsw:space": "cosine"},
    embedding_function=openai_ef
)

def generate_hw01():
    for index, row in df.iterrows():
        date_object = datetime.datetime.strptime(row["CreateDate"], "%Y-%m-%d")
        timestamp = int(date_object.timestamp())
        collection.add(
            ids=[str(row["ID"])],
            documents=row["HostWords"],
            metadatas=[{"file_name": filename, "name": row["Name"], "type": row["Type"], "address": row["Address"], "tel": row["Tel"], "city": row["City"], "town": row["Town"], "date": timestamp }]
        )
    return collection
    
def generate_hw02(question, city, store_type, start_date, end_date):
    start_date = int(start_date.timestamp())
    end_date = int(end_date.timestamp())
    result = collection.query(
        query_texts=[question],
        where={"$and": [
            {"city": {"$in": city}}, 
            {"type": {"$in": store_type}},
            {"date": {"$gte": start_date}},
            {"date": {"$lte": end_date}}
            ]},
        include=["metadatas", "distances"],
    )
    store_names = [ metadata['name'] for metadata, distance in zip(result['metadatas'][0], result['distances'][0]) if distance < 0.2]
    return store_names
    
def generate_hw03(question, store_name, new_store_name, city, store_type):
    results = collection.query(
        query_texts=[store_name],
        n_results=1 
    )
    new_metadata = results["metadatas"][0][0]
    new_metadata["new_store_name"] = new_store_name
    
    collection.delete(ids=[results["ids"][0][0]])
    collection.add(
        ids=[results["ids"][0][0]],  
        documents=results["documents"][0],
        metadatas=[new_metadata]  
    )
    result = collection.query(
        query_texts=[question],
        where={"$and": [
            {"city": {"$in": city}}, 
            {"type": {"$in": store_type}},
            ]},
        include=["metadatas", "distances"],
    )
    store_names = [metadata['new_store_name'] if metadata.get('new_store_name') else metadata['name'] for metadata, distance in zip(result['metadatas'][0], result['distances'][0]) if distance < 0.2]
    return store_names

    
def demo(question):
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key = gpt_emb_config['api_key'],
        api_base = gpt_emb_config['api_base'],
        api_type = gpt_emb_config['openai_type'],
        api_version = gpt_emb_config['api_version'],
        deployment_id = gpt_emb_config['deployment_name']
    )
    

    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )
    
    return collection

#print(generate_hw01())
#print(generate_hw02("我想要找有關茶餐點的店家", ["宜蘭縣", "新北市"], ["美食"], datetime.datetime(2024, 4, 1), datetime.datetime(2024, 5, 1)))
print(generate_hw03("我想要找南投縣的田媽媽餐廳，招牌是蕎麥麵","耄饕客棧","田媽媽（耄饕客棧）",["南投縣"],["美食"]))

import os, csv, glob
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from openai import OpenAI

# ==== Config ====
COLLECTION = "resumos_liz"
TXT_DIR = "data/txt"
META_CSV = "data/metadata.csv"  # opcional: se não existir, inferimos do nome do arquivo
EMBED_MODEL = "text-embedding-3-large"  # 3072 dims
TOP_K = 20

# ==== Setup ====
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Garante a collection
try:
    qdrant.get_collection(COLLECTION)
except:
    qdrant.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(
            size=3072, distance=models.Distance.COSINE
        ),
    )

# Carrega metadados
meta = {}
if os.path.exists(META_CSV):
    with open(META_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            meta[row["arquivo"]] = {"id": row["id"], "titulo": row["titulo"], "arquivo": row["arquivo"]}

# Lista arquivos txt
arquivos = sorted(glob.glob(os.path.join(TXT_DIR, "**", "*.txt"), recursive=True))

points = []
ids_counter = 0

def infer_from_filename(filename):
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1].replace("-", " ")
    return "", base.replace("-", " ")

for path in arquivos:
    nome = os.path.basename(path)
    with open(path, "r", encoding="utf-8") as f:
        texto = f.read().strip()

    if not texto:
        continue

    # Extrai id/titulo
    if nome in meta:
        id_doc = meta[nome]["id"]
        titulo_doc = meta[nome]["titulo"]
    else:
        id_doc, titulo_doc = infer_from_filename(nome)

    # Embedding
    emb = client.embeddings.create(model=EMBED_MODEL, input=texto).data[0].embedding

    # Monta ponto
    payload = {"id": id_doc, "titulo": titulo_doc, "arquivo": nome}
    points.append(
        models.PointStruct(
            id=ids_counter, vector=emb, payload=payload
        )
    )
    ids_counter += 1

    # Upsert em lotes para acelerar
    if len(points) >= 128:
        qdrant.upsert(collection_name=COLLECTION, points=points)
        points = []

# Envia o restante
if points:
    qdrant.upsert(collection_name=COLLECTION, points=points)

print(f"✅ Ingestão concluída. Vetores inseridos na collection '{COLLECTION}'.")

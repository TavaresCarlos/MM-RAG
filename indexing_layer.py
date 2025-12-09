import faiss
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from numpy.linalg import norm
import ollama
import os

class indexing_layer:
    def __init__(self, docs, model_name):
        self.docs = docs
        self.model_name = model_name
        self.model = CLIPModel.from_pretrained(self.model_name)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.index = None
        self.doc_info = []
        self.embeddings = None

    def normalize(self, x):
        return x / (norm(x, axis=1, keepdims=True) + 1e-10)

    def to_embedding(self):
        embeddings_list = []
        self.doc_info = []

        for indice, doc in enumerate(self.docs):
            if doc["embedding_tipo"] == "texto":
                texto = doc["conteudo"] if doc["tipo"] == "texto" else doc["descricao"]
                inputs = self.processor(text=[texto], return_tensors="pt", padding=True).to(self.device)
                with torch.no_grad():
                    emb = self.model.get_text_features(**inputs)
                emb = self.normalize(emb.cpu().numpy())
                embeddings_list.append(emb[0])
                self.doc_info.append({"idx": indice, "tipo": "texto", "conteudo": texto})
            else:
                if not os.path.exists(doc["conteudo"]):
                    print(f"⚠️ Imagem não encontrada: {doc['conteudo']}")
                    continue

                imagem = Image.open(doc["conteudo"]).convert("RGB")
                inputs = self.processor(images=imagem, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    emb = self.model.get_image_features(**inputs)
                emb = self.normalize(emb.cpu().numpy())
                embeddings_list.append(emb[0])
                self.doc_info.append({"idx": indice, "tipo": "imagem", "conteudo": doc["descricao"], "nome": doc["conteudo"], "tipo de uso do solo": doc["tipo de uso do solo"], "cores": doc["cores"], "padrão visual": doc["padrão visual"]})

        self.embeddings = np.array(embeddings_list)
        return self.embeddings

    def upload_vector_database(self, embeddings):
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        return self.index

    def search(self, query_embedding, k=3):
        if self.index is None:
            print("Erro: Índice não foi criado. Execute upload_vector_database primeiro.")
            return None, None
        
        scores, indices = self.index.search(query_embedding, k=k)
        return scores, indices
        
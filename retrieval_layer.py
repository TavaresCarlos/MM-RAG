import faiss
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from numpy.linalg import norm
import ollama
import os


class retrieval_layer:
    rank_final = []
    query = None

    def __init__(self, model_name):
        self.model_name = model_name
        self.model = CLIPModel.from_pretrained(self.model_name)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.emb = None

    def normalize(self, x):
        return x / (norm(x, axis=1, keepdims=True) + 1e-10)

    def get_query_embedding(self, query, tipo_consulta):
        self.query = query
        if tipo_consulta == "texto":
            inputs = self.processor(text=[query], return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                emb = self.model.get_text_features(**inputs)
        else:
            if not os.path.exists(query):
                print("⚠️ Imagem de consulta não disponível.")
                return None
            imagem = Image.open(query).convert("RGB")
            inputs = self.processor(images=imagem, return_tensors="pt").to(self.device)
            with torch.no_grad():
                emb = self.model.get_image_features(**inputs)

        self.emb = self.normalize(emb.cpu().numpy())
        return self.emb

    def is_cafe(self, doc_info):
        for doc in doc_info:
            return doc['conteudo'] == 'área de café'

    def metricas(self, docs_info, indices, scores):
        #RECALL@K e PRECISION@K
        cont = 0
        for i in indices[0]:
            cont+=1
            if docs_info[i]['conteudo'] == 'área de café':
                precision = 1 / (cont)
                print("Recall@K: 1.0")
                print("Precision@K: ", precision)
                break
        #SIMILARITY SPREAD
        spread = max(scores[0]) - min(scores[0])
        print("Similarity Spread: ", spread)
        print()

    def retrieval(self, query, index, doc_info, tipo_consulta="texto", k=3, threshold=0.6): 
        query_emb = self.get_query_embedding(query, tipo_consulta)
        if query_emb is None:
            return

        scores, indices = index.search(query_emb, k=k)

        eh_cafe = self.is_cafe(doc_info)

        if eh_cafe:
            self.metricas(doc_info, indices, scores)

        #print(f"\n🔎 Consulta: '{query if tipo_consulta == 'texto' else 'Imagem'}'")
        #print(f"Resultados encontrados: {len(indices[0])}")
        #print(f"Doc info type: {type(doc_info)}, length: {len(doc_info) if hasattr(doc_info, '__len__') else 'N/A'}")
        
        # DEBUG: Verificar o conteúdo de doc_info
        #if doc_info and hasattr(doc_info, '__len__'):
        #    print(f"Primeiro elemento de doc_info: {doc_info[0] if len(doc_info) > 0 else 'Lista vazia'}")
        
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            # Verificar se o índice é válido e se doc_info é uma lista de dicionários
            if (isinstance(doc_info, list) and 
                len(doc_info) > idx and 
                isinstance(doc_info[idx], dict)):
                
                info = doc_info[idx]
                emoji = "📄" if info["tipo"] == "texto" else "🖼️"
                print(f"{rank}. {emoji} {info['tipo'].title()}: {info['conteudo']}  | Nome do aquivo: {info['conteudo']} | Nome do arquivo: {info['nome']} | Score: {score:.4f}")
                self.rank_final.append((info['nome'], score, info['conteudo'], info['tipo de uso do solo'], info['cores'], info['padrão visual']))
            else:
                print(f"{rank}. Índice inválido ou estrutura incorreta: {idx}")

    def get_rank_final(self):
        return self.rank_final

    def get_query(self):
        return self.query

        
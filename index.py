'''
ESTRATÉGIAS DE COMO REALIZAR UMA BUSCA ESPARSA USANDO IMAGENS
'''
import faiss
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from numpy.linalg import norm
import ollama
import os

from indexing_layer import indexing_layer
from retrieval_layer import retrieval_layer
from generation_layer import generate_layer

# ==============================
#  CONFIGURAÇÃO DO MODELO
# ==============================

model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def normalize(x):
    return x / (norm(x, axis=1, keepdims=True) + 1e-10)

# ==============================
#  DADOS DE TESTE
# ==============================

docs = [
    {"tipo": "imagem", "conteudo": "./assets/img/1.png", "descricao": "área de café", "embedding_tipo": "imagem", "tipo de uso do solo": "Área agrícola com fileiras regulares de plantas cultivadas.", "cores": "Verde-claro, marrom", "padrão visual": "Linhas paralelas alternando vegetação e solo exposto"}, 
    {"tipo": "imagem", "conteudo": "./assets/img/2.png", "descricao": "área de café", "embedding_tipo": "imagem", "tipo de uso do solo": "Área agrícola com fileiras regulares de plantas cultivadas.", "cores": "Verde-claro, marrom", "padrão visual": "Linhas paralelas alternando vegetação e solo exposto"}, 
    {"tipo": "imagem", "conteudo": "./assets/img/9.png", "descricao": "área de pastagem", "embedding_tipo": "imagem", "tipo de uso do solo": "Área escura e densa, com cobertura vegetal uniforme.", "cores": "Verde-escuro, marrom", "padrão visual": "Textura homogênea sem linhas regulares"},
    {"tipo": "imagem", "conteudo": "./assets/img/4.png", "descricao": "área de floresta", "embedding_tipo": "imagem", "tipo de uso do solo": "Área com vegetação de tonalidade azulada, possivelmente densa e irregular.", "cores": "Azul-esverdeado, verde-escuro", "padrão visual": "Denso, com variação de sombras"},
    {"tipo": "imagem", "conteudo": "./assets/img/5.png", "descricao": "área de floresta", "embedding_tipo": "imagem", "tipo de uso do solo": "Área de vegetação com textura média e tonalidade verde-acinzentada.", "cores": "Verde, cinza", "padrão visual": "Uniforme, sem linhas visíveis"},
    {"tipo": "imagem", "conteudo": "./assets/img/6.png", "descricao": "área de floresta", "embedding_tipo": "imagem", "tipo de uso do solo": "Área de vegetação com textura média e tonalidade verde-acinzentada.", "cores": "Verde, cinza", "padrão visual": "Uniforme, sem linhas visíveis"}
]


# ==============================
#  EXECUÇÃO PRINCIPAL
# ==============================

print("=" * 60)
print("BUSCA MULTIMODAL OTIMIZADA")
print("=" * 60)

# Inicializar e criar índice
indexing = indexing_layer(docs, "openai/clip-vit-base-patch32")
embeddings = indexing.to_embedding()
index = indexing.upload_vector_database(embeddings)

# Inicializar retriever
retriever = retrieval_layer("openai/clip-vit-base-patch32")

#print(f"Índice criado com {len(embeddings)} embeddings")
#print(f"Documentos processados: {len(indexing.doc_info)}")
#print(f"Estrutura do doc_info: {type(indexing.doc_info)}")

# Verificar a estrutura dos dados
'''
if indexing.doc_info:
    print(f"Primeiro documento: {indexing.doc_info[0]}")
'''

while True:
    print("\n\nConsidere as opções: ")
    print("1- Busca textual")
    print("2- Busca visual")
    print("0- Sair")

    try:
        opcao = int(input("Digite a opcao desejada: "))
    except ValueError:
        print("Por favor, digite um número válido.")
        continue

    if opcao == 0:
        break
    elif opcao == 1:
        query = input("Digite a query textual: ")
        # CORREÇÃO: Garantir que estamos passando a lista correta
        retriever.retrieval(query, index, indexing.doc_info, "texto", 3)
    elif opcao == 2:
        query = input("Nome da imagem (ex: 1.png): ")
        # CORREÇÃO: Garantir que estamos passando a lista correta
        retriever.retrieval(query, index, indexing.doc_info, "imagem", 3)
        rank = retriever.get_rank_final()
        
        rank.append((retriever.get_query(), 1))

        response = generate_layer(rank)
        resultado = response.process_image(rank)
        print(resultado)
        
    else:
        print("Opção inválida. Tente novamente.")

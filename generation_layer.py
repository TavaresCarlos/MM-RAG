import faiss
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from numpy.linalg import norm
import ollama
import os
import base64

import requests
import json

class generate_layer():
    llm_generation = []
    rank = None

    def __init__(self, rank):
        self.rank = rank

    def encode_image_to_base64(self, path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def process_image(self, images_path, model="llava:latest"):
        """
        Classifica uma imagem alvo comparando com 3 exemplos textuais,
        usando LLaVA multimodal rodando na GPU.
        """

        # Construção do prompt reduzido e otimizado
        prompt = f"""
    Analise a imagem enviada e compare com os três exemplos abaixo.

    Exemplo 1:
    - Score: {images_path[0][1]}
    - Descrição: {images_path[0][2]}
    - Uso do solo: {images_path[0][3]}
    - Cores: {images_path[0][4]}
    - Padrão visual: {images_path[0][5]}

    Exemplo 2:
    - Score: {images_path[1][1]}
    - Descrição: {images_path[1][2]}
    - Uso do solo: {images_path[1][3]}
    - Cores: {images_path[1][4]}
    - Padrão visual: {images_path[1][5]}

    Exemplo 3:
    - Score: {images_path[2][1]}
    - Descrição: {images_path[2][2]}
    - Uso do solo: {images_path[2][3]}
    - Cores: {images_path[2][4]}
    - Padrão visual: {images_path[2][5]}

    TAREFA:
    Com base apenas no conteúdo visual da imagem enviada e nas descrições acima,
    diga com qual exemplo a imagem alvo mais se parece.

    REGRAS:
    - Responda somente JSON válido.
    - Se estiver em dúvida, retorne "0" como classe.

    FORMATO:
    {{
        "classe": "1|2|3|0",
        "explicacao": "texto"
    }}
    """

        system_prompt = """
    Você é um analista de sensoriamento remoto.
    Seu trabalho é comparar imagens e descrever qual exemplo fornecido
    é mais compatível com a cena observada.
    Responda de forma objetiva.
    """

        # Apenas a imagem alvo é enviada para análise multimodal
        image_b64 = self.encode_image_to_base64(images_path[3][0])

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": prompt,
                "images": [image_b64]   # <---- SOMENTE a imagem alvo
            }
        ]

        print("Enviando prompt para LLaVA (GPU)...")

        try:
            response = ollama.chat(
                model=model,
                messages=messages,
                options={
                    "temperature": 0.1,   # mais preciso e estável
                    "top_k": 20,
                    "top_p": 0.8,
                    "num_ctx": 4096       # ótimo para LLaVA com GPU
                }
            )
            print("Finalizado com sucesso.")
            return response["message"]["content"]

        except Exception as e:
            raise Exception(f"Erro na requisição ao LLaVA: {e}")

            prompt = f"""
                Considere as seguintes imagens e seus respectivos scores e descrições retornadas de uma busca RAG:

                Exemplo 1:
                - Caminho: {images_path[0][0]}
                - Score de similaridade: {images_path[0][1]}
                - Descrição: {images_path[0][2]}
                - Uso do solo: {images_path[0][3]}
                - Cores: {images_path[0][4]}
                - Padrão visual: {images_path[0][5]}

                Exemplo 2:
                - Caminho: {images_path[1][0]}
                - Score de similaridade: {images_path[1][1]}
                - Descrição: {images_path[1][2]}
                - Uso do solo: {images_path[1][3]}
                - Cores: {images_path[1][4]}
                - Padrão visual: {images_path[1][5]}

                Exemplo 3:
                - Caminho: {images_path[2][0]}
                - Score de similaridade: {images_path[2][1]}
                - Descrição: {images_path[2][2]}
                - Uso do solo: {images_path[2][3]}
                - Cores: {images_path[2][4]}
                - Padrão visual: {images_path[2][5]}

                Agora, analise a seguinte imagem desconhecida:

                Imagem alvo:
                - Caminho: {images_path[3][0]}

                Tarefa:
                Com base nos três exemplos anteriores e nos scores passados, determine a qual exemplo (1, 2 ou 3) a imagem alvo mais se assemelha.
                Em caso de dúvida, retorne "0"
                Retorne sua resposta **no formato JSON**, contendo:
                - o número do exemplo mais semelhante
                - uma breve explicação do motivo.

                Formato esperado:
                {{
                "classe": "1|2|3|0",
                "explicacao": "A imagem alvo apresenta [característica X] que está mais alinhada com o Exemplo [N] devido a [elemento específico]. 
                Enquanto isso, os outros exemplos mostram [diferenças relevantes]. O score de similaridade [mencionar se reforça essa conclusão]."
                }}

                
            """

            system_prompt = """
                Você é um analista especializado em sensoriamento remoto. 
                Seu papel é identificar padrões e classificar imagens de satélite com base em exemplos de referência.
                Use apenas as informações fornecidas. 
                Se não for possível determinar a classe, responda "Não encontrei correspondência".
            """

            messages = []
            
            if system_prompt:
                messages.append({
                    'role': 'system',
                    'content': system_prompt
                })
            
            messages.append({
                'role': 'user',
                'content': prompt,
                'images': [self.encode_image_to_base64(img[0]) for img in images_path]
            })
            
            #Send request
            print("Enviando o prompt: ", prompt)
            
            try:
                response = ollama.chat(
                    model=model,
                    messages=messages,
                    options={
                        "temperature": 0.5,
                        "top_k": 20,
                        "top_p": 0.8,
                        "num_ctx": 8072
                    }
                )
                print("Finalizado")
                return response['message']['content']
            except Exception as e:
                raise Exception(f"Error Ollama Request: {e}")
            
            '''
            try:
                url = "http://localhost:11434/api/chat" 

                payload = {
                    "model": model,
                    "messages": messages,
                    "stream": False, 
                    "options": {
                        "temperature": 0.5,
                        "top_k": 20,
                        "top_p": 0.8,
                        "num_ctx": 8072
                    }
                }

                response = requests.post(url, json=payload)
                data = response.json()
                print("Finalizado")
                
                return data["message"]["content"]

            except Exception as e:
                raise Exception(f"Erro ao enviar requisição ao Ollama online: {e}")
            '''

    def get_llm_generation(self):
        return self.llm_generation

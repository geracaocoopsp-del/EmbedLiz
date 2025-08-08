import os
import csv

# Caminho da pasta onde estão os .txt
pasta_txt = "data/txt"
arquivo_saida = "data/metadata.csv"

# Lista para guardar os metadados
linhas = []

# Itera sobre todos os arquivos da pasta
for nome_arquivo in os.listdir(pasta_txt):
    if nome_arquivo.lower().endswith(".txt"):
        # Remove a extensão
        nome_sem_extensao = os.path.splitext(nome_arquivo)[0]

        # Divide ID e título (supondo padrão ID_Titulo-Simplificado.txt)
        partes = nome_sem_extensao.split("_", 1)

        if len(partes) == 2:
            id_doc = partes[0]
            titulo_doc = partes[1].replace("-", " ")
        else:
            # Caso não tenha underscore, ID fica vazio
            id_doc = ""
            titulo_doc = nome_sem_extensao.replace("-", " ")

        # Adiciona à lista
        linhas.append([id_doc, titulo_doc, nome_arquivo])

# Salva no CSV
with open(arquivo_saida, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "titulo", "arquivo"])  # Cabeçalho
    writer.writerows(linhas)

print(f"Arquivo '{arquivo_saida}' criado com {len(linhas)} linhas.")

from flask import Flask, render_template, request, jsonify
import chromadb
import os
from groq import Groq
from dotenv import load_dotenv

app = Flask(__name__)

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Obter a chave de API da variável de ambiente
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Inicializar o cliente do ChromaDB
chroma_client = chromadb.Client()
chroma_client = chromadb.PersistentClient(path="db")
collection = chroma_client.get_or_create_collection(name="artigo")

# Função para dividir o texto longo em pedaços menores
def quebra_texto(texto, pedaco_tamanho=1000, sobrepor=200):
    if pedaco_tamanho <= sobrepor:
        raise ValueError("pedaco necessita ser maior do que o sobrepor")

    pedacos = []
    inicio = 0
    while inicio < len(texto):
        final = inicio + pedaco_tamanho
        pedacos.append(texto[inicio:final])
        if final >= len(texto):
            inicio = len(texto)
        else:
            inicio += pedaco_tamanho - sobrepor

    return pedacos

# Ler o arquivo de texto e dividir em pedaços
with open("texto.txt", "r", encoding="utf-8") as file:
    texto = file.read()

pedacos = quebra_texto(texto)

# Adicionar cada pedaço ao ChromaDB
for i, pedaco in enumerate(pedacos):
    collection.add(documents=pedaco, ids=[str(i)])

# Prompt para o assistente
prompt = """
Você é um assistente do Restaurante Erik Sabores.
Use o seguinte contexto para responder a questão, não use nenhuma informação adicional, se nao houver informacao no contexto, responda: Desculpe mas não consigo ajudar.
"""

# Função para consultar o ChromaDB
def consultar_chromadb(questao):
    results = collection.query(query_texts=questao, n_results=2)
    conteudo = results["documents"][0][0] + results["documents"][0][1]
    return conteudo

# Função para gerar resposta usando Groq
def gerar_resposta_groq(prompt, conteudo, questao):
    client = Groq(api_key = GROQ_API_KEY)
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": prompt},
            {"role": "system", "content": conteudo},
            {"role": "user", "content": questao},
        ],
        model="llama-3.1-70b-versatile",
    )
    return chat_completion.choices[0].message.content

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/perguntar', methods=['POST'])
def perguntar():
    questao = request.form.get('questao')

    if questao:
        try:
            conteudo = consultar_chromadb(questao)
            resposta = gerar_resposta_groq(prompt, conteudo, questao)
            return jsonify({'resposta': resposta})
        except AttributeError as e:
            return jsonify({'erro': str(e)})
    else:
        return jsonify({'erro': 'Por favor, digite uma pergunta.'})

if __name__ == '__main__':
    app.run(debug=True)

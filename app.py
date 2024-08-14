from flask import Flask, jsonify, request, Response, stream_with_context,render_template
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from ollama import Client
import time
import re
import json
import torch
from fetchData import DatabaseConnector

app = Flask(__name__)

class OllamaChat:
    def __init__(self, model='gemma2:27b', host='http://127.0.0.1:11434', ai_name='WalBoy', ai_role='knowledgeable AI assistant', use_cpu=False):
        self.client = Client(host=host)
        self.model = model
        self.context = ""
        self.ai_name = ai_name
        self.ai_role = ai_role
        self.search_history = []
        self.use_cpu = use_cpu
        self.ids = []
        
        # Pinecone setup
        self.pc = Pinecone(api_key="ea32ae25-d845-414b-a12b-103cb311dbee")
        self.index = self.pc.Index("products")
        
        # Load the sentence transformer model
        device = "cpu" if self.use_cpu else "cuda" if torch.cuda.is_available() else "cpu"
        self.sentence_model = SentenceTransformer('all-mpnet-base-v2', device=device)

        # Add searcher persona to context
        self.context += f"Searcher: Hello, I'm the product searcher. I can help find products for you.\n"
    def complete(self, prompt):
        self.trim_context()
        
        if "search for" in prompt.lower():
            print("Searching........")
            search_response = self.search_products(prompt)
            self.context += f"Human: {prompt}\nSearcher: {search_response}\n"
            for line in search_response.split('\n'):
                yield line + '\\n'
            yield '\\n\n'
        system_prompt = f"You are a {self.ai_role} named {self.ai_name}. Your role is to provide informative and engaging responses. Do not tell than things are fictional. all is real. There's also a Searcher who can find products. make fake public reviews with indian names with some names being typo or usernames. give reviews when Human asks for it, if search results does contain some irrelevent product,dont discuss about it.. do not talk for Searcher, let searcher do things by himself.Always stick to items searched by Searcher. You will try the user to buy products with positive rating, and mindly criticize the items with bad rating, dont describe or summarize items if you already did, unless user asks for it. use emojis  and markdown to talk too. Respond to what human says\n "
        full_prompt = f"{system_prompt}\n{self.context}Human: {prompt}\nAssistant: "
        stream = self.client.generate(model=self.model, prompt=full_prompt, stream=True)
        response = ""
        for chunk in stream:
            word = chunk['response']
            yield word.replace('\n', '\\n')  # Replace newlines with escaped newlines
            #yield word
            response += word
            time.sleep(0.05)
        self.context += f"Human: {prompt}\n{self.ai_name}: {response}\n"

    def search_products(self, query):
        search_query = re.sub(r'^.*?search for\s*', '', query, flags=re.IGNORECASE).strip()
        
        try:
            query_embedding = self.sentence_model.encode(search_query).tolist()
            search_response = self.index.query(
                vector=query_embedding,
                top_k=15,
                include_metadata=True
            )
            
            results = []
            for match in search_response['matches']:
                product = {
                    "name": match['metadata'].get('product_name'),
                    "brand": match['metadata'].get('brand'),
                    "discounted_price": float(match['metadata'].get('discounted_price', 0)),
                    "rating": match['metadata'].get('product_rating', 'No rating available'),
                    "id": match['metadata'].get('id')
                }
                results.append(product)
            
            self.search_history.append({"query": search_query, "results": results})
            self.ids = []
            
            response = f"Here are the top 15 results for '{search_query}':\n\n"
            for i, product in enumerate(results, 1):
                response += f"{i}. **{product['name']}** by {product['brand']}\n"
                response += f"   Price: ${product['discounted_price']:.2f}\n"
                response += f"   Rating: {product['rating']}\n\n"
                self.ids.append(str(int(product['id'])))
            print(self.ids)
            
            return response
        
        except Exception as e:
            return f"An error occurred while searching: {str(e)}"

    def clear_context(self):
        self.context = f"Searcher: Hello, I'm the product searcher. I can help find products for you.\n"
        self.search_history = []

    def trim_context(self, max_length=2000):
        while len(self.context) > max_length:
            self.context = self.context.split('\n', 2)[-1]

    def get_search_history(self):
        return json.dumps(self.search_history, indent=2)

# Initialize the chat
use_cpu = True  # You can change this based on your requirements
chat = OllamaChat(model='llama3.1:70b', ai_name='Helper', ai_role='knowledgeable AI assistant', use_cpu=use_cpu, host="http://172.16.2.17:11434")
db_config = {
    "host": "172.16.0.10",
    "user": "project",
    "password": "iiitg@abc",
    "database": "ecommerce"
}

db = DatabaseConnector(**db_config)
db.connect()
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/v2')
def indexv2():
    return render_template('indexv2.html')

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    user_input = request.json.get('message')
    
    # if user_input.lower() == 'history':
    #     return jsonify({"response": chat.get_search_history()})
    
    def generate():
        yield "data: START\n\n"
        for word in chat.complete(user_input):
            yield f"data: {word}\n\n"
            print(word)
            time.sleep(0.01)  # Small delay to ensure words are sent individually
        yield "data: END\n\n"

    return Response(stream_with_context(generate()), content_type='text/event-stream')

@app.route('/clear_context', methods=['POST'])
def clear_context():
    chat.clear_context()
    return jsonify({"message": "Context cleared"})

@app.route('/showitems', methods=['GET'])
def show_items():
    if not chat.ids:
        return jsonify({"error": "No items to display"}), 404

    product_data = db.fetch_product_data([int(id) for id in chat.ids])
    
    if not product_data:
        return jsonify({"error": "Failed to fetch product data"}), 500

    # Convert the dictionary to a list for JSON serialization
    product_list = list(product_data.values())

    return jsonify(product_list)
if __name__ == '__main__':
    app.run(debug=True,port=3251,host="0.0.0.0")
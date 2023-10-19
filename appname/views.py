# myapp/views.py
import os
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from .models import Chat
import json
import logging
from dotenv import load_dotenv
import spacy


# Find the .env file and load the environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)


# Access the environment variables
OPENAI_KEY = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = OPENAI_KEY
pdf_link = "./dataset.pdf"


# PDF processing and model initialization
loader = PyMuPDFLoader(pdf_link)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
texts = text_splitter.split_documents(documents)
persist_directory = "./storage"
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=persist_directory)
vectordb.persist()
retriever = vectordb.as_retriever()
llm = ChatOpenAI(model_name='gpt-3.5-turbo')
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# Initialize an empty conversation history
conversation_history = []

@csrf_exempt
def chat_function(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        query = data.get('query', '')
        response = ''
        if has_context_switched(query, conversation_history):
            # Clear conversation history on context switch
            conversation_history.clear()
        # Append the current query to the conversation history
        conversation_history.append(query)
        try:
            # Combine the entire conversation history as context
            context = ' '.join(conversation_history)
            re = qa(context)
            logger = logging.getLogger(__name__)
            response = re['result']
            logger.info(response)
        except Exception as err:
            response = "Error: " + str(err)
        # Store the conversation in the database
        chat = Chat.objects.create(query=query, response=response)
        # Return the response
        return JsonResponse({'query': chat.query, 'response': chat.response})

def has_context_switched(query, conversation_history):
    # Define a threshold for context switching, e.g., if the query is too dissimilar
    threshold = 0.7
    # Calculate the similarity between the new query and the entire conversation history
    similarities = [similarity(query, prev_query) for prev_query in conversation_history]
    # If the maximum similarity is below the threshold, consider it a context switch
    if max(similarities, default=0) < threshold:
        return True
    else:
        return False
# Implement a function to measure the similarity between two queries
def similarity(query1, query2):
    # Implement a similarity measurement method, such as cosine similarity, Jaccard index, or other NLP techniques
    # You can use NLP libraries like spaCy or scikit-learn to calculate similarity
    # Return a value between 0 and 1, where 1 means the queries are identical
    # Example code using cosine similarity with spaCy:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc1 = nlp(query1)
    doc2 = nlp(query2)
    similarity = doc1.similarity(doc2)
    return similarity
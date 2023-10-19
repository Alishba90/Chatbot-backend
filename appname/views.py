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
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=10)  # Adjusted chunk_size to resolve the batch size error
texts = text_splitter.split_documents(documents)
persist_directory = "./storage"
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=persist_directory)
vectordb.persist()
retriever = vectordb.as_retriever()
llm = ChatOpenAI(model_name='gpt-3.5-turbo')
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

@csrf_exempt
def chat_function(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        
        query = data.get('query', '')
        response = ''  # Add your logic to get the response based on the query
        try:
            re = qa(query)
            
            logger = logging.getLogger(__name__)
            response = re['result']
            logger.info(response)
            
        except Exception as err:
            response = "Error " + str(err)

        
        chat = Chat.objects.create(query=query, response=response)
        return JsonResponse({'query': chat.query, 'response': chat.response})

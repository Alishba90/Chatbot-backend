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
from django.http import JsonResponse
from .models import Chat
import json

os.environ['OPENAI_API_KEY'] = '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'

# PDF processing and model initialization
loader = PyMuPDFLoader("./data.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=550, chunk_overlap=10)
texts = text_splitter.split_documents(documents)

persist_directory = "./storage"
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(documents=texts, 
                                 embedding=embeddings,
                                 persist_directory=persist_directory)
vectordb.persist()

retriever = vectordb.as_retriever()
llm = ChatOpenAI(model_name='gpt-3.5-turbo')
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# Chatbot API
@csrf_exempt
def chatbot_api(request):
    if request.method == 'POST':
        user_input = request.POST.get('question', '')
        query = f"###Prompt {user_input}"
        try:
            llm_response = qa(query)
            return JsonResponse({"responseeeeeeeeeeee": llm_response["result"]})
        except Exception as err:
            return JsonResponse({"error": str(err)})
    else:
        return JsonResponse({"error": "Invalid request method. Please use POST."})

@csrf_exempt
def alishbafunction(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        query = data.get('query', 'Hello ALishba')
        response = 'Hello ALishba'  # Add your logic to get the response based on the query
        chat = Chat.objects.create(query=query, response=response)
        return JsonResponse({'query': chat.query, 'response': chat.response})
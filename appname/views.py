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
import logging

os.environ['OPENAI_API_KEY'] = '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'

pdf_link="./data.pdf"

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

@csrf_exempt
def alishbafunction(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        
        query = data.get('query', '')
        response = ''  # Add your logic to get the response based on the query
        try:
            re = qa(query)
            
            logger = logging.getLogger(__name__)
            response=re['result']
            logger.info(f'Response: {response}')
            
        except Exception as err:
            response="Error "+str(err)

        
        chat = Chat.objects.create(query=query, response=response)
        return JsonResponse({'query': chat.query, 'response': chat.response})
from flask import Flask, jsonify, request, send_file
from gtts import gTTS
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
import os

app = Flask(__name__)

os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

loader = TextLoader("data1.txt")
vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
index = VectorStoreIndexWrapper(vectorstore=vectorstore)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chain = ConversationalRetrievalChain.from_llm(llm=ChatOpenAI(), retriever=index.vectorstore.as_retriever(), memory=memory, verbose=True)

@app.route('/generate-text/<input_text>', methods=['POST'])
def generate_text(input_text):
    result = chain({"question": input_text, "chat_history": []})
    generated_text = result['answer']

    tts = gTTS(text=generated_text, lang='en')
    tts.save("output.mp3")

    return jsonify({
        'generated_text': generated_text,
        'audio_url': request.host_url + 'audio'
    })

@app.route('/audio')
def get_audio():
    return send_file("output.mp3", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)

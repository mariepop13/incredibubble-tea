import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
 
# Contenu de la barre latérale
with st.sidebar:
    st.title('🤗💬 Application de chat LLM')
    st.markdown('''
    ## À propos
    Cette application est un chatbot alimenté par LLM, construit avec :
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - Modèle LLM [OpenAI](https://platform.openai.com/docs/models)
 
    ''')
    add_vertical_space(5)
    st.write('Créé avec ❤️ par [mariepop13](https://youtube.com/@engineerprompt)')
 
load_dotenv()
 
def main():
    st.header("Chat avec un fichier PDF 💬")
 
 
    # Télécharger un fichier PDF
    pdf = st.file_uploader("Téléchargez votre PDF", type='pdf')
 
    # st.write(pdf)
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
 
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)
 
        # # embeddings
        nom_fichier = pdf.name[:-4]
        st.write(f'{nom_fichier}')
        # st.write(chunks)
 
        if os.path.exists(f"{nom_fichier}.pkl"):
            with open(f"{nom_fichier}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            # st.write('Embeddings Loaded from the Disk')s
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{nom_fichier}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
 
        # embeddings = OpenAIEmbeddings()
        # VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
 
        # Accepter les questions/requêtes de l'utilisateur
        query = st.text_input("Posez des questions sur votre fichier PDF :")
        # st.write(query)
 
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
 
            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)
 
if __name__ == '__main__':
    main()
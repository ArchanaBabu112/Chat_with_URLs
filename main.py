import os
import streamlit as st
import faiss
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()  # Take environment variables from .env (especially OpenAI API key)

st.title("Webpage Query Assistant with LLM")
st.sidebar.title("Upload the URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:  # Only append non-empty URLs
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
index_file_path = "faiss_index_file.index"
metadata_file_path = "faiss_metadata.pkl"

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked and urls:  # Only proceed if there are URLs to process
    # Load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅ ")
    data = loader.load()
    
    # Split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    
    if docs:  # Ensure there are documents to process
        # Create embeddings and save them to FAISS index
        embeddings = OpenAIEmbeddings()
        vectorstore_openai = FAISS.from_documents(docs, embeddings)
        main_placeholder.text("Embedding Vector Started Building...✅✅✅")
        time.sleep(2)

        # Save the FAISS index to a file
        faiss.write_index(vectorstore_openai.index, index_file_path)

        # Save the metadata (docstore and index_to_docstore_id)
        metadata = {
            "docstore": vectorstore_openai.docstore,
            "index_to_docstore_id": vectorstore_openai.index_to_docstore_id
        }
        with open(metadata_file_path, "wb") as f:
            pickle.dump(metadata, f)
    else:
        main_placeholder.error("No documents were loaded. Please check the URLs and try again.")

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(index_file_path) and os.path.exists(metadata_file_path):
        # Load the FAISS index from the file
        index = faiss.read_index(index_file_path)
        
        # Load the metadata
        with open(metadata_file_path, "rb") as f:
            metadata = pickle.load(f)
        
        # Recreate the vectorstore with the loaded index and metadata
        vectorstore = FAISS(
            embedding_function=OpenAIEmbeddings(),  # Recreate embeddings
            docstore=metadata["docstore"],
            index=index,
            index_to_docstore_id=metadata["index_to_docstore_id"]
        )
        
        # Setup the retrieval chain
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm
                                                     , retriever=vectorstore.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)
        
        # Display the result
        st.header("Answer")
        st.write(result["answer"])
        
        # Display sources, if available
        sources = result.get("sources", "")
        # if sources:
        #     st.subheader("Sources:")
        #     sources_list = sources.split("\n")  # Split the sources by newline
        #     for source in sources_list:
        #         if source.strip():  # Check if the source is not empty
        #             st.write(source)
    else:
        st.error("No index or metadata file found. Please process URLs first.")
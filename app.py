import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

apiKey = "google_api_key"

st.set_page_config(page_title="Simple Document Summarizer", layout="wide")

# Concise Streamlit content
st.markdown("""
## RAGDemo: Simple Document Summarizer

1. **Upload Documents**: Upload PDFs, DOCX, TXT, or CSV files.
2. **Ask a Question**: After processing, ask questions related to the content for accurate answers.
""")

def extractPdfText(pdfFiles):
    text = ""
    for pdf in pdfFiles:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    return text

def splitTextIntoChunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

def createVectorStore(textChunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=apiKey)
    vectorStore = FAISS.from_texts(textChunks, embedding=embeddings)
    vectorStore.save_local("faissIndex")

def getQaChain():
    promptTemplate = """
    Answer the question based on the context provided. If the answer is unavailable, say "Answer is not available in the context."
    
    Context: {context}
    Question: {question}
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=apiKey)
    prompt = PromptTemplate(template=promptTemplate, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def processUserQuery(query):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=apiKey)
    vectorDb = FAISS.load_local("faissIndex", embeddings)
    docs = vectorDb.similarity_search(query)
    qaChain = getQaChain()
    response = qaChain({"input_documents": docs, "question": query}, return_only_outputs=True)
    st.write("Response: ", response["output_text"])

def main():
    st.header("AI Chatbot Assistant")

    query = st.text_input("Ask a question about the uploaded documents:", key="userQuery")

    if query:  
        processUserQuery(query)

    with st.sidebar:
        st.title("Menu")
        uploadedFiles = st.file_uploader("Upload your documents here", accept_multiple_files=True, key="fileUploader")
        if st.button("Process Documents", key="processBtn"):
            if uploadedFiles:
                with st.spinner("Processing..."):
                    rawText = extractPdfText(uploadedFiles)
                    textChunks = splitTextIntoChunks(rawText)
                    createVectorStore(textChunks)
                    st.success("Documents processed and vector store created!")
            else:
                st.error("Please upload some files to proceed.")

if __name__ == "__main__":
    main()

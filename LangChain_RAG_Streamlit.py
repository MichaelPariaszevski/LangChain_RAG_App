import streamlit as st
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os


def load_document_2(file):
    import os

    name, extension = os.path.splitext(file)
    print(name)
    print("-" * 50)
    print(extension)
    print("-" * 50)
    if extension == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader

        print(f"Loading {file}")
        loader = PyPDFLoader(file)
    elif extension == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader

        print(f"Loading {file}")
        loader = Docx2txtLoader(file)
    elif extension == ".txt":
        from langchain_community.document_loaders import TextLoader

        loader = TextLoader(file)
    else:
        print("Document format is not supported!")
        return None
    data = loader.load()
    print("-" * 50)
    print(f"Successfully Loaded File: {file}")
    return data


def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(
        data
    )  # If data is not already split into pages, use create_documents
    return chunks


def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store


def ask_and_get_answer(vector_store, q, k=3):
    from langchain.chains import RetrievalQA
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=1)
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever
    )
    answer = chain.invoke(q)
    return answer["result"]


def calculate_embedding_cost(texts):
    import tiktoken

    enc = tiktoken.encoding_for_model("text-embedding-3-small")
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    # print(f"Total Tokens: {total_tokens}")
    # print(f"Embedding Cost in USD: {total_tokens/1000*0.0004:.6f}")
    return (total_tokens, total_tokens / 1000 * 0.0004)


def clear_history():
    if "history" in st.session_state:  # If the session state exists, delete it
        del st.session_state["history"]


def clear_history_2():  # Delete st.session_state["history"] or delete st.session_state.history
    del st.session_state["history"]


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv, find_dotenv

    load_dotenv(find_dotenv(), override=True)

    # def clear_history():
    #     if "history" in st.session_state: # If the session state exists, delete it
    #         del st.session_state["history"]

    st.image("LangChain_image.png")
    st.subheader("LLM Question-Answering Application")

    with st.sidebar:

        api_key = st.text_input("OpenAI API Key: ", type="password")

        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        uploaded_file = st.file_uploader("Upload a file: ", type=["pdf", "docx", "txt"])
        chunk_size = st.number_input(
            "Chunk size: ",
            min_value=100,
            max_value=2048,
            value=512,
            on_change=clear_history,
        )
        k = st.number_input(
            "K value: ", min_value=1, max_value=20, value=3, on_change=clear_history
        )

        add_data = st.button("Add data", on_click=clear_history)

        clear_history_button = st.button("Clear Cache", on_click=clear_history)

        if (
            uploaded_file and add_data
        ):  # If the user uploaded a file and clicked the "add_data" button
            with st.spinner("Reading, chunking, and embedding file ... "):

                bytes_data = (
                    uploaded_file.read()
                )  # Read the file from its current binary state
                file_name = os.path.join(
                    "./", uploaded_file.name
                )  # Copy the file (bytes_data) to the current directory ("./") and rename it "uploaded_file.name"
                with open(file_name, "wb") as f:
                    f.write(bytes_data)

                data = load_document_2(file_name)

                chunks = chunk_data(data, chunk_size=chunk_size)
                st.write(f"Chunk size {chunk_size}, Chunks: {len(chunks)}")

                tokens, embedding_cost = calculate_embedding_cost(chunks)
                st.write(f"Embeddings cost: ${embedding_cost:.4f}")

                vector_store = create_embeddings(chunks)
                st.session_state.vs = vector_store

                st.success("File uploaded, chunked, and embedded successfully")
    q = st.text_input("Ask a question about the content of your file: ")
    if q:
        if "vs" in st.session_state:
            vector_store = st.session_state.vs
            st.write(f"k: {k}")
            answer = ask_and_get_answer(vector_store, q, k)
            st.text_area("LLM Answer: ", value=answer)

            st.divider()

            if "history" not in st.session_state:
                st.session_state["history"] = ""
            value = f"Q: {q}\nA: {answer}"
            st.session_state["history"] = (
                f'{value} \n {"-"*100} \n {st.session_state["history"]}'
            )
            h = st.session_state["history"]
            st.text_area(
                label="Chat History", value=h, key="history", height=400
            )  # key=history means that we will store the value of this widget in the session state as the value of a key named "history" (the session state is like a dictionary)

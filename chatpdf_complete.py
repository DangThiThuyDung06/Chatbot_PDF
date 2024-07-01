import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.agents import AgentExecutor, create_tool_calling_agent
from qdrant_client import QdrantClient
import os
import json

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OPENAI_API_KEY"] = ''


# embeddings = SpacyEmbeddings(model_name="en_core_web_sm")
embeddings = OpenAIEmbeddings()
# Initialize Qdrant client
#client = QdrantClient(host="localhost", port=6333)  # Adjust the host and port as necessary

def pdf_read(pdf_doc):
    # text = ""
    # for pdf in pdf_doc:
    #     pdf_reader = PdfReader(pdf)
    #     for page in pdf_reader.pages:
    #         text += page.extract_text()
    # return text
    loader = PyPDFLoader("documents/" + pdf_doc)
    pages = loader.load()
    # text = ""
    # for page in pages:
    #     text += page.page_content  # Combine text from all pages
    return pages

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=48)
    chunks = text_splitter.split_documents(text)
    # chunks = text_splitter.split_text(text)
    return chunks

def vector_store(text_chunks):
    # if not text_chunks:
    #     st.error("Không có đoạn văn bản nào để tạo kho lưu trữ vector.")
    #     return None
    vector_store = Qdrant.from_documents(
        text_chunks,
        embedding=embeddings,
        path="qdrant_db",
        collection_name="my_documents_2",
    )
    return vector_store

def save_chat_history(chat_history):
    with open("chat_history.json", "w", encoding="utf-8") as f:
        json.dump(chat_history, f, ensure_ascii=False, indent=4)

def load_chat_history():
    try:
        with open("chat_history.json", "r", encoding="utf-8") as f:
            chat_history = json.load(f)
    except FileNotFoundError:
        chat_history = []
    return chat_history

def load_vector_store():
    # if not _text_chunks:
    #     st.error("Không có đoạn văn bản nào để tải kho lưu trữ vector.")
    #     return None
    # qdrant = Qdrant.from_documents(
    #     _text_chunks,
    #     embeddings,
    #     path="qdrant_db",
    #     collection_name="                     ",
    # )
    # return qdrant
    client = QdrantClient(path="./qdrant_db")
    db = Qdrant(collection_name="my_documents_2",
                embeddings=embeddings, client=client)
    return db
    # db = Qdrant(collection_name=collection_name, embeddings=embedding_function, path=persist_directory)
def get_conversational_chain(tools, ques, chat_history):
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0, api_key=os.getenv('OPENAI_API_KEY'), max_tokens=4096)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Bạn là một trợ lý thông thái với nhiệm vụ hỗ trợ người dùng truy xuất và cung cấp thông tin từ tài liệu PDF. Bạn sẽ trả lời các câu hỏi liên quan đến nội dung trong file PDF được cung cấp một cách chi tiết và chính xác nhất.

                ### Quy trình hội thoại:
                1. Chào hỏi người dùng:
                - Chào mừng người dùng với lời chào thân thiện và chuyên nghiệp.
                - Ví dụ: "Xin chào! Tôi có thể giúp gì cho bạn hôm nay?"

                2. Kiểm tra nhu cầu hỗ trợ của người dùng:
                - Hỏi người dùng xem họ có cần giúp đỡ gì không và lắng nghe yêu cầu của họ.
                - Ví dụ: "Bạn cần tôi hỗ trợ gì từ tài liệu này?"

                3. Trả lời yêu cầu của người dùng:
                a. Nếu yêu cầu là cung cấp hoặc trích xuất thông tin từ PDF:
                    - Đọc kỹ tài liệu PDF và cung cấp thông tin chi tiết, chính xác. Trích dẫn nguyên văn từ tài liệu khi cần thiết.
                    - Ví dụ: "Theo tài liệu, đoạn văn bản tại trang X, đoạn Y nói rằng..."

                b. Nếu yêu cầu là tóm tắt hoặc liệt kê thông tin:
                    - Đọc kỹ tài liệu và tổng hợp các thông tin quan trọng theo yêu cầu của người dùng.
                    - Ví dụ câu hỏi: "Có bao nhiêu...?", "Liệt kê tất cả...?"
                    - Tóm tắt hoặc liệt kê các thông tin chi tiết từ tài liệu một cách rõ ràng và có cấu trúc.
                    - Ví dụ: "Tài liệu liệt kê các điểm chính như sau: 1) ..., 2) ..., 3) ..."

                ### Mục tiêu:
                - Đảm bảo trả lời chính xác và chi tiết dựa trên nội dung tài liệu PDF.
                - Cung cấp trải nghiệm người dùng thân thiện, chuyên nghiệp và hữu ích."""
            ),
            ("user", "\n".join([f"User: {entry['input']}\nAssistant: {entry['output']}" for entry in chat_history])),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    tool = [tools]
    agent = create_tool_calling_agent(llm, tool, prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tool, verbose=True)
    response = agent_executor.invoke({"input": ques})
    st.write("Trả lời: ", response['output'])

    chat_history.append({"input": ques, "output": response['output']})
    save_chat_history(chat_history)

def user_input(user_question, chat_history):
    new_db = load_vector_store()
    if new_db:
        retriever = new_db.as_retriever()
        retrieval_chain = create_retriever_tool(retriever, "pdf_extractor", "This tool is to give answer to queries from the pdf")
        get_conversational_chain(retrieval_chain, user_question, chat_history)

def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("RAG based Chat with PDF")

    chat_history = load_chat_history()

    user_question = st.text_input("Hỏi một câu hỏi từ các tập tin PDF")

    if user_question:
        user_input(user_question, chat_history)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Tải lên các tập tin PDF của bạn và nhấn nút Gửi & Xử lý", accept_multiple_files=True)
        if st.button("Gửi & Xử lý"):
            if pdf_docs:
                with st.spinner("Đang xử lý..."):
                    for pdf_doc in pdf_docs:
                        pdf_content = pdf_doc.read()
                        save_path = os.path.join("./documents", pdf_doc.name)
                        with open(save_path, "wb") as f:
                            f.write(pdf_content)
                        st.success(f"Đã lưu {pdf_doc.name}")
                        raw_text = pdf_read(pdf_doc.name)
                        text_chunks = get_chunks(raw_text)
                        # print(text_chunks)
                        if text_chunks:
                            vector_store(text_chunks)
                        else:
                            st.error(f"Không thể trích xuất văn bản từ {pdf_doc.name}")

if __name__ == "__main__":
    main()



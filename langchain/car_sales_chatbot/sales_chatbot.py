import gradio as gr

from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter


def init_data():
    with open("real_estate_sales_data.txt", 'r', encoding='utf-8') as f:
        real_estate_sales = f.read()
        text_splitter = CharacterTextSplitter(
            separator=r'\d+\.',
            chunk_size=100,
            chunk_overlap=0,
            length_function=len,
            is_separator_regex=True,
        )
        docs = text_splitter.create_documents([real_estate_sales])
        db = FAISS.from_documents(docs, OpenAIEmbeddings())
        db.save_local("real_estates_sale")

def initialize_sales_bot(vector_store_dir: str="real_estates_sale"):
    db = FAISS.load_local(vector_store_dir, OpenAIEmbeddings())
    global LLM
    LLM = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0)
    
    global SALES_BOT    
    SALES_BOT = RetrievalQA.from_chain_type(LLM,
                                            retriever=db.as_retriever(search_type="similarity_score_threshold",
                                                                     search_kwargs={"score_threshold": 0.8}))
    # 返回向量数据库的检索结果
    SALES_BOT.return_source_documents = True

    return SALES_BOT

def sales_chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    # TODO: 从命令行参数中获取
    enable_chat = True

    ans = SALES_BOT({"query": message})
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果
    if ans["source_documents"] or enable_chat:
        if len(ans['source_documents']) == 0:
            input = [("system", "你是一个资深汽车销售，你可以回答所有和汽车有关的问题。")]
            response = LLM.invoke(input).content
            print(f"[result]{response}")
            return response
        else:
            print(f"[source_documents]{ans['source_documents']}")
            return ans["result"]
    # 否则输出套路话术
    else:
        return "这个问题我要问问领导"
    

def launch_gradio():
    demo = gr.ChatInterface(
        fn=sales_chat,
        title="汽车销售",
        # retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="0.0.0.0")

if __name__ == "__main__":
    init_data()
    initialize_sales_bot()
    launch_gradio()

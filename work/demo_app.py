import os
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
from uuid import uuid4

import langchain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.documents.base import Document
from langchain.chains.combine_documents.stuff import (
    StuffDocumentsChain,
    create_stuff_documents_chain,
)
from langchain.chains.llm import LLMChain
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.runnables.base import RunnableSequence
from langchain.output_parsers.regex import RegexParser
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import load_tools, AgentExecutor, create_openai_functions_agent
from langchain.chains.combine_documents.refine import RefineDocumentsChain
from langchain_community.callbacks.manager import get_openai_callback
from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores import FAISS
from langsmith import Client
from langfuse.callback import CallbackHandler

load_dotenv(override=True)  # .envファイルの中身を環境変数に設定


def set_langsmith():
    unique_id = uuid4().hex[0:8]
    os.environ["LANGCHAIN_PROJECT"] = f"Tracing Walkthrough - {unique_id}"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    client = Client()
    return unique_id


def unset_langsmith():
    os.environ["LANGCHAIN_TRACING_V2"] = "false"


def initialize_vectorstore() -> (
    FAISS
):  # ベクトルDBをlangchain内のインスタンスとして作成する関数
    embedding = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(
        "/workspace/work/vector_DB",
        embedding,
        allow_dangerous_deserialization=True,
    )
    return vectorstore


def create_stuff_chain(for_langfuse: bool) -> RunnableSequence:
    prompt = ChatPromptTemplate.from_template(
        "以下のquestionにcontextで与えられた情報のみを用いて答えてください。"
        "わからない場合は「わかりません」と出力してください。"
        "context:{context}"
        "question:{question}"
    )
    if for_langfuse:
        callbacks = [
            CallbackHandler(
                public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
                secret_key=os.environ["LANGFUSE_SECRET_KEY"],
                host=os.environ["LANGFUSE_HOST"],
                session_id=str(uuid4())[:8],
            )
        ]
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, callbacks=callbacks)
        stuff_chain = create_stuff_documents_chain(llm, prompt)
        return stuff_chain, callbacks
    else:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        stuff_chain = create_stuff_documents_chain(llm, prompt)
        return stuff_chain


def create_refine_chain(for_langfuse: bool, callbacks) -> RunnableSequence:
    if for_langfuse:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, callbacks=callbacks)
    else:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    document_prompt = PromptTemplate(  # 文書を整形するためのプロンプト(任意)
        input_variables=["page_content"], template="{page_content}"
    )
    initial_prompt = ChatPromptTemplate.from_template(  # 初回に使用されるプロンプト
        "以下のquestionにcontextで与えられた情報のみを用いて答えてください。"
        "context:{context}"
        "question:{question}"
    )
    refine_prompt = ChatPromptTemplate.from_template(  # 2回目以降に使用されるプロンプト
        "以下のquestionに対するあなたの以前の回答が「{prev_response}」です。"
        "追加の情報である以下のcontextを用いて以前の回答をより良い回答にしてください。"
        "question:{question}"
        "context:{context}"
    )
    refine_chain = RefineDocumentsChain(
        initial_llm_chain=LLMChain(llm=llm, prompt=initial_prompt),
        refine_llm_chain=LLMChain(llm=llm, prompt=refine_prompt),
        document_prompt=document_prompt,
        document_variable_name="context",
        initial_response_name="prev_response",
    )
    return refine_chain


def create_agent_chain(
    for_langfuse: bool, callbacks
) -> AgentExecutor:  # エージェントを作る関数
    if for_langfuse:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, callbacks=callbacks)
    else:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    tool1 = load_tools(tool_names=["ddg-search"])  # ツールを初期化し作成
    tool2 = [
        create_retriever_tool(
            st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3}),
            "search_about_hikakukgakuron",
            f"さまざまな研究方法を説明した文書である比較科学論について検索して, 関連性が高い一部を返します。",
        )
    ]
    tools = tool1 + tool2
    system_prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                "あなたは聞かれた質問に答える優秀なAIアシスタントです。"
            ),
            HumanMessagePromptTemplate.from_template("{question}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=system_prompt)
    return AgentExecutor(agent=agent, tools=tools, callbacks=callbacks)


# 以下がstreamlit用コード
load_dotenv(override=True)
st.title("LLM実験管理ツール用デモアプリ")

tool_choice = st.radio(
    "記録する実験管理ツールを選択してください", ("LangSmith", "Langfuse")
)
chain_choice = st.radio(
    "chainの種別を選択してください",
    (
        "RAGの簡単なchain",
        "RAGの複雑なchain",
        "LLMエージェント",
    ),
)
st.markdown(f"**{chain_choice}**の結果が**{tool_choice}**上に記録されます。")

if "vectorstore" not in st.session_state:  # vectorstoreを初期化
    st.session_state.vectorstore = initialize_vectorstore()

if "memory" not in st.session_state:  # memoryを初期化
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

if "chat_memory" not in st.session_state:  # 会話履歴表示用memory(list[dict])を初期化
    st.session_state.chat_memory = []

if "stuff_chain" not in st.session_state:  # stuffチェイン(通常)の初期化
    st.session_state.stuff_chain = create_stuff_chain(for_langfuse=False)

if "stuff_chain_langfuse" not in st.session_state:  # stuffチェイン(langfuse用)の初期化
    chain, callbacks = create_stuff_chain(for_langfuse=True)
    st.session_state.stuff_chain_langfuse = chain
    st.session_state.callbacks_langfuse = callbacks

if "refine_chain" not in st.session_state:  # refineチェイン(通常)の初期化
    st.session_state.refine_chain = create_refine_chain(
        for_langfuse=False, callbacks=st.session_state.callbacks_langfuse
    )

if (
    "refine_chain_langfuse" not in st.session_state
):  # refineチェイン(langfuse用)の初期化
    st.session_state.refine_chain_langfuse = create_refine_chain(
        for_langfuse=True, callbacks=st.session_state.callbacks_langfuse
    )

if "agent" not in st.session_state:
    st.session_state.agent = create_agent_chain(
        for_langfuse=False, callbacks=st.session_state.callbacks_langfuse
    )

if "agent_langfuse" not in st.session_state:
    st.session_state.agent_langfuse = create_agent_chain(
        for_langfuse=True, callbacks=st.session_state.callbacks_langfuse
    )

for message in st.session_state.chat_memory:  # 以前の会話履歴の表示
    with st.chat_message("user"):
        st.markdown(message["prompt"])
    with st.chat_message("assistant"):
        st.markdown(message["result"])
        st.markdown(message["link"])

prompt = st.chat_input(
    "比較科学論に関する質問はありますか?(ex:アマゾン型の研究とはなんですか?)"
)  # chat入力欄の表示

if prompt:
    if tool_choice == "LangSmith":  # LangSmith idのセット
        uid = set_langsmith()
    else:
        unset_langsmith()
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("文書検索中..."):
        retriever = st.session_state.vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )  # 3×200文字のretrieverを作成
        context = retriever.get_relevant_documents(prompt)  # ベクトルDBからデータ取得

    with st.spinner("回答生成中..."):  # 回答の生成と表示
        if tool_choice == "Langfuse":  # langfuseに記録する場合
            if chain_choice == "RAGの簡単なchain":
                chain = st.session_state.stuff_chain_langfuse
                result = chain.invoke(
                    {"context": context, "question": prompt},
                    config={"callbacks": st.session_state.callbacks_langfuse},
                )
            elif chain_choice == "RAGの複雑なchain":
                chain = st.session_state.refine_chain_langfuse
                result = chain.invoke(
                    {"input_documents": context, "question": prompt},
                    config={"callbacks": st.session_state.callbacks_langfuse},
                )["output_text"]
            elif chain_choice == "LLMエージェント":
                chain = st.session_state.agent_langfuse
                result = chain.invoke(
                    {"question": prompt},
                    config={"callbacks": st.session_state.callbacks_langfuse},
                )["output"]
        else:  # langsmithに記録する場合
            if chain_choice == "RAGの簡単なchain":
                result = st.session_state.stuff_chain.invoke(
                    {"context": context, "question": prompt}
                )
            elif chain_choice == "RAGの複雑なchain":
                result = st.session_state.refine_chain.invoke(
                    {"input_documents": context, "question": prompt}
                )["output_text"]
            elif chain_choice == "LLMエージェント":
                chain = st.session_state.agent
                result = chain.invoke({"question": prompt})["output"]
    with st.chat_message("assistant"):
        st.markdown(result)
        if tool_choice == "Langfuse":  # langfuseに記録した場合
            link_message = "[ここにアクセスして記録を確認](http://localhost:3000)"
        else:  # langsmithに記録した場合
            link_message = (
                "[ここにアクセスして記録を確認](https://smith.langchain.com/)"
            )
        st.markdown(link_message)

    st.session_state.chat_memory.append(  # メモリへの保存
        {"prompt": prompt, "result": result, "link": link_message}
    )

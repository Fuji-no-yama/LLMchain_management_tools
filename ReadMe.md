# LangChainから使えるLLM実験管理ツール比較用デモアプリ
langfuseサーバをセルフホストするための「langfuse」ディレクトリとコード等が格納されている「work」ディレクトリの2つが存在します

```
.
├── LICENSE
├── ReadMe.md
├── langfuse
│   ├ ...
├── requirements.txt
└── work
    ├── add_document.py
    ├── demo_app.py
    ├── doc
    └── vector_DB
```

## workフォルダ内の各ファイル名と概要
- add_document.py : ベクトルDBへ文書をベクトル化して登録するプログラム
- demo_app.py : 3種類のチェインとRAGを用いてDXに関する質問応答を行い, そのログを指定した実験管理ツールに記録するデモアプリです。

## 各プログラムの概要と実行方法
※各プログラム実行前にルートディレクトリにて開発コンテナ機能を用いてコンテナを作成し, workディレクトリに移動してください。


<details><summary>demo_app.py</summary>

### 概要
LangChainを用いて作成された以下の3種類のchainを自由に選択して事前に登録されている「比較科学論」([青空文庫](https://www.aozora.gr.jp/cards/001569/card53214.html)より)の内容について質問できるデモアプリです。その際のchainの動作について指定したLLM実験管理ツールに記録することができます。

- 簡単なRAG chain : stuff chainを用いて実装
- 複雑なRAG chain : refine chainを用いて実装
- LLMエージェント chain : web検索とRAG検索が使用できるエージェントとして実装

### 実行方法
- このリポジトリをクローン  
    ```bash
    git clone https://github.com/Fuji-no-yama/LLMchain_management_tools
    ```
- ルートディレクトリの.devcontainerを用いて開発コンテナを作成する
- workフォルダ内部に.env.sampleを参考に.envファイルを作成し`OPENAI_API_KEY`にAPIキーを記入する
- [LangSmithの公式サイト](https://smith.langchain.com/)にアクセスしてLangChainのアカウントを作成する
- 左下の「setting」ページからAPIキーを発行し.envファイルの`LANGCHAIN_API_KEY`に書き込む
- [http://localhost:3000](http://localhost:3000)にアクセスしてlangfuseのアカウントを作成する
- ページの左下にあるsettingから同様にsecret keyとpublic keyを作成し, .envファイルの`LANGFUSE_SECRET_KEY`と`LANGFUSE_PUBLIC_KEY`に記載する
- デモアプリリポジトリのルートディレクトリの中の「work」ディレクトリに移動し`streamlit run langfuse_langsmith_test.py`でアプリを起動する
- [https://localhost:8501](https://localhost:8501)にアクセスする
- 画面上部のボタンで記録する媒体と使用するchainを選択してプロンプトに質問を入力する。
- 回答が生成されたら選択した記録媒体ごとにリンクが表示されるため, それを押して実験管理ツールの画面にて実際の記録を確認する。

</details>

<details><summary>add_document.py</summary>

### 概要
ローカルのベクトルDBに指定した文書をベクトル化して登録するpythonプログラムです。(ファイル名を指定して実行するとDBが書き変わってしまうため実行には注意してください)

### 実行方法
- .envファイルをwork/以下に作成し, 以下の内容を記載する。
```
OPENAI_API_KEY=***
PINECONE_API_KEY=***
PINECONE_INDEX_NAME=***
```
- プログラム中の`filepath =`の部分に追加したい文書のファイルパスを記入する
- 指定した文書が青空文庫のテキストファイルの場合はルビを削除するためのremove_ruby関数のコメントアウトを外す
- `python add_document.py`で実行する
</details>
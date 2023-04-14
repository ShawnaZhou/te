# ChatGLM Application Based on Local Knowledge

## Introduction

🌍 [_中文文档_](README.md)

🤖️ A local knowledge based LLM Application with [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) and [langchain](https://github.com/hwchase17/langchain).

💡 Inspired by [document.ai](https://github.com/GanymedeNil/document.ai) by [GanymedeNil](https://github.com/GanymedeNil) and [ChatGLM-6B Pull Request](https://github.com/THUDM/ChatGLM-6B/pull/216) by [AlexZhangji](https://github.com/AlexZhangji).

✅ In this project, [GanymedeNil/text2vec-large-chinese](https://huggingface.co/GanymedeNil/text2vec-large-chinese/tree/main) is used as Embedding Model，and [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) used as LLM。Based on those models，this project can be deployed **offline** with all **open source** models。

## Usage

### 1. install python packages
```commandline
pip install -r requirements
```
Attention: With langchain.document_loaders.UnstructuredFileLoader used to connect with local knowledge file, you may need some other dependencies as mentioned in  [langchain documentation](https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/unstructured_file.html)

### 2. Run [knowledge_based_chatglm.py](knowledge_based_chatglm.py) script
```commandline
python knowledge_based_chatglm.py
```

## Roadmap
- [x] local knowledge based application with langchain + ChatGLM-6B
- [x] unstructured files loaded with langchain
- [ ] more different file format loaded with langchain
- [ ] implement web ui DEMO with gradio/streamlit 
- [ ] implement API with fastapi，and web ui DEMO with API


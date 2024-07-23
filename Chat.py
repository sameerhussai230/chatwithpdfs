import streamlit as st
from langchain.schema.messages import AIMessage, HumanMessage
import re
from llama_index.llms.groq import Groq
from llama_index.llms.groq import Groq
from llama_index.core import PromptTemplate
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import download_loader, Document, VectorStoreIndex, ServiceContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import IndexNode
from langchain.embeddings import HuggingFaceInstructEmbeddings
# from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext
import time
import os
from llama_index.core.prompts.prompts import SimpleInputPrompt
from llama_index.core import SimpleDirectoryReader

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from pathlib import Path
from llama_index.core import QueryBundle
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
# import NodeWithScore
from llama_index.core.schema import NodeWithScore


from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import CitationQueryEngine


# Retrievers
from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
)

from dotenv import load_dotenv
import os
load_dotenv()

Groq_api_key=os.getenv("GROQ_API_KEY")
@st.cache_resource(show_spinner=False)
def get_models(model_name):
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-large-en-v1.5"
    )
    # os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

    system_prompt = """
    You are a Q&A assistant. Your goal is to answer questions as
    accurately as possible based on the instructions and context provided.
    You are not to divert from the context provided
    If the answer is not present in the context provided , then first warn the user politely about the answer being not there and then generate the Answer accorrding to query
    """
    Settings._prompt_helper = system_prompt

    ## Default format supportable by LLama2
    # query_wrapper_prompt=SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")

    def messages_to_prompt(messages):
        prompt = ""
        for message in messages:
            if message.role == 'system':
                prompt += f"<|system|>\n{message.content}</s>\n"
            elif message.role == 'user':
                prompt += f"<|user|>\n{message.content}</s>\n"
            elif message.role == 'assistant':
                prompt += f"<|assistant|>\n{message.content}</s>\n"

        # ensure we start with a system prompt, insert blank if needed
        if not prompt.startswith("<|system|>\n"):
            prompt = "<|system|>\n</s>\n" + prompt

        # add final assistant prompt
        prompt = prompt + "<|assistant|>\n"
        print(prompt)

        return prompt

    llm = Groq(model=model_name, api_key=Groq_api_key,temperature=0.5,system_prompt=system_prompt)

    return llm, embed_model





def gen_nodes(doc):
    parser = LlamaParse(
    api_key=os.getenv("LLAMA-PARSE"),  # can also be set in your env as LLAMA_CLOUD_API_KEY
    result_type="markdown",  # "markdown" and "text" are available
    verbose=True,
)
    file_extractor = {".pdf": parser}
    documents = SimpleDirectoryReader(
    input_files=[f"{doc}"],file_extractor=file_extractor
).load_data()
    node_parser = SentenceSplitter(chunk_size=1024) 
    base_nodes = node_parser.get_nodes_from_documents(documents)
    # set node ids to be a constant
    for idx, node in enumerate(base_nodes):
        node.id_ = f"node-{idx}"
    sub_chunk_sizes = [256, 512]
    sub_node_parsers = [SentenceSplitter(chunk_size=c) for c in sub_chunk_sizes]

    all_nodes = []
    for base_node in base_nodes:
        for n in sub_node_parsers:
            sub_nodes = n.get_nodes_from_documents([base_node])
            sub_inodes = [
                IndexNode.from_text_node(sn, base_node.node_id) for sn in sub_nodes
            ]
            all_nodes.extend(sub_inodes)

        # also add original node to node
        original_node = IndexNode.from_text_node(base_node, base_node.node_id)
        all_nodes.append(original_node)
    all_nodes_dict = {n.node_id: n for n in all_nodes}

    return all_nodes

@st.cache_data(show_spinner=False)
def get_query_engine(path):
    service_context = ServiceContext.from_defaults(
    llm=llm, embed_model=embed_model
)
    all_nodes=gen_nodes(path)
    print("all nodes made")
    storage_context = StorageContext.from_defaults()
    vector_ind = VectorStoreIndex(
all_nodes, service_context=service_context,storage_context=storage_context
)
    print("index made")
    vector_r = vector_ind.as_retriever(similarity_top_k=2)
    query_engine_chunk = RetrieverQueryEngine.from_args(
    vector_r, service_context=service_context)
    return query_engine_chunk


def get_response(prompt):
    # query_engine = RetrieverQueryEngine(retriever=retriever)
    response=st.session_state.query_engine.query(prompt)
    return response

def save_uploadedfile(uploadedfile):
    with open(os.path.join("tempDir",uploadedfile.name),"wb") as f:
        f.write(uploadedfile.getbuffer())
    # return st.success("Saved File:{} to tempDir".format(uploadedfile.name))

# st.set_page_config(page_title="Chat with PDFs", page_icon="🤖")
st.title("Chat with PDFs")

# sidebar
left_column, right_column = st.columns(2)
with st.sidebar:
    st.header("Settings")
    pdf=st.file_uploader("Upload your PDF",type="pdf")
    
    options = ["Mixtral 8x7b", "LLaMA3 70b", "LLaMA3 8b"]
    selected_option = st.selectbox("Choose a Model:", options)
    if selected_option =="LLaMA3 8b":
        option = "llama3-8b-8192"
    elif selected_option =="Mixtral 8x7b":
        option = "mixtral-8x7b-32768"
    elif selected_option== "LLaMA3 70b":
        option="llama3-70b-8192"
    print(option)

    # Create the dropdown list
    

    # Display the selected option
    st.write(f"You selected: {selected_option}")
    
llm,embed_model=get_models(option)
    


if pdf is None :
    for key in st.session_state.keys():
        del st.session_state[key]
    st.info("Please upload pdf")
else:
    save_uploadedfile(pdf)
    Settings.llm=llm
    Settings.embed_model=embed_model
    path=f"tempDir/{pdf.name}"
    # pdf="temp/maternity_benefit.pdf"
    # session state
    import base64

    file = path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = F'<embed id="pdfViewer" src="data:application/pdf;base64,{base64_pdf}" width="500" height="1000" type="application/pdf">'

    # Displaying File
    # st.markdown(pdf_display, unsafe_allow_html=True)

    # Scroll to and highlight text
    # html = (f'''
    #    <style>
    # section[data-testid="stSidebar"] 
    # {
    #     width: 300px;
    # }
    # </style>
    # ''')
    
    if "query_engine" not in st.session_state:
        with st.spinner("Analysing Data!"):
            st.session_state.query_engine = get_query_engine(path)    
        # user input
    user_query = st.chat_input("Type your message here...")
    with right_column:
        # st.sidebar.set_width(300)
        st.markdown(pdf_display, unsafe_allow_html=True)
    with left_column:    
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
        # if "chat_history" not in st.session_state:
        #     st.session_state.chat_history = [
        #         AIMessage(content="Hello, I am a bot. How can I help you?"),
        #     ]tempDir/maternity_benefit.pdf
        
        # message_placeholder = st.empty()
        if user_query is not None and user_query != "":
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)
            with st.spinner("Retrieveing Answer"):
                response = get_response(user_query)
                text=response.metadata
                document_info = str(text)
                print(document_info)
                print(response)
                #find = re.findall(r"'page_label': '[^']*', 'file_name': '[^']*'", document_info)
                # page=re.findall(r"'page_label': '[^']*'",document_info)[0]
                page=1
            st.markdown(f"Answer is from Page no: {page}")
            st.markdown(response)
            print(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            for f in os.listdir("tempDir"):
                os.remove(os.path.join("tempDir", f))
        

    
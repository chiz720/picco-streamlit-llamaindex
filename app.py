import streamlit as st
import openai
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader,Settings

# Load environment variables 

openai.api_key = st.secrets['openai_key']

st.set_page_config(page_title="PiCCO Bot", page_icon=":books:")
st.title('KSA-CCSK Advanced Hemodynamic Monitoring Bot :books:')
st.info("Make sure you register for the 2024 KSA-CCSK Annual Scientific Congress [click to register](https://conference.anaesthesiakenya.co.ke)")

if "messages" not in st.session_state.keys(): # initialize the chat messages history
    st.session_state.messages = [
        {
            "role":"assistant",
            "content": "Ask me a question about Advanced Hemodynamic Monitoring using PiCCO Techonology",
        }
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    reader = SimpleDirectoryReader(input_dir="data ", recursive=True)
    docs = reader.load_data()
    Settings.llm = OpenAI(
        model="gpt-3.5-turbo",
        temperature=0.2,
        system_prompt="""You are an Intensivist with over 30 years experience in advanced
        hemodynamic monitoring, with special interest in PiCCO technology. Your job is to
        answer technical questions in advanced hemodynamic monitoring and PiCCO. You are ble 
        to explain complex concepts, including formulas. Keep your answers technical and 
        clinical. Do not hallucinate features.
        """
    )
    index = VectorStoreIndex.from_documents(docs)
    return index

index = load_data()

if "chat_engine" not in st.session_state.keys(): # Initilize chat engine
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_question", verbose=True,streaming=True
    )

if prompt := st.chat_input("Ask a question"):  # prompt for user input and save to chat history
    st.session_state.messages.append({"role":"user", "content":"prompt"})

for message in st.session_state.messages: # write history to UI
    with st.chat_message(message["role"]):
        st.write(message["content"])

# if last message is not from assistant, generate new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response_stream = st.session_state.chat_engine.stream_chat(prompt)
        st.write_stream(response_stream.response_gen)
        message = {"role":"assistant", "content": response_stream.response}
        # add response to message history 
        st.session_state.messages.append(message)
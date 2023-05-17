
import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space

from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from langchain.prompts.prompt import PromptTemplate

_template = """You are a LinearB customer support agent. You are chatting with a customer who is asking about LinearB product.
Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CUSTOM_HELP_BOT_PROMPT = PromptTemplate.from_template(_template)

from PIL import Image
import os

PERSIST_DIRECTORY = ".chroma/help_docs"


st.set_page_config(page_title="LinearB chatbot")

image = Image.open('./linearb.png')

st.title('LinearB Chatbot')
st.image(image, width=100)

if "chain" not in st.session_state:
    import config
    db = Chroma(persist_directory=PERSIST_DIRECTORY, 
            embedding_function=OpenAIEmbeddings(), 
            collection_name="help_docs")
    
    # # expose this index in a retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":4})
    qa_retrieval_chain = ConversationalRetrievalChain.from_llm(llm=OpenAI(),
                                                               condense_question_prompt=CUSTOM_HELP_BOT_PROMPT,
                                                            retriever=retriever,
                                                            return_source_documents=False)
    st.session_state['chain'] = qa_retrieval_chain
    st.session_state['history'] = []
    

# Sidebar contents
with st.sidebar:
    st.title('LinearB Chatbot')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [help docs](https://linearb.helpdocs.io/)
    ''')
    add_vertical_space(5)
    st.write('LinearB is a software delivery intelligence platform.\nLinearB helps engineering leaders deliver products faster and with higher quality.')

# Generate empty lists for generated and past.
## generated stores AI generated responses
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["What would you like to know about LinearB product?"]
## past stores User's questions
if 'past' not in st.session_state:
    st.session_state['past'] = ['Hi!']

# Layout of input/response containers
input_container = st.container()
colored_header(label='', description='', color_name='blue-30')
response_container = st.container()

# User input
## Function for taking user provided prompt as input
def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text

## Applying the user input box
with input_container:
    user_input = get_text()

# Response output
## Function for taking user prompt as input followed by producing AI generated responses
def generate_response(prompt):
    for i, message in enumerate(st.session_state['history']):
        print(f"{i}:{message}")
    chain = st.session_state.chain
    response = chain({'chat_history':st.session_state.history,'question':prompt})
    return response

## Conditional display of AI generated responses as a function of user provided prompts
with response_container:
    if user_input:
        response = generate_response(user_input)
        st.session_state.history.append((response['question'], response['answer']))
        st.session_state.past.append(user_input)
        st.session_state.generated.append(response['answer'])
        
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))
    
    st.button('Clear chat history', on_click=lambda: st.session_state.clear())
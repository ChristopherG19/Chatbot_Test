from flask import Flask, request, jsonify
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from huggingface_hub import hf_hub_download

# https://huggingface.co/docs/huggingface_hub/v0.16.3/en/package_reference/file_download#huggingface_hub.hf_hub_download
hf_hub_download(
    repo_id="TheBloke/Llama-2-7B-Chat-GGML",
    filename="llama-2-7b-chat.ggmlv3.q2_K.bin",
    local_dir="./models"
)

app = Flask(__name__)

def getLLamaresponse(input_text, no_words, blog_style):
    llm = CTransformers(model='models/llama-2-7b-chat.ggmlv3.q2_K.bin',
                        model_type='llama',
                        config={'max_new_tokens':256,
                                'temperature':0.01})
    
    template = """{input_text}"""
    
    prompt = PromptTemplate(input_variables=["blog_style","input_text",'no_words'],
                            template=template)
    
    response = llm(prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words))
    return response

@app.route('/', methods=['GET'])
def Status():
    print("Up")
    return jsonify({'response': "UP"})

@app.route('/llm/generate_text', methods=['POST'])
def generate_blogs():
    data = request.get_json()
    input_text = data.get('input_text')
    no_words = data.get('no_words')
    blog_style = data.get('blog_style')
    response = getLLamaresponse(input_text, no_words, blog_style)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

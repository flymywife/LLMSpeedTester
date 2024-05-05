import os
import time
import gradio as gr
from llama_cpp import Llama


class LlamaConversation:
    def __init__(self, llm_model_path: str, temperature: float, top_p: float, top_k: int, token_limit: int, max_response: int, n_ctx: int):
        self.llm = Llama(model_path=llm_model_path, n_ctx=n_ctx)
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.token_limit = token_limit
        self.max_response = max_response
    
    def generate_response(self, query: str) -> str:
        start_time = time.time()
        response = self.llm(query, max_tokens=self.token_limit, temperature=self.temperature, top_p=self.top_p, top_k=self.top_k)
        if isinstance(response, dict):
            response_text = response['choices'][0]['text'].strip()[:self.max_response]
        else:
            response_text = response.strip()[:self.max_response]
        end_time = time.time()
        elapsed_time = end_time - start_time
        tokens_per_second = len(response_text) / elapsed_time
        return response_text, elapsed_time, tokens_per_second

    def gradio_interface(self, query: str, llm_model_path: str, temperature: float, top_p: float, top_k: int, token_limit: int, max_response: int, n_ctx: int) -> str:
        self.llm = Llama(model_path=llm_model_path, n_ctx=n_ctx)
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.token_limit = token_limit
        self.max_response = max_response
        response, elapsed_time, tokens_per_second = self.generate_response(query)
        return response + f"\n\nElapsed time: {elapsed_time:.2f} seconds\nTokens per second: {tokens_per_second:.2f}"

# LLMモデルのパスのリスト
llm_model_paths = [
    "LLMmodels/Phi-3-mini-4k-instruct-q4.gguf",
    "LLMmodels/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
    "LLMmodels/codellama-34b-instruct.Q4_K_M.gguf",
    "LLMmodels/falcon-40b-Q4_K_M.gguf",
    "LLMmodels/Meta-Llama-3-70B-Instruct-Q4_K_M.gguf",
    "LLMmodels/command-r-plus-Q4_K_M-00002-of-00002.gguf"
    # 他のLLMモデルのパスを追加
]

# LlamaConversationのインスタンスを作成
conversation = LlamaConversation(llm_model_paths[0], temperature=0.7, top_p=0.95, top_k=40, token_limit=100, max_response=500, n_ctx=2048)

# Gradioインターフェースの作成
iface = gr.Interface(
    fn=conversation.gradio_interface,
    inputs=[
        gr.components.Textbox(label="User Query"),
        gr.components.Dropdown(llm_model_paths, label="LLM Model"),
        gr.components.Slider(minimum=0.1, maximum=1.0, value=0.7, label="Temperature"),
        gr.components.Slider(minimum=0.1, maximum=1.0, value=0.95, label="Top P"),
        gr.components.Slider(minimum=1, maximum=100, step=1, value=40, label="Top K"),
        gr.components.Slider(minimum=1, maximum=2000, step=1, value=512, label="Token Limit"),
        gr.components.Slider(minimum=1, maximum=1000, step=1, value=500, label="Max Response"),
        gr.components.Slider(minimum=1, maximum=4096, step=1, value=512, label="n_ctx"),
    ],
    outputs=gr.components.Textbox(label="Llama Response"),
    title="Llama Conversation",
    description="Chat with the Llama model.",
)

# Gradioの起動
iface.launch()
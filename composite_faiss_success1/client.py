from __future__ import annotations

import os
import streamlit as st
import torch

from collections.abc import Iterable
from typing import Any, Protocol
from huggingface_hub.inference._text_generation import TextGenerationStreamResponse, Token
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList
from conversation import Conversation

from langchain_community.vectorstores import FAISS
from config import CONFIG

from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

TOOL_PROMPT = 'Answer the following questions as best as you can. You have access to the following tools:'

MODEL_PATH = os.environ.get('MODEL_PATH', 'THUDM/chatglm3-6b')
PT_PATH = os.environ.get('PT_PATH', None)
PRE_SEQ_LEN = int(os.environ.get("PRE_SEQ_LEN", 128))
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)

vector_store_path = CONFIG['db_source']
embeddings_model_name = CONFIG['embedding_model']
index_name = 'my_index'


@st.cache_resource
def get_client() -> Client:
    client = HFClient(MODEL_PATH, TOKENIZER_PATH, PT_PATH, vector_store_path)
    return client


class Client(Protocol):
    def generate_stream(self,
                        system: str | None,
                        tools: list[dict] | None,
                        history: list[Conversation],
                        **parameters: Any
                        ) -> Iterable[TextGenerationStreamResponse]:
        ...


def stream_chat(
        self, tokenizer, query: str,
        history: list[tuple[str, str]] = None,
        role: str = "user",
        past_key_values=None,
        max_new_tokens: int = 256,
        do_sample=True, top_p=0.8,
        temperature=0.8,
        repetition_penalty=1.0,
        length_penalty=1.0, num_beams=1,
        logits_processor=None,
        return_past_key_values=False,
        **kwargs
):
    class InvalidScoreLogitsProcessor(LogitsProcessor):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
            if torch.isnan(scores).any() or torch.isinf(scores).any():
                scores.zero_()
                scores[..., 5] = 5e4
            return scores

    if history is None:
        history = []

    print("\n== Input ==\n", query)
    print("\n==History==\n", history)

    if logits_processor is None:
        logits_processor = LogitsProcessorList()
    logits_processor.append(InvalidScoreLogitsProcessor())
    eos_token_id = [tokenizer.eos_token_id, tokenizer.get_command("<|user|>"),
                    tokenizer.get_command("<|observation|>")]
    gen_kwargs = {"max_new_tokens": max_new_tokens,
                  "do_sample": do_sample,
                  "top_p": top_p,
                  "temperature": temperature,
                  "logits_processor": logits_processor,
                  "repetition_penalty": repetition_penalty,
                  "length_penalty": length_penalty,
                  "num_beams": num_beams,
                  **kwargs
                  }

    if past_key_values is None:
        inputs = tokenizer.build_chat_input(query, history=history, role=role)
    else:
        inputs = tokenizer.build_chat_input(query, role=role)
    inputs = inputs.to(self.device)
    if past_key_values is not None:
        past_length = past_key_values[0][0].shape[0]
        if self.transformer.pre_seq_len is not None:
            past_length -= self.transformer.pre_seq_len
        inputs.position_ids += past_length
        attention_mask = inputs.attention_mask
        attention_mask = torch.cat((attention_mask.new_ones(1, past_length), attention_mask), dim=1)
        inputs['attention_mask'] = attention_mask
    history.append({"role": role, "content": query})
    input_sequence_length = inputs['input_ids'].shape[1]
    if input_sequence_length + max_new_tokens >= self.config.seq_length:
        yield "Current input sequence length {} plus max_new_tokens {} is too long. The maximum model sequence length is {}. You may adjust the generation parameter to enable longer chat history.".format(
            input_sequence_length, max_new_tokens, self.config.seq_length
        ), history
        return

    if input_sequence_length > self.config.seq_length:
        yield "Current input sequence length {} exceeds maximum model sequence length {}. Unable to generate tokens.".format(
            input_sequence_length, self.config.seq_length
        ), history
        return

    for outputs in self.stream_generate(**inputs, past_key_values=past_key_values,
                                        eos_token_id=eos_token_id, return_past_key_values=return_past_key_values,
                                        **gen_kwargs):
        if return_past_key_values:
            outputs, past_key_values = outputs
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
        response = tokenizer.decode(outputs)
        if response and response[-1] != "�":
            new_history = history
            if return_past_key_values:
                yield response, new_history, past_key_values
            else:
                yield response, new_history


class HFClient(Client):
    def __init__(self, model_path: str, tokenizer_path: str, pt_checkpoint: str = None, vector_store_path: str = CONFIG['db_source']):

        self.model_path = model_path

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

        self.embedding_function = SentenceTransformerEmbeddings(model_name=embeddings_model_name)

        self.vector_store_path = vector_store_path


        if pt_checkpoint is not None and os.path.exists(pt_checkpoint):
            config = AutoConfig.from_pretrained(
                model_path,
                trust_remote_code=True,
                pre_seq_len=PRE_SEQ_LEN
            )
            self.model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                config=config,
                device_map="auto").eval()
            # add .quantize(4).cuda() before .eval() and remove device_map="auto" to use int4 model
            prefix_state_dict = torch.load(os.path.join(pt_checkpoint, "pytorch_model.bin"))
            new_prefix_state_dict = {}
            for k, v in prefix_state_dict.items():
                if k.startswith("transformer.prefix_encoder."):
                    new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
            print("Loaded from pt checkpoints", new_prefix_state_dict.keys())
            self.model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
        else:
            self.model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, device_map="auto").eval()
            # add .quantize(4).cuda() before .eval() and remove device_map="auto" to use int4 model


    def load_vector_store(self, vector_store_path):
        if vector_store_path and os.path.exists(vector_store_path):
            vector_store = FAISS.load_local(
                self.vector_store_path,
                self.embedding_function,
                index_name=index_name,
                allow_dangerous_deserialization=True
            )
        return vector_store

    def retrieve_documents(self, query: str):
        # 确保向量存储已加载
        vector_store = self.load_vector_store(self.vector_store_path)
        # 使用loaded_vector_store进行文档检索
        results = vector_store.similarity_search_with_score(query)
        return results

    def get_document(self, query: str):
        retrieved_docs = self.retrieve_documents(query)
        # 按照得分对检索到的文档进行排序，得分由高到低
        retrieved_docs = sorted(retrieved_docs, key=lambda x: x[1], reverse=True)

        # 提取得分最高的前三个文档
        top_documents = retrieved_docs[:3] if len(retrieved_docs) >= 3 else retrieved_docs

        # 组合前三个文档的内容
        combined_content = ""
        for doc in top_documents:
            if doc:
                combined_content += doc[0].page_content + "\n\n"  # 添加换行符以分隔不同文档的内容

        return combined_content

    def generate_stream(
            self,
            system: str | None,
            tools: list[dict] | None,
            history: list[Conversation],
            **parameters: Any
    ) -> Iterable[TextGenerationStreamResponse]:
        chat_history = [{
            'role': 'system',
            'content': system if not tools else TOOL_PROMPT,
        }]

        if tools:
            chat_history[0]['tools'] = tools

        for conversation in history[:-1]:
            chat_history.append({
                'role': str(conversation.role).removeprefix('<|').removesuffix('|>'),
                'content': conversation.content,
            })

        query = history[-1].content
        role = str(history[-1].role).removeprefix('<|').removesuffix('|>')
        text = ''

        # 是错的
        # # 使用检索功能获取相关文档
        # retrieved_docs = self.retrieve_documents(query)
        # # 将检索到的文档添加到历史中，以供模型生成响应时使用
        # for doc in retrieved_docs:
        #     chat_history.append({
        #         'role': 'document',
        #         'content': doc.content,

        self.results = self.get_document(query)
        if self.results:
            prompt = f"""
            基于检索到的信息和你对苏州旅游了解，回答用户的问题。请确保答案详实而准确，围绕用户的查询点提供有用的信息，同时保持语言的自然和通顺。在回答时，尽量综合利用检索到的文档内容，并适当融入你所知的苏州旅游相关知识，但不要明确提及"根据检索到的文档"这一表述。在可能的情况下，提供具体细节，比如历史背景、文化特色、旅游景点介绍、美食推荐等，使答案更加丰富和吸引人。

            检索到的文档内容如下：
            {self.results}

            用户提出的问题是：
            {query}

            请结合上述内容，用中文撰写你的回答：
            """
            query = prompt


        for new_text, _ in stream_chat(
                self.model,
                self.tokenizer,
                query,
                chat_history,
                role,
                **parameters,
        ):
            word = new_text.removeprefix(text)
            word_stripped = word.strip()
            text = new_text
            yield TextGenerationStreamResponse(
                generated_text=text,
                token=Token(
                    id=0,
                    logprob=0,
                    text=word,
                    special=word_stripped.startswith('<|') and word_stripped.endswith('|>'),
                )
            )


import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import re
import torch
import transformers

from utils.general_utils import passages2string
from c2_generation.src.llm_generator import LLMGenerator_api, LLMGenerator_hf_local
from c2_generation.src.retrievers_local import BM25Retriever, RerankRetriever, DenseRetriever
from c2_generation.src.prompt_templetes import (
    SYSTEM_PROMPT_NO_RETRIEVAL,
    SYSTEM_PROMPT_SINGLE_RETRIEVAL,
    SYSTEM_PROMPT_RESEARCH_INST,
    PROMPT_SEARCHR1,
    PROMPT_STEPSEARCH
)

class BasicRAG:
    def __init__(self, device, args):
        self.args = args
        
        # --- Generators
        if args.model_source == 'api':
            self.generator = LLMGenerator_api(args.model_name_or_path)
        elif args.model_source == 'hf_local':
            backbone_model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name_or_path, dtype=torch.bfloat16).to(device) # attn_implementation="eager"
            backbone_tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)
            self.generator = LLMGenerator_hf_local(backbone_model, backbone_tokenizer, device, args)
        else:
            raise NotImplementedError
        
        # --- Retrievers 
        if args.retriever_name == 'bm25':
            self.retriever = BM25Retriever(args)  
        elif args.retriever_name in ['rerank_l6', 'rerank_l12']:
            self.retriever = RerankRetriever(args)
        elif args.retriever_name in ['contriever', 'dpr', 'e5', 'bge']:
            self.retriever = DenseRetriever(args)

    # --- Information Extraction Functions
    def get_think(self, text):
        pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        matches = pattern.findall(text)
        return matches[0] if matches else None

    def get_query(self, text):
        pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
        matches = pattern.findall(text)
        return matches[0] if matches else None

    def get_answer(self, text):
        pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
        matches = pattern.findall(text)
        return matches[0] if matches else None

class NoRetrieval(BasicRAG):
    def __init__(self, device, args):
        super().__init__(device, args)
        self.user_prompt_template = "Question: {user_query}"
    
    def inference(self, question, generation_temp=0.7):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_NO_RETRIEVAL},
            {"role": "user", "content": self.user_prompt_template.format(user_query=question)}
        ]
        output_text, _ = self.generator.generate(messages, temperature=generation_temp)
        reasoning = self.get_think(output_text)
        prediction = self.get_answer(output_text)
        reasoning_path = [{'think': reasoning, 'prediction': prediction}]
        
        return reasoning_path, prediction

class SingleRetrieval(BasicRAG):
    def __init__(self, device, args):
        super().__init__(device, args)
        self.user_prompt_template = "<information>{documents}</information>\n\nQuestion: {user_query}"
    
    def inference(self, question, generation_temp=0.7):
        search_docs = self.retriever.search(question)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_SINGLE_RETRIEVAL},
            {"role": "user", "content": self.user_prompt_template.format(
                documents=passages2string(search_docs),
                user_query=question
            )}
        ]
        output_text, _ = self.generator.generate(messages, temperature=generation_temp)
        reasoning = self.get_think(output_text)
        prediction = self.get_answer(output_text)
        reasoning_path = [
            {'think': '', 'search_query': '', 'docs': search_docs},
            {'think': reasoning, 'prediction': prediction}
        ]
        
        return reasoning_path, prediction

class ReSearch_Model(BasicRAG):
    def __init__(self, device, args):
        super().__init__(device, args)
        self.curr_step_template = '\n{output_text}<result>{search_results}</result>\n'
        self.answer_template = '<answer> \boxed{answer} </answer>'
    
    def get_boxed_answer(self, text: str) -> str:
        match = re.search(r"\\boxed\{(.*?)\}", text)
        return match.group(1).strip() if match else None   
    
    def inference(self, question, generation_temp=0.7):
        input_prompt = question
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT_RESEARCH_INST},
            {'role': 'user', 'content': input_prompt}
        ]
        
        reasoning_path = []
        while True:
            output_text, output_ = self.generator.generate(
                messages,
                self.generator.rar_stopping_criteria,
                temperature=generation_temp
            )
            if output_[-1].item() in self.generator.curr_eos:
                break

            tmp_query = self.get_query(output_text)
            if tmp_query:
                search_docs = self.retriever.search(tmp_query)
                search_results = passages2string(search_docs)
            else:
                search_docs, search_results = [], ''
                
            reasoning_path.append({
                'think': self.get_think(output_text),
                'search_query': tmp_query,
                'docs': search_docs
            })
            search_text = self.curr_step_template.format(output_text=output_text, search_results=search_results)
            input_prompt += search_text
            messages = [
                {'role': 'system', 'content': SYSTEM_PROMPT_RESEARCH_INST},
                {'role': 'user', 'content': input_prompt}
            ]
        
        one_step_think = self.get_think(output_text)
        prediction = self.get_boxed_answer(output_text)
        reasoning_path.append({'think': one_step_think, 'prediction': prediction})
            
        return reasoning_path, prediction

class SearchR1_Model(BasicRAG):
    def __init__(self, device, args):
        super().__init__(device, args)
        self.curr_step_template = '\n\n{output_text}<information>{search_results}</information>\n\n'
        
    def inference(self, question, generation_temp=0.7):
        input_prompt = PROMPT_SEARCHR1.format(question=question)
        messages = [{"role": "user", "content": input_prompt}]
        
        reasoning_path = []
        while True:
            output_text, output_ = self.generator.generate(
                messages,
                self.generator.rar_stopping_criteria,
                temperature=generation_temp
            )
            if output_[-1].item() in self.generator.curr_eos:
                break
        
            tmp_query = self.get_query(output_text)
            if tmp_query:
                search_docs = self.retriever.search(tmp_query)
                search_results = passages2string(search_docs)
            else:
                search_docs, search_results = [], ''

            reasoning_path.append({
                'think': self.get_think(output_text),
                'search_query': tmp_query,
                'docs': search_docs
            })
            search_text = self.curr_step_template.format(output_text=output_text, search_results=search_results)
            input_prompt += search_text
            messages = [{"role": "user", "content": input_prompt}]

        one_step_think = self.get_think(output_text)
        prediction = self.get_answer(output_text)
        reasoning_path.append({'think': one_step_think, 'prediction': prediction})
            
        return reasoning_path, prediction

class StepSearch_Model(BasicRAG):
    def __init__(self, device, args):
        super().__init__(device, args)
        self.curr_step_template = '\n\n{output_text}<information>{search_results}</information>\n\n'
    
    def inference(self, question, generation_temp=0.7):
        input_prompt = PROMPT_STEPSEARCH.format(question=question)
        messages = [{"role": "user", "content": input_prompt}]
        
        reasoning_path = []
        while True:
            output_text, output_ = self.generator.generate(
                messages,
                self.generator.rar_stopping_criteria,
                temperature=generation_temp
            )
            if output_[-1].item() in self.generator.curr_eos:
                break
        
            tmp_query = self.get_query(output_text)
            if tmp_query:
                search_docs = self.retriever.search(tmp_query)
                search_results = passages2string(search_docs)
            else:
                search_docs, search_results = [], ''

            reasoning_path.append({
                'think': self.get_think(output_text),
                'search_query': tmp_query,
                'docs': search_docs
            })
            search_text = self.curr_step_template.format(output_text=output_text, search_results=search_results)
            input_prompt += search_text
            messages = [{"role": "user", "content": input_prompt}]

        one_step_think = self.get_think(output_text)
        pred_answer = self.get_answer(output_text)
        reasoning_path.append({'think': one_step_think, 'prediction': pred_answer})
            
        return pred_answer, reasoning_path

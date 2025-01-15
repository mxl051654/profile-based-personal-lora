"""
utils for llm
"""
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


class VllmAGent:
    def __init__(self, model_name_or_path=None, adapter_name_or_path=None, beam_search=False):
        mem = 0.4
        in_max_len = 2048
        out_max_len = 1024
        temp = 0.5

        if adapter_name_or_path:
            self.lora_path = adapter_name_or_path[0]
            self.llm = LLM(
                model=model_name_or_path,
                tokenizer=model_name_or_path,
                gpu_memory_utilization=mem,  # 最大显存预算, vllm kv缓存会占用显存
                max_model_len=in_max_len,
                enable_lora=True,
                disable_log_stats=False,  # 显示进度
                disable_async_output_proc=True,
                max_lora_rank=64,  # default 16
            )
        else:
            self.lora_path = None
            self.llm = LLM(
                model=model_name_or_path,
                tokenizer=model_name_or_path,
                gpu_memory_utilization=mem,  # 最大显存预算, vllm kv缓存会占用显存
                max_model_len=in_max_len,
                disable_log_stats=False,  # 显示进度
                disable_async_output_proc=True,
                max_lora_rank=64,  # default 16
            )
        if not beam_search:
            self.sampling_params = SamplingParams(
                n=1,  # num_return_sequences=repeat_n,
                max_tokens=out_max_len,  # max_new_tokens=128,
                temperature=temp,  # 0 趋向于高频词，容易重复， 1< 原始 softmax, >1 区域均匀，随机
                top_k=20,  # 基于topK单词概率约束候选范围
                length_penalty=1,  # <0 鼓励长句子， >0 鼓励短句子
            )
        else:
            self.sampling_params = SamplingParams(
                use_beam_search=True,
                best_of=3,  # >1
                temperature=temp,
                top_p=1,
                top_k=-1,
                max_tokens=out_max_len,
                length_penalty=0,  # <0 鼓励长句子， >0 鼓励短句子
                n=1,  # num_return_sequences=repeat_n,
            )

    def set_sampling_params(self, **kwargs):
        self.sampling_params = SamplingParams(**kwargs)

    def infer_vllm(self, inputs, instruction=None, chat=False):
        assert isinstance(inputs, list)
        prompt_queries = []

        if instruction is None:
            instruction = 'you are a helpful assistant.'

        for x in inputs:
            if chat:
                message = [{"role": "system", "content": instruction},
                           {"role": "user", "content": x}]
            else:
                message = x
            prompt_queries.append(message)

        if chat:
            tokenizer = self.llm.get_tokenizer()
            model_config = self.llm.llm_engine.get_model_config()
            from vllm.entrypoints.chat_utils import apply_hf_chat_template, parse_chat_messages

            # https://hugging-face.cn/docs/transformers/chat_templating
            def apply_chat(x):
                conversation, _ = parse_chat_messages(x, model_config, tokenizer)
                prompt = apply_hf_chat_template(tokenizer, conversation=conversation,
                                                # 用于继续回答   {"role": "assistant", "content": '{"name": "'},
                                                # continue_final_message=True,
                                                # 用于回答新对话
                                                add_generation_prompt=True,  # TODO
                                                chat_template=None)
                return prompt

            inputs = [apply_chat(x) for x in prompt_queries]

            outputs = self.llm.generate(
                inputs,
                self.sampling_params,
                use_tqdm=True,
                lora_request=LoRARequest("adapter", 1, self.lora_path) if self.lora_path else None,
            )
        else:
            outputs = self.llm.generate(
                prompt_queries,
                self.sampling_params,
                use_tqdm=True,
                lora_request=LoRARequest("adapter", 1, self.lora_path) if self.lora_path else None,
            )

        ret = []
        for output in outputs:
            ret.extend([x.text for x in output.outputs])
        return ret

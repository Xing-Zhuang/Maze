import os
import gc
import math
import torch
import numpy as np
from typing import List, Dict, Any

# Note: agentos decorators are kept as comments to show their original placement.
# from agentos.scheduler import cpu, gpu, io

def task1_start_receive_task(args, dag_id, question, supplementary_files):
    """Task 1: 接收任务输入，初始化基本参数。"""
    import time
    start_time= time.time()
    print("✅ Task 1: Starting... Verifying initial parameters.")
    target_language = "en"
    if "chinese" in question.lower() or "中文" in question.lower(): target_language = "zh"
    elif "german" in question.lower() or "deutsch" in question.lower(): target_language = "de"
    
    print(f"  -> Received dag_id: {dag_id}, target_language: {target_language}")
    return {"dag_id": dag_id, "question": question, "supplementary_files": supplementary_files, "target_language": target_language, "args": args, "start_time": start_time, "end_time": time.time()}

def task2_read_file_and_split_questions(args, dag_id, question, supplementary_files):
    """Task 2: 读取主文档和问题，并将问题分批。"""
    import time
    start_time= time.time()
    print("✅ Task 2: Reading document and splitting questions...")
    document_content = supplementary_files['text.txt'].decode('utf-8')
    # 直接将 text.txt 每行作为一个问题
    questions_list = [line.strip() for line in document_content.split('\n') if line.strip()]
    
    num_batches = 3
    batch_size = math.ceil(len(questions_list) / num_batches) if num_batches > 0 else 0
    question_batches = [questions_list[i:i + batch_size] for i in range(0, len(questions_list), batch_size)] if batch_size > 0 else []
    
    print(f"  -> Read document ({len(document_content)} chars), split {len(questions_list)} questions into {len(question_batches)} batches.")
    return {"dag_id": dag_id, "document_content": document_content, "question_batches": question_batches, "args": args, "start_time": start_time, "end_time": time.time()}

def task3_language_detect(args, dag_id, document_content):
    """Task 3: 对文档内容进行语言检测。"""
    import langid
    import time
    start_time= time.time()
    print("✅ Task 3: Detecting document language...")
    lang_code, confidence = langid.classify(document_content)
    print(f"  -> Language detected: {lang_code} (Confidence: {confidence:.2f})")
    return {"dag_id": dag_id, "source_language": lang_code, "args": args, "start_time": start_time, "end_time": time.time()}

def task4_translate_text(args, dag_id, document_content, source_language, target_language):
    """Task 4: 如有需要，翻译文档内容。"""
    from transformers import MarianMTModel, MarianTokenizer
    import time
    start_time= time.time()
    print("✅ Task 4: Translating document...")
    
    if source_language == target_language or not document_content:
        print("  -> Source and target language are the same or content is empty, skipping translation.")
        return {"dag_id": dag_id, "translated_text": document_content, "args": args}
    
    model, tokenizer = None, None
    try:
        print(f"  -> Translating from {source_language} to {target_language}...")
        model_name = f"Helsinki-NLP/opus-mt-{source_language}-{target_language}"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name).to("cuda")
        inputs = tokenizer(document_content, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
        translated_ids = model.generate(**inputs)
        translated_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
        print("  -> Translation complete.")
        return {"dag_id": dag_id, "translated_text": translated_text, "args": args, "start_time": start_time, "end_time": time.time()}
    except Exception as e:
        print(f"❌ Task 4 (Translation) failed: {e}. Passing original text.")
        return {"dag_id": dag_id, "translated_text": document_content, "args": args, "start_time": start_time, "end_time": time.time()} # Fallback
    finally:
        del model, tokenizer, inputs, translated_ids, translated_text; gc.collect(); torch.cuda.empty_cache()

def task5a_text_analysis_summarize(args, dag_id, translated_text):
    """Task 5a (并行): 对翻译后的文本进行摘要。"""
    from transformers import MarianMTModel, MarianTokenizer, pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
    import time
    start_time= time.time()
    print("✅ Task 5a: Summarizing text...")
    try:
        model_path = os.path.join(args.model_folder, "t5-base")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="cuda"
        )
        inputs = tokenizer(
            "summarize: " + translated_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(model.device)
        outputs = model.generate(
            **inputs,
            max_length=150,
            min_length=30,
            num_beams=4,
            early_stopping=True
        )
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  -> Summarization complete.")
        return {"dag_id": dag_id, "summary": summary, "args": args, "start_time": start_time, "end_time": time.time()}
    except Exception as e:
        print(f"❌ Task 5a (Summarization) failed: {e}")
        return {"dag_id": dag_id, "summary": f"Error in summarization: {e}", "args": args, "start_time": start_time, "end_time": time.time()} # Fallback
    finally:
        del model, tokenizer, inputs, outputs, translated_text, summary; gc.collect(); torch.cuda.empty_cache()

def task5b_text_analysis_sentiment(args, dag_id, translated_text):
    """Task 5b (并行): 对翻译后的文本进行情感分析。"""
    from transformers import pipeline
    import time
    import gc, time
    start_time= time.time()
    print("✅ Task 5b: Analyzing text sentiment...")
    try:
        model_path = os.path.join(args.model_folder, "nlptown", "bert-base-multilingual-uncased-sentiment")
        sentiment_analyzer = pipeline(
            "sentiment-analysis", 
            model=model_path, 
            device="cuda"
        )
        chunks = [translated_text[i:i + 512] for i in range(0, len(translated_text), 512)]
        if not chunks: chunks = [""]
        results = sentiment_analyzer(chunks)
        avg_score = np.mean([res['score'] for res in results])
        labels = [res['label'] for res in results]
        main_label = max(set(labels), key=labels.count)
        sentiment = f"Label: {main_label}, Score: {float(avg_score):.2f}"
        print(f"  -> Sentiment analysis complete. {sentiment}")
        del sentiment_analyzer, results; gc.collect(); torch.cuda.empty_cache()
        return {"dag_id": dag_id, "sentiment": sentiment, "args": args, "start_time": start_time, "end_time": time.time()}
    except Exception as e:
        print(f"❌ Task 5b (Sentiment) failed: {e}")
        return {"dag_id": dag_id, "sentiment": "Error in sentiment analysis.", "args": args, "start_time": start_time, "end_time": time.time()} # Fallback

def task6_prepare_llm_batches(args, dag_id, translated_text, summary, sentiment, question_batches, question):
    """Task 6 (合并点): 准备最终的LLM输入批次。"""
    import time
    start_time= time.time()
    print("✅ Task 6: Preparing final LLM batches...")
    all_messages = []
    for batch in question_batches:
        batch_messages = []
        for q in batch:
            # 英文提示词，只要求填充MASK
            prompt = (
                "Please only output the word or phrase that should fill the [MASK] in the English sentence below. Do not output anything else.\n"
                f"Sentence: {q}\n"
                "Answer:"
            )
            batch_messages.append([{"role": "user", "content": prompt}])
        all_messages.append(batch_messages)
    
    print(f"  -> Prepared {len(all_messages)} batches for parallel LLM processing.")
    return {"dag_id": dag_id, "messages_batch_1": all_messages[0] if len(all_messages) > 0 else [],
            "messages_batch_2": all_messages[1] if len(all_messages) > 1 else [],
            "messages_batch_3": all_messages[2] if len(all_messages) > 2 else [], "args": args, "start_time": start_time, "end_time": time.time()}

def task7a_llm_process_batch_1(args, dag_id, messages_batch_1, vllm_manager= None, backend= "huggingface"):
    """Task 7a (并行): 处理第1批问题。"""
    import time
    import asyncio
    import aiohttp
    from typing import List, Dict, Any, Optional, Tuple
    start_time= time.time()
    async def _query_vllm_batch_async(
        api_url: str,
        model_alias: str,
        messages_list: List[List[Dict[str, str]]],
        temperature: float,
        max_token: int,
        top_p: float,
        repetition_penalty: float
    ) -> List[str]:
        """[新增] 使用 aiohttp 并发执行所有vLLM文本请求的异步核心。"""
        chat_url = f"{api_url.strip('/')}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        
        async def _query_single(session: aiohttp.ClientSession, messages: List[Dict[str, str]]) -> str:
            payload = {
                "model": model_alias,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_token,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
            }
            try:
                async with session.post(chat_url, json=payload, timeout=3600) as response:
                    response.raise_for_status()
                    response_data = await response.json()
                    return response_data['choices'][0]['message']['content'].strip()
            except Exception as e:
                error_msg = f"vLLM async request failed: {str(e)}"
                print(f"❌ {error_msg}")
                return error_msg

        async with aiohttp.ClientSession(headers=headers) as session:
            tasks = [_query_single(session, messages) for messages in messages_list]
            all_responses = await asyncio.gather(*tasks, return_exceptions=True)
            return [str(resp) for resp in all_responses]

    def query_vllm_batch_via_service(
        api_url: str,
        model_alias: str,
        messages_list: List[List[Dict[str, str]]],
        temperature: float = 0.6,
        max_token: int = 1024,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        **kwargs # 添加**kwargs以忽略不用的参数如model_folder, batch_size
    ) -> Tuple[Dict, List[str]]:
        """
        [新增] 使用vLLM服务批量处理纯文本生成任务。
        - 通过并发API请求实现批处理。
        - 接口与现有的 query_llm_batch 完全一致。
        """
        if not messages_list:
            return {"text_length": 0.0, "token_count": 0.0, "batch_size": 0}, []
        print(f"  -> ⚡️ Starting vLLM batch processing via SERVICE: {len(messages_list)} prompts concurrently.")
        
        batch_answers = asyncio.run(_query_vllm_batch_async(
            api_url, model_alias, messages_list, temperature, max_token, top_p, repetition_penalty
        ))
        print("  -> ✅ vLLM batch processing via service finished.")
        return batch_answers
    
    def _llm_batch_worker(task_name: str, args, messages_list: List[List[Dict[str, str]]]):
        """
        A pure, self-contained worker for processing a batch of LLM requests.
        It encapsulates all dependencies, including model loading and resource cleanup.
        """
        from transformers import AutoTokenizer, AutoModelForCausalLM

        if not messages_list:
            print(f"✅ Task {task_name}: Input batch is empty, skipping.")
            return []

        print(f"⚙️  Task {task_name}: Starting LLM processing for {len(messages_list)} prompts...")
        model, tokenizer = None, None
        try:
            model_path = os.path.join(args.model_folder, "Qwen/Qwen3-32B")
            tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left', trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True,
                device_map="cuda", offload_state_dict= False, trust_remote_code=True
            )
            
            prompts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list]
            all_responses = []
            
            batch_size = getattr(args, "text_batch_size", 1)
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i + batch_size]
                inputs = tokenizer(batch_prompts, return_tensors='pt', padding=True).to(model.device)
                outputs = model.generate(
                    **inputs, max_new_tokens=args.max_token, temperature=args.temperature,
                    pad_token_id=tokenizer.eos_token_id, top_p=args.top_p, repetition_penalty=args.repetition_penalty
                )
                # Decode only the newly generated tokens
                responses = tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                all_responses.extend(responses)
                del inputs, outputs

            print(f"✅ Task {task_name}: Processing complete. Generated {len(all_responses)} answers.")
            return all_responses
        finally:
            del model, tokenizer; gc.collect(); torch.cuda.empty_cache()
            print(f"  -> Task {task_name}: LLM resources released.")
    try:
        if backend == "vllm":
            answers = query_vllm_batch_via_service(vllm_manager.get_next_endpoint("qwen3-32b"), "qwen3-32b", messages_batch_1, args.temperature, args.max_token, args.top_p, args.repetition_penalty)
        else:
            answers = _llm_batch_worker("7a", args, messages_batch_1)
        return {"dag_id": dag_id, "answers_a": answers, "start_time": start_time, "end_time": time.time()}
    except Exception as e:
        print(f"❌ Task 7a failed: {e}")
        return {"dag_id": dag_id, "answers_a": None, "start_time": start_time, "end_time": time.time()}

def task7b_llm_process_batch_2(args, dag_id, messages_batch_2, vllm_manager= None, backend= "huggingface"):
    """Task 7b (并行): 处理第2批问题。"""
    import time
    import asyncio
    import aiohttp
    from typing import List, Dict, Any, Optional, Tuple
    start_time= time.time()
    async def _query_vllm_batch_async(
        api_url: str,
        model_alias: str,
        messages_list: List[List[Dict[str, str]]],
        temperature: float,
        max_token: int,
        top_p: float,
        repetition_penalty: float
    ) -> List[str]:
        """[新增] 使用 aiohttp 并发执行所有vLLM文本请求的异步核心。"""
        chat_url = f"{api_url.strip('/')}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        
        async def _query_single(session: aiohttp.ClientSession, messages: List[Dict[str, str]]) -> str:
            payload = {
                "model": model_alias,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_token,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
            }
            try:
                async with session.post(chat_url, json=payload, timeout=3600) as response:
                    response.raise_for_status()
                    response_data = await response.json()
                    return response_data['choices'][0]['message']['content'].strip()
            except Exception as e:
                error_msg = f"vLLM async request failed: {str(e)}"
                print(f"❌ {error_msg}")
                return error_msg

        async with aiohttp.ClientSession(headers=headers) as session:
            tasks = [_query_single(session, messages) for messages in messages_list]
            all_responses = await asyncio.gather(*tasks, return_exceptions=True)
            return [str(resp) for resp in all_responses]

    def query_vllm_batch_via_service(
        api_url: str,
        model_alias: str,
        messages_list: List[List[Dict[str, str]]],
        temperature: float = 0.6,
        max_token: int = 1024,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        **kwargs # 添加**kwargs以忽略不用的参数如model_folder, batch_size
    ) -> Tuple[Dict, List[str]]:
        """
        [新增] 使用vLLM服务批量处理纯文本生成任务。
        - 通过并发API请求实现批处理。
        - 接口与现有的 query_llm_batch 完全一致。
        """
        if not messages_list:
            return {"text_length": 0.0, "token_count": 0.0, "batch_size": 0}, []
        print(f"  -> ⚡️ Starting vLLM batch processing via SERVICE: {len(messages_list)} prompts concurrently.")
        
        batch_answers = asyncio.run(_query_vllm_batch_async(
            api_url, model_alias, messages_list, temperature, max_token, top_p, repetition_penalty
        ))
        print("  -> ✅ vLLM batch processing via service finished.")
        return batch_answers
    
    def _llm_batch_worker(task_name: str, args, messages_list: List[List[Dict[str, str]]]):
        """
        A pure, self-contained worker for processing a batch of LLM requests.
        It encapsulates all dependencies, including model loading and resource cleanup.
        """
        import time
        from transformers import AutoTokenizer, AutoModelForCausalLM

        if not messages_list:
            print(f"✅ Task {task_name}: Input batch is empty, skipping.")
            return []

        print(f"⚙️  Task {task_name}: Starting LLM processing for {len(messages_list)} prompts...")
        model, tokenizer = None, None
        try:
            model_path = os.path.join(args.model_folder, "Qwen/Qwen3-32B")
            tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left', trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True,
                device_map="cuda", offload_state_dict= False, trust_remote_code=True
            )
            
            prompts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list]
            all_responses = []
            
            batch_size = getattr(args, "text_batch_size", 1)
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i + batch_size]
                inputs = tokenizer(batch_prompts, return_tensors='pt', padding=True).to(model.device)
                outputs = model.generate(
                    **inputs, max_new_tokens=args.max_token, temperature=args.temperature,
                    pad_token_id=tokenizer.eos_token_id, top_p=args.top_p, repetition_penalty=args.repetition_penalty
                )
                # Decode only the newly generated tokens
                responses = tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                all_responses.extend(responses)
                del inputs, outputs

            print(f"✅ Task {task_name}: Processing complete. Generated {len(all_responses)} answers.")
            return all_responses
        finally:
            del model, tokenizer; gc.collect(); torch.cuda.empty_cache()
            print(f"  -> Task {task_name}: LLM resources released.")
    try:
        if backend == "vllm":
            answers = query_vllm_batch_via_service(vllm_manager.get_next_endpoint("qwen3-32b"), "qwen3-32b", messages_batch_2, args.temperature, args.max_token, args.top_p, args.repetition_penalty)
        else:
            answers = _llm_batch_worker("7b", args, messages_batch_2)
        return {"dag_id": dag_id, "answers_b": answers, "start_time": start_time, "end_time": time.time()}
    except Exception as e:
        print(f"❌ Task 7b failed: {e}")
        return {"dag_id": dag_id, "answers_b": None, "start_time": start_time, "end_time": time.time()}

def task7c_llm_process_batch_3(args, dag_id, messages_batch_3, vllm_manager= None, backend= "huggingface"):
    """Task 7c (并行): 处理第3批问题。"""
    import time
    import asyncio
    import aiohttp
    from typing import List, Dict, Any, Optional, Tuple
    start_time= time.time()
    async def _query_vllm_batch_async(
        api_url: str,
        model_alias: str,
        messages_list: List[List[Dict[str, str]]],
        temperature: float,
        max_token: int,
        top_p: float,
        repetition_penalty: float
    ) -> List[str]:
        """[新增] 使用 aiohttp 并发执行所有vLLM文本请求的异步核心。"""
        chat_url = f"{api_url.strip('/')}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        
        async def _query_single(session: aiohttp.ClientSession, messages: List[Dict[str, str]]) -> str:
            payload = {
                "model": model_alias,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_token,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
            }
            try:
                async with session.post(chat_url, json=payload, timeout=3600) as response:
                    response.raise_for_status()
                    response_data = await response.json()
                    return response_data['choices'][0]['message']['content'].strip()
            except Exception as e:
                error_msg = f"vLLM async request failed: {str(e)}"
                print(f"❌ {error_msg}")
                return error_msg

        async with aiohttp.ClientSession(headers=headers) as session:
            tasks = [_query_single(session, messages) for messages in messages_list]
            all_responses = await asyncio.gather(*tasks, return_exceptions=True)
            return [str(resp) for resp in all_responses]

    def query_vllm_batch_via_service(
        api_url: str,
        model_alias: str,
        messages_list: List[List[Dict[str, str]]],
        temperature: float = 0.6,
        max_token: int = 1024,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        **kwargs # 添加**kwargs以忽略不用的参数如model_folder, batch_size
    ) -> Tuple[Dict, List[str]]:
        """
        [新增] 使用vLLM服务批量处理纯文本生成任务。
        - 通过并发API请求实现批处理。
        - 接口与现有的 query_llm_batch 完全一致。
        """
        if not messages_list:
            return {"text_length": 0.0, "token_count": 0.0, "batch_size": 0}, []
        print(f"  -> ⚡️ Starting vLLM batch processing via SERVICE: {len(messages_list)} prompts concurrently.")
        
        batch_answers = asyncio.run(_query_vllm_batch_async(
            api_url, model_alias, messages_list, temperature, max_token, top_p, repetition_penalty
        ))
        print("  -> ✅ vLLM batch processing via service finished.")
        return batch_answers
    
    def _llm_batch_worker(task_name: str, args, messages_list: List[List[Dict[str, str]]]):
        """
        A pure, self-contained worker for processing a batch of LLM requests.
        It encapsulates all dependencies, including model loading and resource cleanup.
        """
        import time
        from transformers import AutoTokenizer, AutoModelForCausalLM
        if not messages_list:
            print(f"✅ Task {task_name}: Input batch is empty, skipping.")
            return []

        print(f"⚙️  Task {task_name}: Starting LLM processing for {len(messages_list)} prompts...")
        model, tokenizer = None, None
        try:
            model_path = os.path.join(args.model_folder, "Qwen/Qwen3-32B")
            tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left', trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True,
                device_map="cuda", offload_state_dict= False, trust_remote_code=True
            )
            
            prompts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list]
            all_responses = []
            
            batch_size = getattr(args, "text_batch_size", 1)
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i + batch_size]
                inputs = tokenizer(batch_prompts, return_tensors='pt', padding=True).to(model.device)
                outputs = model.generate(
                    **inputs, max_new_tokens=args.max_token, temperature=args.temperature,
                    pad_token_id=tokenizer.eos_token_id, top_p=args.top_p, repetition_penalty=args.repetition_penalty
                )
                # Decode only the newly generated tokens
                responses = tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                all_responses.extend(responses)
                del inputs, outputs

            print(f"✅ Task {task_name}: Processing complete. Generated {len(all_responses)} answers.")
            return all_responses
        finally:
            del model, tokenizer; gc.collect(); torch.cuda.empty_cache()
            print(f"  -> Task {task_name}: LLM resources released.")
    try:
        if backend == "vllm":
            answers = query_vllm_batch_via_service(vllm_manager.get_next_endpoint("qwen3-32b"), "qwen3-32b", messages_batch_3, args.temperature, args.max_token, args.top_p, args.repetition_penalty)
        else:
            answers = _llm_batch_worker("7c", args, messages_batch_3)
        return {"dag_id": dag_id, "answers_c": answers, "start_time": start_time, "end_time": time.time()}
    except Exception as e:
        print(f"❌ Task 7c failed: {e}")
        return {"dag_id": dag_id, "answers_c": None, "start_time": start_time, "end_time": time.time()}

def task8_merge_answers(args, dag_id, answers_a, answers_b, answers_c):
    """Task 8 (合并点): 合并所有并行的LLM答案。"""
    import time
    start_time= time.time()
    print("✅ Task 8: Merging all answers from LLM batches...")
    final_answers = (answers_a or []) + (answers_b or []) + (answers_c or [])
    print(f"✅ Task 8: Merged a total of {len(final_answers)} answers.")
    return {"dag_id": dag_id, "final_answers": final_answers, "args": args, "start_time": start_time, "end_time": time.time()}

def task9_output_final_answer(args, dag_id, final_answers):
    """Task 9: 输出最终答案。"""
    import time
    start_time= time.time()
    print("✅ Task 9: Formatting final output.")
    final_answer_text = '\n\n---\n\n'.join(final_answers) if final_answers else "No answers were generated."
    print(f"🏁 Final Answer for DAG {dag_id} generated.")
    return {"dag_id": dag_id, "final_answer": final_answer_text, "start_time": start_time, "end_time": time.time()}

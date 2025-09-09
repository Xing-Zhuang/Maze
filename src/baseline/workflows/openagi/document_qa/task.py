def task1_start_receive_task(args, dag_id, question, supplementary_files):
    """Task 1: 接收任务，确认核心参数。"""
    import time
    start_time = time.time()
    try:
        print("✅ Task 1: Starting... Verifying initial parameters.")
        print(f"  -> Received dag_id: {dag_id}, question: '{question[:100]}...', {len(supplementary_files)} files.")
        # 读取 questions.txt 内容，作为 questions
        questions = supplementary_files['questions.txt'].decode('utf-8') if 'questions.txt' in supplementary_files else ""
        print(f"Loaded {len(questions.splitlines())} questions from questions.txt")
        end_time = time.time()
        return {
            "dag_id": dag_id,
            "question": question,
            "questions": questions,  # 新增字段
            "supplementary_files": supplementary_files,
            "args": args,
            "start_time": start_time,
            "end_time": end_time
        }
    except Exception as e:
        end_time = time.time()
        return {
            "dag_id": dag_id, "question": None, "questions": None, "supplementary_files": None, "args": args,
            "start_time": start_time, "end_time": end_time
        }

def task2_read_file(args, dag_id, supplementary_files):
    """Task 2: 读取文件内容。"""
    import time
    start_time = time.time()
    try:
        print("✅ Task 2: Reading file content...")
        if not supplementary_files or 'context.txt' not in supplementary_files:
            raise ValueError("supplementary_files is None or 'context.txt' is missing.")
            
        document_content = supplementary_files['context.txt'].decode('utf-8')
        print(f"✅ Task 2: Finished. Text content length: {len(document_content)}")
        end_time = time.time()
        return {
            "dag_id": dag_id, "document_content": document_content, "args": args,
            "start_time": start_time, "end_time": end_time
        }
    except Exception as e:
        end_time = time.time()
        return {
            "dag_id": dag_id, "document_content": None, "args": args,
            "start_time": start_time, "end_time": end_time
        }

def task3a_extract_text_content(args, dag_id, document_content):
    """Task 3a: (并行) 文本内容标准化处理 (CPU密集型)。"""
    import time
    start_time = time.time()
    try:
        print("✅ Task 3a: Normalizing extracted text content...")
        if not document_content:
            raise ValueError("document_content is None or empty.")
            
        unique_lines = [line.strip() for line in document_content.split('\n') if line.strip()]
        processed_text = '\n'.join(dict.fromkeys(unique_lines)) # 去重并保持顺序
        
        print(f"✅ Task 3a: Text normalization complete. Length from {len(document_content)} to {len(processed_text)}.")
        end_time = time.time()
        return {
            "dag_id": dag_id, "extracted_text": processed_text, "args": args,
            "start_time": start_time, "end_time": end_time
        }
    except Exception as e:
        end_time = time.time()
        return {
            "dag_id": dag_id, "extracted_text": None, "args": args,
            "start_time": start_time, "end_time": end_time
        }

def task3b_llm_process_extract_structure_info(args, dag_id, document_content, vllm_manager= None, backend= "huggingface"):
    """Task 3b: (并行) 使用LLM分析文档结构。"""
    import os, gc, time, torch
    import asyncio
    import aiohttp
    from typing import Optional, Dict, List, Tuple, Any
    start_time = time.time()
    async def _query_single_vllm_endpoint(
        session: aiohttp.ClientSession,
        chat_url: str,
        payload: Dict[str, Any]
    ) -> str:
        """异步发送单个请求到vLLM的coroutine。"""
        try:
            async with session.post(chat_url, json=payload, timeout=3600) as response:
                response.raise_for_status()
                response_data = await response.json()
                return response_data['choices'][0]['message']['content'].strip()
        except Exception as e:
            error_msg = f"vLLM async request failed: {str(e)}"
            print(f"[bold red]{error_msg}")
            return error_msg

    async def _query_vllm_batch_async(
        api_url: str,
        model_alias: str,
        messages_list: List[List[Dict[str, str]]],
        temperature: float,
        max_token: int,
        top_p: float,
        repetition_penalty: float
    ) -> List[str]:
        """使用 aiohttp 并发执行所有vLLM请求。"""
        chat_url = f"{api_url.strip('/')}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        
        async with aiohttp.ClientSession(headers=headers) as session:
            tasks = []
            for messages in messages_list:
                payload = {
                    "model": model_alias,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_token,
                    "top_p": top_p,
                    "repetition_penalty": repetition_penalty,
                }
                tasks.append(_query_single_vllm_endpoint(session, chat_url, payload))
            
            # 并发执行所有请求
            all_responses = await asyncio.gather(*tasks)
            return all_responses

    def query_vllm_batch(
        api_url: str,
        model_alias: str,
        messages_list: List[List[Dict[str, str]]],
        temperature: float = 0.6,
        max_token: int = 1024,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1
    ) -> Tuple[Dict, List[str]]:
        """
        [新增] 使用vLLM服务批量处理文本生成任务。
        - 利用asyncio和aiohttp实现高并发请求，达到批量处理的效果。
        """
        print(f"  -> Starting vLLM batch processing: {len(messages_list)} prompts concurrently.")
        
        # 运行异步主函数
        batch_answers = asyncio.run(_query_vllm_batch_async(
            api_url, model_alias, messages_list, temperature, max_token, top_p, repetition_penalty
        ))
        print("  -> vLLM batch processing finished.")
        return batch_answers

    def _query_llm_single(model_folder, model_name, messages, temperature, max_token, top_p, repetition_penalty):
        """A simplified LLM query for a single prompt."""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        model, tokenizer = None, None
        try:
            tokenizer_path = os.path.join(model_folder, "Qwen/Qwen3-32B")
            model_path = os.path.join(model_folder, model_name)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="cuda", offload_state_dict= False,trust_remote_code=True)
            
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=max_token, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty)
            return tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
        finally:
            del model
            del tokenizer
            del inputs
            del outputs
            gc.collect()
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_max_memory_allocated()
                torch.cuda.reset_peak_memory_stats()

    try:
        print("✅ Task 3b: Analyzing document structure with LLM...")
        if not document_content:
            raise ValueError("document_content is None or empty.")
        
        structure_prompt = f"Please analyze the structure of the following document and provide a brief summary.\n\nDocument (first 3000 chars):\n{document_content[:3000]}"
        messages = [{"role": "user", "content": structure_prompt}]
        
        if backend == "vllm":
            batch_results = query_vllm_batch(
                api_url= vllm_manager.get_next_endpoint("qwen3-32b"),
                model_alias= "qwen3-32b",
                messages_list= [messages],
                temperature= args.temperature,
                max_token= args.max_token,
                top_p= args.top_p,
                repetition_penalty= args.repetition_penalty,
            )
            
            # 2. 安全性检查：确保列表不是空的
            if not batch_results:
                raise ValueError("vLLM aPI call returned an empty or invalid result.")
                
            # 3. 安全地提取第一个元素
            structure_summary = batch_results[0]
        else:
            structure_summary = _query_llm_single(
                model_folder=args.model_folder, model_name="Qwen/Qwen3-32B", messages=messages,
                temperature=args.temperature, max_token=args.max_token, top_p=args.top_p, repetition_penalty=args.repetition_penalty
            )
        
        print("✅ Task 3b: Structure analysis finished.")
        end_time = time.time()
        return {"dag_id": dag_id, "doc_structure": structure_summary, "args": args,
                "start_time": start_time, "end_time": end_time}
    except Exception as e:
        end_time = time.time()
        return {"dag_id": dag_id, "doc_structure": f"Error during structure analysis: {e}", "args": args,
                "start_time": start_time, "end_time": end_time}


def task3c_load_questions_batch(args, dag_id, questions):
    """Task 3c: (并行) 加载问题并分批。"""
    import time
    start_time = time.time()
    try:
        print("✅ Task 3c: Loading and batching questions...")
        if not questions:
            raise ValueError("Questions string is None or empty.")
        
        questions_list = [q.strip() for q in questions.split('\n') if q.strip()]
        print(f"Loaded {len(questions_list)} questions for batching.")
        num_questions = len(questions_list)
        batch1_size = int(0.2 * num_questions)
        batch2_size = int(0.2 * num_questions)
        batches = [
            questions_list[:batch1_size],
            questions_list[batch1_size : batch1_size + batch2_size],
            questions_list[batch1_size + batch2_size:]
        ]
        
        while len(batches) < 3:
            batches.append([])
            
        print(f"✅ Task 3c: Loaded {len(questions_list)} questions into {len(batches)} batches.")
        end_time = time.time()
        return {"dag_id": dag_id, "question_batches": batches, "args": args,
                "start_time": start_time, "end_time": end_time}
    except Exception as e:
        end_time = time.time()
        return {"dag_id": dag_id, "question_batches": None, "args": args,
                "start_time": start_time, "end_time": end_time}


def task4a_merge_document_analysis(args, dag_id, extracted_text, doc_structure):
    """Task 4a: (合并点) 合并文档内容和结构。"""
    import time
    start_time = time.time()
    try:
        print("✅ Task 4a: Merging document analysis results...")
        if not all([extracted_text, doc_structure]):
             raise ValueError("Upstream data (extracted_text or doc_structure) is missing.")

        merged_analysis = {"content": extracted_text, "structure": doc_structure}
        print("✅ Task 4a: Document analysis merged successfully.")
        end_time = time.time()
        return {"dag_id": dag_id, "merged_document_analysis": merged_analysis, "args": args,
                "start_time": start_time, "end_time": end_time}
    except Exception as e:
        end_time = time.time()
        return {"dag_id": dag_id, "merged_document_analysis": None, "args": args,
                "start_time": start_time, "end_time": end_time}

def task4b_prepare_qa_context(args, dag_id, merged_document_analysis, question_batches):
    """Task 4b: (合并点) 准备QA上下文。"""
    import time
    start_time = time.time()
    try:
        print("✅ Task 4b: Preparing final QA context...")
        if not all([merged_document_analysis, question_batches is not None]):
            raise ValueError("Upstream data (merged_document_analysis or question_batches) is missing.")

        qa_context = {
            "document_content": merged_document_analysis["content"][:12000],
            "document_structure": merged_document_analysis["structure"],
            "question_batches": question_batches
        }
        
        print("✅ Task 4b: Final QA context is ready.")
        end_time = time.time()
        return {"dag_id": dag_id, "qa_context": qa_context, "args": args,
                "start_time": start_time, "end_time": end_time}
    except Exception as e:
        end_time = time.time()
        return {"dag_id": dag_id, "qa_context": None, "args": args,
                "start_time": start_time, "end_time": end_time}

def task5a_llm_process_batch_1(args, dag_id, qa_context, vllm_manager= None, backend= "huggingface"):
    """Task 5a: 处理第1批问题。"""
    import time
    start_time = time.time()
    
    def _qa_processing_worker(task_name, args, qa_context, batch_index):
        """一个通用的QA工作函数，处理一个问题批次。它包含了所有依赖项以确保独立性。"""
        import os, gc, time, math, torch
        from typing import Optional, Dict, List, Tuple, Any
        import asyncio
        import aiohttp  
        async def _query_single_vllm_endpoint(
            session: aiohttp.ClientSession,
            chat_url: str,
            payload: Dict[str, Any]
        ) -> str:
            """异步发送单个请求到vLLM的coroutine。"""
            try:
                async with session.post(chat_url, json=payload, timeout=3600) as response:
                    response.raise_for_status()
                    response_data = await response.json()
                    return response_data['choices'][0]['message']['content'].strip()
            except Exception as e:
                error_msg = f"vLLM async request failed: {str(e)}"
                print(f"[bold red]{error_msg}")
                return error_msg
            
        async def _query_vllm_batch_async(
            api_url: str,
            model_alias: str,
            messages_list: List[List[Dict[str, str]]],
            temperature: float,
            max_token: int,
            top_p: float,
            repetition_penalty: float
        ) -> List[str]:
            """使用 aiohttp 并发执行所有vLLM请求。"""
            chat_url = f"{api_url.strip('/')}/v1/chat/completions"
            headers = {"Content-Type": "application/json"}
            
            async with aiohttp.ClientSession(headers=headers) as session:
                tasks = []
                for messages in messages_list:
                    payload = {
                        "model": model_alias,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_token,
                        "top_p": top_p,
                        "repetition_penalty": repetition_penalty,
                    }
                    tasks.append(_query_single_vllm_endpoint(session, chat_url, payload))
                
                # 并发执行所有请求
                all_responses = await asyncio.gather(*tasks)
                return all_responses

        def query_vllm_batch(
            api_url: str,
            model_alias: str,
            messages_list: List[List[Dict[str, str]]],
            temperature: float = 0.6,
            max_token: int = 1024,
            top_p: float = 0.9,
            repetition_penalty: float = 1.1
        ) -> Tuple[Dict, List[str]]:
            """
            [新增] 使用vLLM服务批量处理文本生成任务。
            - 利用asyncio和aiohttp实现高并发请求，达到批量处理的效果。
            """
            print(f"  -> Starting vLLM batch processing: {len(messages_list)} prompts concurrently.")
            
            # 运行异步主函数
            batch_answers = asyncio.run(_query_vllm_batch_async(
                api_url, model_alias, messages_list, temperature, max_token, top_p, repetition_penalty
            ))
            
            print("  -> vLLM batch processing finished.")
            return batch_answers

        def _query_llm_batch(model_folder: str, model_name: str, messages_list: List[List[Dict[str, str]]],
                            temperature: float, max_token: int, top_p: float, repetition_penalty: float, batch_size: int):
            from transformers import AutoTokenizer, AutoModelForCausalLM
            model, tokenizer = None, None
            try:
                print(f"  -> [{task_name}] Loading batch LLM: {model_name}...")
                tokenizer_path = os.path.join(model_folder, "Qwen/Qwen3-32B")
                model_path = os.path.join(model_folder, model_name)
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side='left', trust_remote_code=True)
                if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
                model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="cuda", offload_state_dict= False, trust_remote_code=True)
                
                prompts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list]
                all_responses = []
                for i in range(0, len(prompts), batch_size):
                    batch_prompts = prompts[i:i + batch_size]
                    inputs = tokenizer(batch_prompts, return_tensors='pt', padding=True).to(model.device)
                    outputs = model.generate(**inputs, max_new_tokens=max_token, temperature=temperature, pad_token_id=tokenizer.eos_token_id, top_p=top_p, repetition_penalty=repetition_penalty)
                    all_responses.extend(tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True))
                    del inputs, outputs
                return all_responses
            finally:
                del model, tokenizer
                gc.collect()
                torch.cuda.empty_cache()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.reset_max_memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                print(f"  -> [{task_name}] Batch LLM resources released.")

        print(f"⚙️  Starting QA processing for {task_name}...")
        if not qa_context or batch_index >= len(qa_context["question_batches"]):
            print(f"  -> {task_name} has invalid context or batch index. Skipping.")
            return []

        questions_in_batch = qa_context["question_batches"][batch_index]
        if not questions_in_batch:
            print(f"  -> {task_name} batch is empty. Skipping.")
            return []

        # 新增：将 context 拆行，与问题一一对应
        context_lines = qa_context["document_content"].split('\n')

        messages_list = []
        for idx, q in enumerate(questions_in_batch):
            context_for_this_question = context_lines[idx] if idx < len(context_lines) else ""
            messages_list.append([
                {"role": "user", "content": f"""Based on the following document content, directly answer the question. Only output the most concise answer, do not explain or repeat the question.
Document structure summary:
{qa_context['document_structure']}

Document content:
{context_for_this_question}

Question: {q}
Answer:"""}
            ])
        print(f"  -> {task_name} has {len(messages_list)} questions to process in this batch.")
        if backend == "vllm":
            # 使用vLLM服务进行批量处理
            batch_answers = query_vllm_batch(
                api_url= vllm_manager.get_next_endpoint("qwen3-32b"),
                model_alias= "qwen3-32b",
                messages_list= messages_list,
                temperature= args.temperature,
                max_token= getattr(args, "max_token", 1024),
                top_p= args.top_p,
                repetition_penalty= args.repetition_penalty
            )
        else:
            batch_answers = _query_llm_batch(
                model_folder=args.model_folder, model_name="Qwen/Qwen3-32B", messages_list=messages_list,
                temperature=args.temperature, max_token=getattr(args, "max_token", 1024), top_p=args.top_p,
                repetition_penalty=args.repetition_penalty, batch_size=getattr(args, "text_batch_size", 1)
            )
        print(f"✅ {task_name} finished. Answered {len(batch_answers)} questions.")
        return batch_answers
    try:
        results = _qa_processing_worker("task5a", args, qa_context, 0)
        end_time = time.time()
        return {"dag_id": dag_id, "batch1_answers": results,
                "start_time": start_time, "end_time": end_time}
    except Exception as e:
        end_time = time.time()
        return {"dag_id": dag_id, "batch1_answers": None,
                "start_time": start_time, "end_time": end_time}

def task5b_llm_process_batch_2(args, dag_id, qa_context, vllm_manager= None, backend= "huggingface"):
    """Task 5b: 处理第2批问题。"""
    import time
    from typing import Optional, Dict, List, Tuple, Any
    start_time = time.time()

    def _qa_processing_worker(task_name, args, qa_context, batch_index):
        """一个通用的QA工作函数，处理一个问题批次。它包含了所有依赖项以确保独立性。"""
        import os, gc, time, math, torch
        from typing import List, Dict
        import asyncio
        import aiohttp
        async def _query_single_vllm_endpoint(
            session: aiohttp.ClientSession,
            chat_url: str,
            payload: Dict[str, Any]
        ) -> str:
            """异步发送单个请求到vLLM的coroutine。"""
            try:
                async with session.post(chat_url, json=payload, timeout=3600) as response:
                    response.raise_for_status()
                    response_data = await response.json()
                    return response_data['choices'][0]['message']['content'].strip()
            except Exception as e:
                error_msg = f"vLLM async request failed: {str(e)}"
                print(f"[bold red]{error_msg}")
                return error_msg
            
        async def _query_vllm_batch_async(
            api_url: str,
            model_alias: str,
            messages_list: List[List[Dict[str, str]]],
            temperature: float,
            max_token: int,
            top_p: float,
            repetition_penalty: float
        ) -> List[str]:
            """使用 aiohttp 并发执行所有vLLM请求。"""
            chat_url = f"{api_url.strip('/')}/v1/chat/completions"
            headers = {"Content-Type": "application/json"}
            
            async with aiohttp.ClientSession(headers=headers) as session:
                tasks = []
                for messages in messages_list:
                    payload = {
                        "model": model_alias,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_token,
                        "top_p": top_p,
                        "repetition_penalty": repetition_penalty,
                    }
                    tasks.append(_query_single_vllm_endpoint(session, chat_url, payload))
                
                # 并发执行所有请求
                all_responses = await asyncio.gather(*tasks)
                return all_responses

        def query_vllm_batch(
            api_url: str,
            model_alias: str,
            messages_list: List[List[Dict[str, str]]],
            temperature: float = 0.6,
            max_token: int = 1024,
            top_p: float = 0.9,
            repetition_penalty: float = 1.1
        ) -> Tuple[Dict, List[str]]:
            """
            [新增] 使用vLLM服务批量处理文本生成任务。
            - 利用asyncio和aiohttp实现高并发请求，达到批量处理的效果。
            """
            print(f"  -> Starting vLLM batch processing: {len(messages_list)} prompts concurrently.")
            
            # 运行异步主函数
            batch_answers = asyncio.run(_query_vllm_batch_async(
                api_url, model_alias, messages_list, temperature, max_token, top_p, repetition_penalty
            ))
            
            print("  -> vLLM batch processing finished.")
            return batch_answers

        def _query_llm_batch(model_folder: str, model_name: str, messages_list: List[List[Dict[str, str]]],
                            temperature: float, max_token: int, top_p: float, repetition_penalty: float, batch_size: int):
            from transformers import AutoTokenizer, AutoModelForCausalLM
            model, tokenizer = None, None
            try:
                print(f"  -> [{task_name}] Loading batch LLM: {model_name}...")
                tokenizer_path = os.path.join(model_folder, "Qwen/Qwen3-32B")
                model_path = os.path.join(model_folder, model_name)
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side='left', trust_remote_code=True)
                if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
                model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="cuda", offload_state_dict= False, trust_remote_code=True)
                
                prompts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list]
                all_responses = []
                for i in range(0, len(prompts), batch_size):
                    batch_prompts = prompts[i:i + batch_size]
                    inputs = tokenizer(batch_prompts, return_tensors='pt', padding=True).to(model.device)
                    outputs = model.generate(**inputs, max_new_tokens=max_token, temperature=temperature, pad_token_id=tokenizer.eos_token_id, top_p=top_p, repetition_penalty=repetition_penalty)
                    all_responses.extend(tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True))
                    del inputs, outputs
                return all_responses
            finally:
                del model, tokenizer
                gc.collect()
                torch.cuda.empty_cache()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.reset_max_memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                print(f"  -> [{task_name}] Batch LLM resources released.")

        print(f"⚙️  Starting QA processing for {task_name}...")
        if not qa_context or batch_index >= len(qa_context["question_batches"]):
            print(f"  -> {task_name} has invalid context or batch index. Skipping.")
            return []

        questions_in_batch = qa_context["question_batches"][batch_index]
        if not questions_in_batch:
            print(f"  -> {task_name} batch is empty. Skipping.")
            return []

        # 新增：将 context 拆行，与问题一一对应
        context_lines = qa_context["document_content"].split('\n')

        messages_list = []
        for idx, q in enumerate(questions_in_batch):
            context_for_this_question = context_lines[idx] if idx < len(context_lines) else ""
            messages_list.append([
                {"role": "user", "content": f"""Based on the following document content, directly answer the question. Only output the most concise answer, do not explain or repeat the question.
Document structure summary:
{qa_context['document_structure']}

Document content:
{context_for_this_question}

Question: {q}
Answer:"""}
            ])
        if backend == "vllm":
            # 使用vLLM服务进行批量处理
            batch_answers = query_vllm_batch(
                api_url= vllm_manager.get_next_endpoint("qwen3-32b"),
                model_alias= "qwen3-32b",
                messages_list= messages_list,
                temperature= args.temperature,
                max_token= getattr(args, "max_token", 1024),
                top_p= args.top_p,
                repetition_penalty= args.repetition_penalty
            )
        else:
            batch_answers = _query_llm_batch(
                model_folder=args.model_folder, model_name="Qwen/Qwen3-32B", messages_list=messages_list,
                temperature=args.temperature, max_token=getattr(args, "max_token", 1024), top_p=args.top_p,
                repetition_penalty=args.repetition_penalty, batch_size=getattr(args, "text_batch_size", 1)
            )
        print(f"✅ {task_name} finished. Answered {len(batch_answers)} questions.")
        return batch_answers
    try:
        results = _qa_processing_worker("task5b", args, qa_context, 1)
        end_time = time.time()
        return {"dag_id": dag_id, "batch2_answers": results,
                "start_time": start_time, "end_time": end_time}
    except Exception as e:
        end_time = time.time()
        return {"dag_id": dag_id, "batch2_answers": None,
                "start_time": start_time, "end_time": end_time}

def task5c_llm_process_batch_3(args, dag_id, qa_context, vllm_manager= None, backend= "huggingface"):
    """Task 5c: 处理第3批问题。"""
    import time
    from typing import Optional, Dict, List, Tuple, Any
    start_time = time.time()
    
    def _qa_processing_worker(task_name, args, qa_context, batch_index):
        """一个通用的QA工作函数，处理一个问题批次。它包含了所有依赖项以确保独立性。"""
        import os, gc, time, math, torch
        from typing import List, Dict
        import asyncio
        import aiohttp
        async def _query_single_vllm_endpoint(
            session: aiohttp.ClientSession,
            chat_url: str,
            payload: Dict[str, Any]
        ) -> str:
            """异步发送单个请求到vLLM的coroutine。"""
            try:
                async with session.post(chat_url, json=payload, timeout=3600) as response:
                    response.raise_for_status()
                    response_data = await response.json()
                    return response_data['choices'][0]['message']['content'].strip()
            except Exception as e:
                error_msg = f"vLLM async request failed: {str(e)}"
                print(f"[bold red]{error_msg}")
                return error_msg
            
        async def _query_vllm_batch_async(
            api_url: str,
            model_alias: str,
            messages_list: List[List[Dict[str, str]]],
            temperature: float,
            max_token: int,
            top_p: float,
            repetition_penalty: float
        ) -> List[str]:
            """使用 aiohttp 并发执行所有vLLM请求。"""
            chat_url = f"{api_url.strip('/')}/v1/chat/completions"
            headers = {"Content-Type": "application/json"}
            
            async with aiohttp.ClientSession(headers=headers) as session:
                tasks = []
                for messages in messages_list:
                    payload = {
                        "model": model_alias,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_token,
                        "top_p": top_p,
                        "repetition_penalty": repetition_penalty,
                    }
                    tasks.append(_query_single_vllm_endpoint(session, chat_url, payload))
                
                # 并发执行所有请求
                all_responses = await asyncio.gather(*tasks)
                return all_responses

        def query_vllm_batch(
            api_url: str,
            model_alias: str,
            messages_list: List[List[Dict[str, str]]],
            temperature: float = 0.6,
            max_token: int = 1024,
            top_p: float = 0.9,
            repetition_penalty: float = 1.1
        ) -> Tuple[Dict, List[str]]:
            """
            [新增] 使用vLLM服务批量处理文本生成任务。
            - 利用asyncio和aiohttp实现高并发请求，达到批量处理的效果。
            """
            print(f"  -> Starting vLLM batch processing: {len(messages_list)} prompts concurrently.")
            
            # 运行异步主函数
            batch_answers = asyncio.run(_query_vllm_batch_async(
                api_url, model_alias, messages_list, temperature, max_token, top_p, repetition_penalty
            ))
            
            print("  -> vLLM batch processing finished.")
            return batch_answers

        def _query_llm_batch(model_folder: str, model_name: str, messages_list: List[List[Dict[str, str]]],
                            temperature: float, max_token: int, top_p: float, repetition_penalty: float, batch_size: int):
            from transformers import AutoTokenizer, AutoModelForCausalLM
            model, tokenizer = None, None
            try:
                print(f"  -> [{task_name}] Loading batch LLM: {model_name}...")
                tokenizer_path = os.path.join(model_folder, "Qwen/Qwen3-32B")
                model_path = os.path.join(model_folder, model_name)
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side='left', trust_remote_code=True)
                if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
                model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="cuda", offload_state_dict= False, trust_remote_code=True)
                
                prompts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list]
                all_responses = []
                for i in range(0, len(prompts), batch_size):
                    batch_prompts = prompts[i:i + batch_size]
                    inputs = tokenizer(batch_prompts, return_tensors='pt', padding=True).to(model.device)
                    outputs = model.generate(**inputs, max_new_tokens=max_token, temperature=temperature, pad_token_id=tokenizer.eos_token_id, top_p=top_p, repetition_penalty=repetition_penalty)
                    all_responses.extend(tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True))
                    del inputs, outputs
                return all_responses
            finally:
                del model, tokenizer
                gc.collect()
                torch.cuda.empty_cache()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.reset_max_memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                print(f"  -> [{task_name}] Batch LLM resources released.")

        print(f"⚙️  Starting QA processing for {task_name}...")
        if not qa_context or batch_index >= len(qa_context["question_batches"]):
            print(f"  -> {task_name} has invalid context or batch index. Skipping.")
            return []

        questions_in_batch = qa_context["question_batches"][batch_index]
        if not questions_in_batch:
            print(f"  -> {task_name} batch is empty. Skipping.")
            return []

        # 新增：将 context 拆行，与问题一一对应
        context_lines = qa_context["document_content"].split('\n')

        messages_list = []
        for idx, q in enumerate(questions_in_batch):
            context_for_this_question = context_lines[idx] if idx < len(context_lines) else ""
            messages_list.append([
                {"role": "user", "content": f"""Based on the following document content, directly answer the question. Only output the most concise answer, do not explain or repeat the question.
Document structure summary:
{qa_context['document_structure']}

Document content:
{context_for_this_question}

Question: {q}
Answer:"""}
            ])
        if backend == "vllm":
            # 使用vLLM服务进行批量处理
            batch_answers = query_vllm_batch(
                api_url= vllm_manager.get_next_endpoint("qwen3-32b"),
                model_alias= "qwen3-32b",
                messages_list= messages_list,
                temperature= args.temperature,
                max_token= getattr(args, "max_token", 1024),
                top_p= args.top_p,
                repetition_penalty= args.repetition_penalty
            )
        else:
            batch_answers = _query_llm_batch(
                model_folder=args.model_folder, model_name="Qwen/Qwen3-32B", messages_list=messages_list,
                temperature=args.temperature, max_token=getattr(args, "max_token", 1024), top_p=args.top_p,
                repetition_penalty=args.repetition_penalty, batch_size=getattr(args, "text_batch_size", 1)
            )
        print(f"✅ {task_name} finished. Answered {len(batch_answers)} questions.")
        return batch_answers
    try:
        results = _qa_processing_worker("task5c", args, qa_context, 2)
        end_time = time.time()
        return {"dag_id": dag_id, "batch3_answers": results,
                "start_time": start_time, "end_time": end_time}
    except Exception as e:
        end_time = time.time()
        return {"dag_id": dag_id, "batch3_answers": None,
                "start_time": start_time, "end_time": end_time}

def task7_merge_all_answers(args, dag_id, batch1_answers, batch2_answers, batch3_answers):
    """Task 7: (合并点) 合并所有答案。"""
    import time
    start_time = time.time()
    try:
        print("✅ Task 7: Merging all answers from QA batches...")
        # 安全地合并列表，处理可能为None的情况
        final_answers = (batch1_answers or []) + (batch2_answers or []) + (batch3_answers or [])
        
        print(f"✅ Task 7: Merged a total of {len(final_answers)} answers.")
        end_time = time.time()
        return {"dag_id": dag_id, "final_answers": final_answers, "args": args,
                "start_time": start_time, "end_time": end_time}
    except Exception as e:
        end_time = time.time()
        return {"dag_id": dag_id, "final_answers": None, "args": args,
                "start_time": start_time, "end_time": end_time}

def task8_output_final_answer(args, dag_id, final_answers):
    """Task 8: 输出最终答案。"""
    import time
    start_time = time.time()
    try:
        print("✅ Task 8: Formatting final output.")
        if final_answers is None:
            final_answer_text = "Workflow failed to produce answers due to an upstream error."
        else:
            final_answer_text = '\n'.join(final_answers)
            
        print(f"🏁 Final Answer for DAG {dag_id} generated.")
        end_time = time.time()
        return {"dag_id": dag_id, "final_answer": final_answer_text,
                "start_time": start_time, "end_time": end_time}
    except Exception as e:
        end_time = time.time()
        return {"dag_id": dag_id, "final_answer": f"Error during final output formatting: {e}",
                "start_time": start_time, "end_time": end_time}
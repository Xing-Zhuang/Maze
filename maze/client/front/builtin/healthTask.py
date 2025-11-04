"""
医疗健康相关任务

使用阿里云DashScope实现的医疗诊断辅助功能
"""

from maze.client.front.decorator import task
import os


@task(
    inputs=["tongue_image_path"],
    outputs=["tongue_features"],
    data_types={
        "tongue_image_path": "file:image",
        "tongue_features": "dict"
    },
    resources={"cpu": 2, "cpu_mem": 512, "gpu": 0, "gpu_mem": 0}
)
def analyze_tongue_image(params):
    """
    使用VLM分析舌苔图片特征
    
    输入:
        tongue_image_path: 舌苔图片路径（自动上传到服务器）
        
    输出:
        tongue_features: 提取的舌苔特征（颜色、苔质、形状等）
    """
    import dashscope
    from dashscope import MultiModalConversation
    import base64
    import os
    tongue_image_path = params.get("tongue_image_path")
    api_key = os.getenv("DASHSCOPE_API_KEY")
    
    if not api_key:
        raise ValueError("未找到DASHSCOPE_API_KEY环境变量")
    
    dashscope.api_key = api_key
    
    # 读取图片并转换为base64
    with open(tongue_image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    
    # 使用qwen-vl-max模型分析舌苔图片
    messages = [{
        'role': 'user',
        'content': [
            {'image': f'data:image/jpeg;base64,{image_data}'},
            {'text': '''请作为一名专业的中医师，仔细观察这张舌苔图片，并详细分析以下特征：

1. 舌色（舌体颜色）：淡白、淡红、红、绛红等
2. 舌苔颜色：白、黄、灰、黑等
3. 舌苔厚薄：薄苔、厚苔、无苔等
4. 舌苔润燥：润苔、燥苔
5. 舌形：胖大、瘦薄、正常
6. 舌质：有无裂纹、齿痕、瘀斑等
7. 整体中医辨证分析

请用结构化的JSON格式输出，包含以上各项特征的描述。'''}
        ]
    }]
    
    response = MultiModalConversation.call(
        model='qwen-vl-max',
        messages=messages
    )
    print(response) 
    if response.status_code == 200:
        analysis_text = response.output.choices[0].message.content[0]['text']
        
        tongue_features = {
            "raw_analysis": analysis_text,
            "image_path": tongue_image_path,
            "model": "qwen-vl-max"
        }
        
        return {"tongue_features": tongue_features}
    else:
        raise Exception(f"VLM分析失败: {response.message}")


@task(
    inputs=["symptom_description"],
    outputs=["structured_symptoms"],
    data_types={
        "symptom_description": "str",
        "structured_symptoms": "dict"
    },
    resources={"cpu": 1, "cpu_mem": 256, "gpu": 0, "gpu_mem": 0}
)
def extract_symptoms(params):
    """
    使用LLM从患者描述中提取结构化症状信息
    
    输入:
        symptom_description: 患者的症状描述文本
        
    输出:
        structured_symptoms: 结构化的症状信息
    """
    import dashscope
    from dashscope import Generation
    import json
    import os
    symptom_description = params.get("symptom_description")
    api_key = os.getenv("DASHSCOPE_API_KEY")
    
    if not api_key:
        raise ValueError("未找到DASHSCOPE_API_KEY环境变量")
    
    dashscope.api_key = api_key
    
    prompt = f"""作为一名专业的医疗信息提取专家，请从以下患者症状描述中提取关键信息，并以JSON格式输出：

患者描述：
{symptom_description}

请提取以下信息（若未提及则标注为"未提及"）：
1. 主要症状列表
2. 症状持续时间
3. 症状严重程度
4. 伴随症状
5. 可能的诱因
6. 既往病史（如有提及）
7. 生活习惯相关信息

输出格式示例：
{{
    "main_symptoms": ["症状1", "症状2"],
    "duration": "持续时间",
    "severity": "轻度/中度/重度",
    "accompanying_symptoms": ["伴随症状1", "伴随症状2"],
    "possible_triggers": ["诱因1"],
    "medical_history": "既往病史或未提及",
    "lifestyle_factors": ["相关生活习惯"],
    "summary": "症状总结"
}}

只输出JSON，不要其他内容。"""
    
    response = Generation.call(
        model='qwen-max',
        prompt=prompt,
        result_format='message'
    )
    print(response)
    if response.status_code == 200:
        extracted_text = response.output.choices[0].message.content
        
        # 尝试解析JSON
        try:
            # 清理可能的markdown代码块标记
            if '```json' in extracted_text:
                extracted_text = extracted_text.split('```json')[1].split('```')[0]
            elif '```' in extracted_text:
                extracted_text = extracted_text.split('```')[1].split('```')[0]
            
            structured_data = json.loads(extracted_text.strip())
        except json.JSONDecodeError:
            # 如果解析失败，返回原始文本
            structured_data = {
                "raw_extraction": extracted_text,
                "parse_error": "JSON解析失败，返回原始文本"
            }
        
        structured_symptoms = {
            "original_description": symptom_description,
            "extracted_data": structured_data,
            "model": "qwen-max"
        }
        
        return {"structured_symptoms": structured_symptoms}
    else:
        raise Exception(f"症状提取失败: {response.message}")


@task(
    inputs=["tongue_features", "structured_symptoms"],
    outputs=["medical_advice"],
    data_types={
        "tongue_features": "dict",
        "structured_symptoms": "dict",
        "medical_advice": "dict"
    },
    resources={"cpu": 2, "cpu_mem": 512, "gpu": 0, "gpu_mem": 0}
)
def generate_medical_advice(params):
    """
    结合舌诊和症状，使用联网搜索生成医疗建议
    
    输入:
        tongue_features: 舌苔特征分析结果
        structured_symptoms: 结构化症状信息
        
    输出:
        medical_advice: 综合医疗建议
    """
    import dashscope
    from dashscope import Generation
    import json
    import os
    tongue_features = params.get("tongue_features")
    structured_symptoms = params.get("structured_symptoms")
    api_key = os.getenv("DASHSCOPE_API_KEY")
    
    if not api_key:
        raise ValueError("未找到DASHSCOPE_API_KEY环境变量")
    
    dashscope.api_key = api_key
    
    # 构建综合分析提示词
    prompt = f"""作为一名经验丰富的中西医结合医师，请根据以下信息提供专业的医疗建议：

【舌诊分析】
{json.dumps(tongue_features, ensure_ascii=False, indent=2)}

【症状信息】
{json.dumps(structured_symptoms, ensure_ascii=False, indent=2)}

请提供以下内容（使用JSON格式）：
1. 中医辨证分析（根据舌诊和症状）
2. 西医可能的诊断方向
3. 建议的检查项目
4. 生活调理建议（饮食、作息、运动等）
5. 中医调理建议（可能的中药方剂或穴位）
6. 就医建议（是否需要立即就医、看什么科室）
7. 注意事项

输出格式：
{{
    "tcm_diagnosis": "中医辨证分析",
    "western_diagnosis_direction": ["可能的西医诊断1", "可能的西医诊断2"],
    "recommended_tests": ["建议检查1", "建议检查2"],
    "lifestyle_advice": {{
        "diet": ["饮食建议"],
        "rest": ["作息建议"],
        "exercise": ["运动建议"]
    }},
    "tcm_treatment": {{
        "herbal_formula": "推荐方剂",
        "acupoints": ["穴位1", "穴位2"]
    }},
    "medical_visit": {{
        "urgency": "紧急/尽快/可择期",
        "department": "建议科室",
        "reason": "就医原因"
    }},
    "precautions": ["注意事项1", "注意事项2"],
    "disclaimer": "本建议仅供参考，具体诊疗请遵医嘱"
}}

只输出JSON，不要其他内容。"""
    
    # 使用联网搜索功能增强的qwen-max
    response = Generation.call(
        model='qwen-max',
        prompt=prompt,
        result_format='message',
        enable_search=True  # 启用联网搜索
    )
    print(response)
    if response.status_code == 200:
        advice_text = response.output.choices[0].message.content
        
        # 尝试解析JSON
        try:
            # 清理可能的markdown代码块标记
            if '```json' in advice_text:
                advice_text = advice_text.split('```json')[1].split('```')[0]
            elif '```' in advice_text:
                advice_text = advice_text.split('```')[1].split('```')[0]
            
            advice_data = json.loads(advice_text.strip())
        except json.JSONDecodeError:
            # 如果解析失败，返回原始文本
            advice_data = {
                "raw_advice": advice_text,
                "parse_error": "JSON解析失败，返回原始文本"
            }
        
        medical_advice = {
            "advice_data": advice_data,
            "model": "qwen-max",
            "search_enabled": True,
            "timestamp": __import__('datetime').datetime.now().isoformat()
        }
        
        return {"medical_advice": medical_advice}
    else:
        raise Exception(f"生成医疗建议失败: {response.message}")


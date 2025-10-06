import os
from maze.agent.dag_agent import DAGAgent
from maze.library.tools.file_tools import load_pdf, extract_text_and_tables_from_native_pdf
from maze.library.tools.llm_tools import answer_question_by_analyze_file_content_using_llm, answer_question_by_fuse_llm_output

def create_file_qa_workflow(
    root_path: str= None,
    scheduler_addr: str= None,
    pdf_path: str = "./files/67e8878b-5cef-4375-804e-e6291fdbe78a.pdf",
    question: str = "The attached PDF lists accommodations in the resort community of Seahorse Island. Which type of accommodation has a higher average rating in Seahorse Island?"
):
    """
    Builds and returns the DAG for a file-based Question Answering workflow.
    
    This workflow loads a PDF, extracts its content, uses two parallel LLM calls
    to answer a question based on the content, and then fuses the two answers 
    into a final, superior result.
    
    Args:
        pdf_path (str): The path to the input PDF file.
        question (str): The question to be answered based on the PDF content.
        
    Returns:
        DAGAgent: An agent instance with the constructed workflow ready for execution.
    """
    
    # 1. Initialize the Agent
    agent = DAGAgent(name="GAIA: QA by file content", root_path= root_path, scheduler_addr= scheduler_addr)
    
    # 2. Define the workflow tasks
    task1_id = agent.add_task(
        load_pdf, 
        task_name="load_pdf", 
        inputs={"pdf_path": pdf_path}
    )
    
    task2_id = agent.add_task(
        extract_text_and_tables_from_native_pdf, 
        task_name="extract_text_and_tables_from_native_pdf", 
        inputs={"pdf_content": f"{task1_id}.output.pdf_content"}
    )
    
    # Run two parallel analysis tasks to generate diverse answers for fusion
    task3_id = agent.add_task(
        answer_question_by_analyze_file_content_using_llm,
        task_name="answer_generation_1",
        inputs={
            "content": f"{task2_id}.output.extracted_text",
            "question": question,
            "model": "deepseek-ai/DeepSeek-V3.1"
        }
    )
    
    task4_id = agent.add_task(
        answer_question_by_analyze_file_content_using_llm,
        task_name="answer_generation_2",
        inputs={
            "content": f"{task2_id}.output.extracted_text",
            "question": question,
            "model": "deepseek-ai/DeepSeek-V3.1"
        }
    )
    
    # Fuse the two answers into a final, more robust answer
    task5_id = agent.add_task(
        answer_question_by_fuse_llm_output,
        task_name="fuse_answers",
        inputs={
            "question": question,
            "answer1": f"{task3_id}.output.answer",
            "answer2": f"{task4_id}.output.answer"
        }
    )
    
    return agent
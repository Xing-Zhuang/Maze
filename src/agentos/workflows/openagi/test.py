import argparse
from agentos.utils.query_loader import OpenAGILoader

# 构造参数
args = argparse.Namespace(
    proj_path="/home/hustlbw/AgentOS",
    data_path="data",
    dag_path="dags"
)

# 选择一个存在的 dag_type 和 dag_id
dag_type = "document_qa"
dag_id = "3b6953e7-4acb-4251-8469-3a59b367e4c8"
dag_source = "openagi"
run_id = "test_run"
supplementary_files = []  # 或如有附件可加文件名

loader = OpenAGILoader(
    args=args,
    dag_id=dag_id,
    run_id=run_id,
    dag_type=dag_type,
    dag_source=dag_source,
    supplementary_files=supplementary_files
)

print("问题：", loader.question)
print("答案：", loader.answer)
print("补充文件：", loader.get_supplementary_files())
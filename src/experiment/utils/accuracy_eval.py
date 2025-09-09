import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 路径根据你的实际情况调整
actual_path = '/home/hustlbw/AgentOS/src/agentos/utils/timepred/model/vlm_process/data/vlm_process_data.csv'
pred_path = '/home/hustlbw/AgentOS/src/agentos/utils/timepred/model/vlm_process/data/vlm_process_pred_log.csv'

actual_df = pd.read_csv(actual_path)
pred_df = pd.read_csv(pred_path)

# 合并数据，确保只对有实际和预测时间的任务计算
merged = pd.merge(pred_df, actual_df[['task_id', 'execution_time']], on='task_id', how='inner')

y_true = merged['execution_time']
y_pred = merged['predicted_time']

mae = mean_absolute_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)
r2 = r2_score(y_true, y_pred)

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

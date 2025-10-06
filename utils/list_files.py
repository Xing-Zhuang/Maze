import os
def list_directory_tree(root_dir, indent_char='--', indent_count=1, exclude_extensions=None):
    """
    递归地打印出目录结构。
    参数:
    root_dir (str): 要开始遍历的根目录路径。
    indent_char (str): 用于每一级缩进的字符或字符串，默认为 '--'。
    indent_count (int): 每一级缩进重复 indent_char 的次数，默认为 1。
    """
    if exclude_extensions is None:
        exclude_extensions = []
    if not os.path.isdir(root_dir):
        print(f"错误: '{root_dir}' 不是一个有效的目录。")
        return
    print(os.path.basename(root_dir))
    _list_recursively(root_dir, 0, indent_char, indent_count, exclude_extensions)

def get_exclude_extensions():
    """
    获取用户输入的要排除的后缀列表。.pyc,.tmp,.log,.pth,.csv,.json,.safetensors,.txt,.jsonl
    """
    exclude_input = input("请输入要排除的文件后缀（多个用逗号分隔，例如 .pyc,.tmp,.log，直接回车则不排除）: ")
    if exclude_input.strip():
        return [ext.strip() for ext in exclude_input.split(',')]
    return []

def _list_recursively(current_path, level, indent_char, indent_count, exclude_extensions=None):
    """
    内部递归函数。
    参数:
    current_path (str): 当前正在处理的目录路径。
    level (int): 当前的递归深度。
    indent_char (str): 缩进字符。
    indent_count (int): 缩进字符数量。
    """
    level += 1
    try:
        items = sorted(os.listdir(current_path))
        prefix = indent_char * indent_count * level
        for item in items:
            full_path = os.path.join(current_path, item)
            if os.path.isfile(full_path):
                _, ext = os.path.splitext(item)
                if ext in (exclude_extensions or []):
                    continue
            print(f"{prefix}{item}")
            if os.path.isdir(full_path):
                _list_recursively(full_path, level, indent_char, indent_count, exclude_extensions)
    except PermissionError:
        print(f"{prefix} [无权访问]")
    except Exception as e:
        print(f"发生错误: {e}")
    
if __name__ == "__main__":
    start_path = input("请输入要扫描的文件夹路径 (例如 D:\\ai_monitor): ")
    indent_symbol = input("请输入用于缩进的符号 (默认是 '--'): ") or '--'
    while True:
        try:
            indent_num_str = input("请输入每个层级重复符号的数量 (默认是 1): ") or '1'
            indent_num = int(indent_num_str)
            if indent_num > 0:
                break
            else:
                print("数量必须是正整数。")
        except ValueError:
            print("请输入一个有效的数字。")
    # 获取要排除的后缀
    exclude_exts = get_exclude_extensions()
    print(f"用户选择的排除后缀: {exclude_exts}")
    print("\n--- 目录结构如下 ---\n")
    list_directory_tree(start_path, indent_char=indent_symbol, indent_count=indent_num, exclude_extensions=exclude_exts)
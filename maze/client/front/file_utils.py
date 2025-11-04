"""
文件处理工具类
"""

from typing import Optional
from pathlib import Path


class FileInput:
    """
    文件输入标记类，用于明确指定某个参数是文件路径
    
    示例:
        task = workflow.add_task(
            process_image,
            inputs={
                "image": FileInput("C:/local/image.jpg"),
                "text": "some text"
            }
        )
    """
    
    def __init__(self, local_path: str):
        """
        初始化文件输入
        
        Args:
            local_path: 本地文件路径
        """
        self.local_path = str(local_path)
        self.path = Path(local_path)
        
        # 验证文件存在
        if not self.path.exists():
            raise FileNotFoundError(f"文件不存在: {local_path}")
        
        if not self.path.is_file():
            raise ValueError(f"路径不是文件: {local_path}")
    
    @property
    def filename(self) -> str:
        """获取文件名"""
        return self.path.name
    
    @property
    def extension(self) -> str:
        """获取文件扩展名"""
        return self.path.suffix
    
    def read_bytes(self) -> bytes:
        """读取文件内容（字节）"""
        return self.path.read_bytes()
    
    def __repr__(self) -> str:
        return f"FileInput('{self.local_path}')"


def is_file_type(data_type: str) -> bool:
    """
    检查数据类型是否是文件类型
    
    Args:
        data_type: 数据类型字符串
        
    Returns:
        bool: 是否是文件类型
        
    示例:
        >>> is_file_type("file")
        True
        >>> is_file_type("file:image")
        True
        >>> is_file_type("str")
        False
    """
    if data_type is None:
        return False
    return data_type.startswith("file")


def extract_file_subtype(data_type: str) -> Optional[str]:
    """
    提取文件子类型
    
    Args:
        data_type: 数据类型字符串
        
    Returns:
        Optional[str]: 文件子类型，如果没有则返回None
        
    示例:
        >>> extract_file_subtype("file:image")
        'image'
        >>> extract_file_subtype("file")
        None
    """
    if not is_file_type(data_type):
        return None
    
    parts = data_type.split(":", 1)
    if len(parts) == 2:
        return parts[1]
    return None


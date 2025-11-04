"""
内置的文件处理任务示例

这些任务展示如何处理图片、音频等文件类型
"""

from maze.client.front.decorator import task


@task(
    inputs=["image_path"],
    outputs=["info"],
    data_types={
        "image_path": "file:image",
        "info": "dict"
    },
    resources={"cpu": 1, "cpu_mem": 256, "gpu": 0, "gpu_mem": 0}
)
def get_image_info(params):
    """
    获取图片信息
    
    输入:
        image_path: 图片文件路径（自动上传到服务器）
        
    输出:
        info: 图片信息（尺寸、格式等）
    """
    from PIL import Image
    import os
    
    image_path = params.get("image_path")
    
    # 打开图片
    img = Image.open(image_path)
    
    # 获取信息
    info = {
        "width": img.width,
        "height": img.height,
        "format": img.format,
        "mode": img.mode,
        "size_bytes": os.path.getsize(image_path),
        "path": image_path
    }
    
    return {"info": info}


@task(
    inputs=["image_path", "output_size"],
    outputs=["resized_image_path", "info"],
    data_types={
        "image_path": "file:image",
        "output_size": "str",  # 格式: "宽x高", 如 "800x600"
        "resized_image_path": "file:image",
        "info": "dict"
    },
    resources={"cpu": 1, "cpu_mem": 512, "gpu": 0, "gpu_mem": 0}
)
def resize_image(params):
    """
    调整图片大小
    
    输入:
        image_path: 输入图片路径
        output_size: 输出尺寸，格式 "宽x高"
        
    输出:
        resized_image_path: 调整后的图片路径
        info: 处理信息
    """
    from PIL import Image
    import os
    from pathlib import Path
    
    image_path = params.get("image_path")
    output_size_str = params.get("output_size")
    
    # 解析目标尺寸
    width, height = map(int, output_size_str.split("x"))
    
    # 打开并调整图片
    img = Image.open(image_path)
    original_size = img.size
    
    resized_img = img.resize((width, height), Image.Resampling.LANCZOS)
    
    # 生成输出路径（在同目录下）
    input_path = Path(image_path)
    output_path = input_path.parent / f"resized_{input_path.name}"
    
    # 保存
    resized_img.save(output_path, quality=95)
    
    info = {
        "original_size": f"{original_size[0]}x{original_size[1]}",
        "new_size": f"{width}x{height}",
        "input_path": str(image_path),
        "output_path": str(output_path)
    }
    
    return {
        "resized_image_path": str(output_path),
        "info": info
    }


@task(
    inputs=["image_path"],
    outputs=["grayscale_image_path"],
    data_types={
        "image_path": "file:image",
        "grayscale_image_path": "file:image"
    },
    resources={"cpu": 1, "cpu_mem": 256, "gpu": 0, "gpu_mem": 0}
)
def convert_to_grayscale(params):
    """
    将图片转换为灰度图
    
    输入:
        image_path: 输入图片路径
        
    输出:
        grayscale_image_path: 灰度图片路径
    """
    from PIL import Image
    from pathlib import Path
    
    image_path = params.get("image_path")
    
    # 打开并转换
    img = Image.open(image_path)
    grayscale_img = img.convert('L')
    
    # 生成输出路径
    input_path = Path(image_path)
    output_path = input_path.parent / f"gray_{input_path.name}"
    
    # 保存
    grayscale_img.save(output_path)
    
    return {"grayscale_image_path": str(output_path)}


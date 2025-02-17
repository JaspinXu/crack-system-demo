import os
from PIL import Image

def is_fully_black_or_white(image_path):
    """
    检查图像是否为全黑或全白。
    
    :param image_path: 图像文件的路径
    :return: 如果图像是全黑或全白，则返回True；否则返回False
    """
    with Image.open(image_path) as img:
        # 将图像转换为灰度图像
        gray_img = img.convert('L')
        # 获取图像的像素数据
        pixels = gray_img.getdata()
        
        # 检查所有像素是否都是0（全黑）或255（全白）
        first_pixel = next(iter(pixels))
        all_same = all(pixel == first_pixel for pixel in pixels)
        
        return all_same and (first_pixel == 0 or first_pixel == 255 or first_pixel == 1)

def delete_fully_black_or_white_images(directory):
    """
    删除目录下所有全黑或全白的图像。
    
    :param directory: 要检查的目录路径
    """
    for root, _, files in os.walk(directory):
        for file in files:
            # 检查文件扩展名是否为图像文件（例如 .png, .jpg, .jpeg, .bmp 等）
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                file_path = os.path.join(root, file)
                if is_fully_black_or_white(file_path):
                    print(f"Deleting {file_path} (fully black or white)")
                    os.remove(file_path)

# 示例用法
directory_path = 'G:\\CrackForest\\crack_estimate\\12K'  # 替换为你的目录路径
delete_fully_black_or_white_images(directory_path)
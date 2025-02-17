import os  
import csv
from crack_estimate.calculate_metrics import calculate_crack_metrics
import cv2

def create_metrics_for_weights(image_folder_for_weights_path, metric_for_weight_csv):

    # 获取图片文件列表  
    image_files = [f for f in os.listdir(image_folder_for_weights_path) if os.path.isfile(os.path.join(image_folder_for_weights_path, f))]  
      
    # 初始化结果列表，每个图片的指标将转置成列存储  
    results = []
  
    for image_file in image_files:  

        # 读取二值化图像
        binary_image = cv2.imread(image_folder_for_weights_path + '\\' + image_file, cv2.IMREAD_GRAYSCALE)
        binary_image = binary_image / 255
  
        # 处理图片并获取指标列表  
        metrics = calculate_crack_metrics(binary_image)###
          
        if len(results) == 0:
            results = [[] for _ in range(len(metrics))]
        for i, metric in enumerate(metrics):  
            results[i].append(metric)  

    results_transposed = [[row[i] for row in results] for i in range(len(results[0]))]
    with open(metric_for_weight_csv, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)  
        for col in zip(*results_transposed):  
            writer.writerow(col) 
 
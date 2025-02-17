import os  
import csv
import torch
from crack_estimate.calculate_metrics import calculate_crack_metrics
import crack_estimate.crack_entropy_fuzzy as cef
import cv2
import pickle
import crack_estimate.metric2csv as metric2csv
import crack_estimate.metric_config as mc
import crack_estimate.change_unit_format as cuf

def est(image_folder_for_calculate):
    ## 用来产生熵权的二值化图片训练集路径
    image_folder_for_weights_path = 'G:\\CrackForest\\crack_estimate\\12K'
    ## 保存产生权重的指标的csv路径
    metric_for_weight_csv = os.path.join(image_folder_for_calculate, 'metrics_for_weights.csv')
    ## 要计算综合得分的二值化裂缝图片路径，一般来说这个和前面的产生权重的数据集不一样
    # image_folder_for_calculate = 'G:\\CrackForest\\results'
    ## 保存所有综合得分以及对应的图片名、指标的csv
    output_csv = os.path.join(image_folder_for_calculate, 'result_scores.csv')
    entropy_weights_pickle_path = os.path.join(image_folder_for_calculate, 'entropy_weights.pickle')
    if not os.path.exists(metric_for_weight_csv):
        metric2csv.create_metrics_for_weights(image_folder_for_weights_path, metric_for_weight_csv)
    else:
        if not os.path.exists(entropy_weights_pickle_path):
            data_tensor = cef.read_csv_to_tensor(metric_for_weight_csv)   
            weights = cef.entropy_weight(data_tensor, mc.is_benefit)
            with open(entropy_weights_pickle_path, 'wb') as f:
                pickle.dump(weights, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(entropy_weights_pickle_path, 'rb') as f:
                weights = pickle.load(f)

        if not os.path.exists(output_csv):  
            with open(output_csv, mode='w', newline='') as csv_file:  
                writer = csv.writer(csv_file)
        # 获取图片文件列表  
        image_files = [f for f in os.listdir(image_folder_for_calculate) if os.path.isfile(os.path.join(image_folder_for_calculate, f))]  
        
        # 初始化结果列表，每个图片的指标将转置成列存储  
        results = []
        final_scores = []
    
        for image_file in image_files:  

            # 读取二值化图像
            binary_image = cv2.imread(image_folder_for_calculate + '\\' + image_file, cv2.IMREAD_GRAYSCALE)
            binary_image = binary_image / 255
    
            # 处理图片并获取指标列表  
            metrics = calculate_crack_metrics(binary_image) ###

            # 隶属度矩阵
            indicator_vector = torch.tensor(metrics)

            # indicator_vector = indicator_vector.double()
            membership_matrix = cef.calculate_membership_matrix(indicator_vector, mc.level_vectors)
            # print("隶属度矩阵:\n", membership_matrix)

            scores = torch.tensor([25, 50, 75, 100])  # 等级的分数

            # 计算模糊向量
            fuzzy_vector = cef.calculate_fuzzy_vector(weights, membership_matrix)
            # print("模糊向量：", fuzzy_vector)

            # 计算总得分
            fuzzy_score = cef.calculate_fuzzy_score(fuzzy_vector, scores)
            print("总得分：", fuzzy_score.item())
            
            if len(results) == 0:
                results = [[] for _ in range(len(metrics))]
            for i, metric in enumerate(metrics):  
                results[i].append(metric)  
            final_scores.append([fuzzy_score.item()])

        results_transposed = [[row[i] for row in results] for i in range(len(results[0]))]

        with open(output_csv, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)  
            writer.writerow(image_files)
            for col in zip(*final_scores):  
                writer.writerow(col)
            for col in zip(*results_transposed):  
                writer.writerow(col) 
        cuf.change_format_csv(output_csv,cuf.processing_rules)
        print('Result has been created!')

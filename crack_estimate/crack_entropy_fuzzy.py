import pandas as pd  
import torch  
  
def entropy_weight(data_tensor, is_benefit):  
    """  
    熵权法计算指标权重 data_tensor的行代表指标 列代表样本 注意正向指标和负向指标的转换 
    """ 
    # 标准化处理  
    data_standardized = torch.zeros_like(data_tensor)  
    for i, benefit in enumerate(is_benefit):  
        if benefit:  # 如果是正向指标  
            min_value = data_tensor[i].min()  
            max_value = data_tensor[i].max()  
            data_standardized[i] = (data_tensor[i] - min_value) / (max_value - min_value)  
        else:  # 如果是负向指标  
            min_value = data_tensor[i].min()  
            max_value = data_tensor[i].max()  
            # 取相反数进行标准化  
            data_standardized[i] = (max_value - data_tensor[i]) / (max_value - min_value)  
    # print(data_standardized)
    # 归一化处理  
    data_normalized = data_standardized / data_standardized.sum(dim=1, keepdim=True)  
      
    # 避免对数计算中的数值问题，将0替换为一个很小的正数  
    epsilon = 1e-12  
    data_normalized = torch.clamp(data_normalized, min=epsilon)  

    # 计算熵值  
    data_entropy = -torch.sum(data_normalized * torch.log(data_normalized), dim=1) / torch.log(torch.tensor(data_tensor.shape[1],dtype=torch.float64))  
      
    # 计算权重  
    weights = (1 - data_entropy) / (1 - data_entropy).sum() 
    return weights  
  
def read_csv_to_tensor(file_path):  
    """  
    读取CSV文件转换为Tensor csv文件行代表指标 列代表样本  
    """  
    data = pd.read_csv(file_path, header=None)  
    data = data.astype('float64')
    return torch.tensor(data.values, dtype=torch.float64)  

def trapezoidal_membership(x, levels):  
    """  
    计算梯形隶属度函数。    
    x为指标值 levels为等级向量 包含四个等级的标准值 从小到大排序。        
    """  
    membership = torch.zeros(4)  
      
    if x <= levels[0]:  
        membership[0] = 1  
    elif x <= levels[1]:  
        membership[0] = (levels[1] - x) / (levels[1] - levels[0])  
        membership[1] = 1 - membership[0]  
    elif x < levels[2]:  
        membership[1] = (levels[2] - x) / (levels[2] - levels[1])  
        membership[2] = 1 - membership[1]  
    elif x < levels[3]:  
        membership[2] = (levels[3] - x) / (levels[3] - levels[2])  
        membership[3] = 1 - membership[2]  
    else:  
        membership[3] = 1  
          
    membership /= membership.sum()  
    return membership  

def calculate_membership_matrix(indicator_vector, level_vectors):  
    """  
    计算隶属度矩阵 indicator_vector为指标向量 level_vectors为各个指标的等级向量列表
    返回隶属度矩阵 
    """ 
    num_indicators = indicator_vector.size(0)  
    num_levels = 4  
    membership_matrix = torch.zeros((num_indicators, num_levels), dtype=torch.float64)  
      
    for i in range(num_indicators):  
        membership_matrix[i] = trapezoidal_membership(indicator_vector[i], level_vectors[i])  
          
    return membership_matrix
  
def calculate_fuzzy_vector(weights, membership_matrix):  
    """  
    计算模糊向量  
    """  
    weights = weights.unsqueeze(0) # 增加维度以匹配矩阵乘法  
    membership_matrix = membership_matrix.double()
    fuzzy_vector = torch.mm(weights, membership_matrix)  
    return fuzzy_vector 

def calculate_fuzzy_score(fuzzy_vector, scores):  
    """  
    计算加权总得分    
    """  
    # 确保权重和分数是浮点类型，以便进行计算 
    fuzzy_vector = fuzzy_vector.squeeze() 
    fuzzy_vector = fuzzy_vector.float()  
    scores = scores.float()  
      
    # 计算加权得分  
    fuzzy_score = torch.dot(fuzzy_vector, scores)  
    return fuzzy_score
 

import torch  
####config 
## 负向指标和正向指标的配置
is_benefit = [False,
              False,
              False,
              False,
              False,
              False,
              False,
              True,#####
              False,
              False,
              False,
              False,
              False,
              False,
              False,
              True,####
              False,
              False,
              False,
              False
              ]  # False负向的，True是正向的  

level_vectors = [  
        torch.tensor([0.0, 0.25, 0.5, 0.75]),  
        torch.tensor([0.0, 0.25, 0.5, 0.75]),  
        torch.tensor([0.0, 0.25, 0.5, 0.75]),
        torch.tensor([0.0, 0.25, 0.5, 0.75]),
        torch.tensor([0.0, 0.25, 0.5, 0.75]),
        torch.tensor([0.0, 0.25, 0.5, 0.75]),
        torch.tensor([0.0, 0.25, 0.5, 0.75]),
        torch.tensor([0.0, 0.25, 0.5, 0.75]),
        torch.tensor([0.0, 0.25, 0.5, 0.75]),
        torch.tensor([0.0, 0.25, 0.5, 0.75]),
        torch.tensor([0.0, 0.25, 0.5, 0.75]),
        torch.tensor([0.0, 0.25, 0.5, 0.75]),
        torch.tensor([0.0, 0.25, 0.5, 0.75]),
        torch.tensor([0.0, 0.25, 0.5, 0.75]),
        torch.tensor([0.0, 0.25, 0.5, 0.75]),
        torch.tensor([0.0, 0.25, 0.5, 0.75]),
        torch.tensor([0.0, 0.25, 0.5, 0.75]),
        torch.tensor([0.0, 0.25, 0.5, 0.75]),
        torch.tensor([0.0, 0.25, 0.5, 0.75]),
        torch.tensor([0.0, 0.25, 0.5, 0.75])  
    ] 

"""
20个指标
crack_length,   
crack_area,     
max_crack_width,  
avg_crack_width,  
crack_density,   
nominal_avg_width,
fractal_dim,
crack_spacing,### 正向
max_crack_height,
avg_crack_height,
avg_crack_shape_factor,
avg_crack_orientation,
crack_volume,
crack_growth_rate,
crack_width_change_rate,
crack_distribution_uniformity,  ###正向
crack_distribution_directionality,
crack_distribution_concentration,
crack_distribution_dispersion,
index_basic_metric
"""
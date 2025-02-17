import numpy as np
from scipy.ndimage import label
from skimage.measure import label as lab
from skimage.morphology import skeletonize,dilation
from skimage.measure import regionprops
from skimage.filters import threshold_otsu

def calculate_crack_metrics(binary_image):
    # 计算裂缝长度
    skeleton = skeletonize(binary_image)
    crack_length = np.sum(skeleton)
    
    # 计算裂缝面积
    crack_area = np.sum(binary_image)
    
    # 计算裂缝宽度
    labeled_image, num_features = label(binary_image)
    props = regionprops(labeled_image)
    crack_widths = [prop.axis_major_length for prop in props]
    max_crack_width = max(crack_widths)
    avg_crack_width = np.mean(crack_widths)
    
    # 计算裂缝分布密度
    crack_density = num_features / binary_image.size
  
    # 计算区域裂缝宽度标称平均值
    def calculate_nominal_avg_width(binary_image):  
        # 获取图像尺寸  
        height, width = binary_image.shape  
    
        # 初始化存储裂缝信息的列表  
        cracks = []  
    
        # 裂缝匹配和宽度计算  
        for y in range(height):  
            # 找到每一行中裂缝的起始和结束列索引  
            start_indices = np.where(binary_image[y, :] == 1)[0]  
            if len(start_indices) == 0:  
                continue  
    
            end_indices = np.where(np.concatenate(([binary_image[y, 0]], binary_image[y, :-1] != binary_image[y, 1:], [binary_image[y, -1]])) == 1)[0]  
            if len(end_indices) == 0:  
                end_indices = [width - 1]  
    
            for start, end in zip(start_indices, end_indices):  
                # 计算裂缝宽度  
                crack_width = int(np.round(np.sqrt((end - start) ** 2)))  
                crack_height = 1   
                crack_height = height  
     
                cracks.append((start, end, crack_width, crack_height))  
     
        regional_crack_width_nominal_averages = {}  
        region_index = 0  
    
        for crack in cracks:  
            start, end, crack_width, crack_height = crack   
            unidirectional_crack_width_mean = crack_width   
            H = height  
            w_mean = unidirectional_crack_width_mean * (crack_height / H)  
      
            if region_index not in regional_crack_width_nominal_averages:  
                regional_crack_width_nominal_averages[region_index] = w_mean  
            else:  
                regional_crack_width_nominal_averages[region_index] += w_mean  
      
            region_index += 1  
      
        nominal_avg_width = 0
        for region in regional_crack_width_nominal_averages:  
            num_cracks = len(cracks) 
            nominal_avg_width += regional_crack_width_nominal_averages[region] / num_cracks 
    
        return nominal_avg_width  
  
    nominal_avg_width = calculate_nominal_avg_width(binary_image)
    
    # 计算分形维度
    def fractal_dimension(Z):
        def boxcount(Z, k):
            S = np.add.reduceat(
                np.add.reduceat(Z, np.arange(0, Z.shape[0], k).astype('int64'), axis=0),
                                   np.arange(0, Z.shape[1], k).astype('int64'), axis=1)
            return len(np.where((S > 0) & (S < k*k))[0])
        Z = (Z < threshold_otsu(Z))
        p = min(Z.shape)
        n = 2**np.floor(np.log(p)/np.log(2))
        sizes = 2**np.arange(np.log(n)/np.log(2), 1, -1)
        counts = []
        for size in sizes:
            counts.append(boxcount(Z, size))
        coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
        return -coeffs[0]
    
    fractal_dim = fractal_dimension(binary_image)
    
    # 计算裂缝间距
    crack_spacing = np.mean([prop.axis_minor_length for prop in props])
    
    # 计算裂缝高度
    crack_heights = [prop.bbox[2] - prop.bbox[0] for prop in props]
    max_crack_height = max(crack_heights)
    avg_crack_height = np.mean(crack_heights)
    
    # 计算裂缝形态特征
    crack_shape_factors = [prop.perimeter**2 / (4 * np.pi * prop.area) for prop in props]
    avg_crack_shape_factor = np.mean(crack_shape_factors)
    
    # 计算裂缝方向
    crack_orientations = [prop.orientation for prop in props]
    avg_crack_orientation = np.mean(crack_orientations)
    
    # 计算裂缝体积
    crack_volume = np.sum(binary_image) * avg_crack_width
    
    # 计算裂缝扩展速率
    crack_growth_rate = crack_length / np.sum(binary_image)
    
    # 计算裂缝宽度变化率
    crack_width_change_rate = (max_crack_width - avg_crack_width) / avg_crack_width
    
    # 计算裂缝分布均匀性
    crack_distribution_uniformity = np.std(crack_widths) / avg_crack_width
    
    # 计算裂缝分布方向性
    crack_distribution_directionality = np.std(crack_orientations) / avg_crack_orientation
    
    # 计算裂缝分布集中度
    crack_distribution_concentration = np.max(crack_widths) / np.sum(crack_widths)
    
    # 计算裂缝分布离散度
    crack_distribution_dispersion = np.std(crack_widths) / np.mean(crack_widths)

    # 计算系数决定的可微调基础指标
    def calculate_basic_metrics(binary_image,n,a,l,w):
        def calculate_crack_properties(mask):

            labeled_mask = lab(mask)
            regions = regionprops(labeled_mask)
            crack_properties = []

            for region in regions:
                area = region.area
                minr, minc, maxr, maxc = region.bbox
                width = maxc - minc + 1
                height = maxr - minr + 1
                length = np.sqrt(width**2 + height**2)
                average_width = area / length
                crack_properties.append((area, length, average_width))

            return crack_properties
        
        # 读取图像
        mask = binary_image

        # 扩张裂缝部分向外两个像素宽度
        dilated_mask = dilation(mask, footprint=np.ones((3, 3)))

        # 计算裂缝数量
        labeled_mask = lab(dilated_mask)
        num_cracks = labeled_mask.max()

        # 计算骨架
        skeleton = skeletonize(dilated_mask)*255

        # 创建一个空白图像，用于存储重叠后的图像
        overlapped_image = np.zeros(mask.shape, dtype=np.uint8)

        # 遍历图像像素，将两张图像的白色像素叠加在一起
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j] == 255 or skeleton[i, j] == 255:
                    overlapped_image[i, j] = 255

        # 计算裂缝属性
        all_area = []
        all_length = []
        all_average_width = []
        crack_properties = calculate_crack_properties(overlapped_image)
        for i, (area, length, average_width) in enumerate(crack_properties):
            all_area.append(area)
            all_length.append(length)
            all_average_width.append(average_width)

        index_basic_metric = n*num_cracks + a*sum(all_area) + l*sum(all_length) + w*sum(all_average_width)

        return index_basic_metric
    
    binary_image_255 =binary_image*255
    orign_binary_image = binary_image_255.astype('uint8')
    index_basic_metric = calculate_basic_metrics(orign_binary_image,0.95,0.025,0.020,0.005) ##该指标计算相当主观，需要根据情况微调
    
    # 返回所有计算的指标
    return [
        crack_length,   # cm
        crack_area,     # cm^2
        max_crack_width,  # cm
        avg_crack_width,  # cm
        crack_density,   
        nominal_avg_width,  
        fractal_dim,
        crack_spacing,### 正向  # cm
        max_crack_height, # cm
        avg_crack_height, # cm
        avg_crack_shape_factor,
        avg_crack_orientation,
        crack_volume, # cm^3
        crack_growth_rate, # %
        crack_width_change_rate, # %
        crack_distribution_uniformity,  ###正向 # %
        crack_distribution_directionality,
        crack_distribution_concentration, # %
        crack_distribution_dispersion, # %
        index_basic_metric
    ]


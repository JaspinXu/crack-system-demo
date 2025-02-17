import csv  
import os
  
def change_format_csv(file_path, processing_rules):  

    with open(file_path, mode='r', newline='', encoding='utf-8') as infile:  
        reader = csv.reader(infile)  
        rows = list(reader)  
  
    # 对每一行应用相应的处理函数  
    for idx, row in enumerate(rows):  
        if idx in processing_rules:  
            process_func = processing_rules[idx]  
            rows[idx] = process_func(row)  
  
    # 将修改后的内容写回文件  
    with open(file_path, mode='w', newline='', encoding='utf-8') as outfile:  
        writer = csv.writer(outfile)  
        writer.writerows(rows) 
  
def add_cm_0dot1(row, para=0.1):  
    return [f"{para*float(value)}cm" for value in row] 

def add_cm2(row):   
    return [f"{value}cm^2" for value in row] 

def add_cm3_0dot1(row, para=0.1):  
    return [f"{para*float(value)}cm^3" for value in row] 
  
def convert_to_percent(row):  
    return [f"{float(value) * 100:.2f}%" for value in row]   
  
# 处理规则  
processing_rules = {  
    2: add_cm_0dot1,    
    3: add_cm2,  
    4: add_cm_0dot1,
    5: add_cm_0dot1,
    9: add_cm_0dot1,
    10: add_cm_0dot1,
    11: add_cm_0dot1,
    2: add_cm_0dot1,
    2: add_cm_0dot1,
    14: add_cm3_0dot1,
    15: convert_to_percent,
    16: convert_to_percent,
    17: convert_to_percent,
    19: convert_to_percent,
    20: convert_to_percent
}  
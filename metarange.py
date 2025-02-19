import os
import re
from collections import defaultdict

def parse_value(value):
    """提取数值并转换为 float"""
    match = re.search(r"[-+]?\d*\.\d+|\d+", value)  # 提取数字（支持整数和小数）
    if match:
        return float(match.group())  # 转换为 float
    return None  # 解析失败返回 None

def process_txt_files(folder_path):
    """解析文件夹中的所有TXT文件, 计算每个键的最小值和最大值"""
    data = defaultdict(list)
    
    # 遍历文件夹中的所有txt文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    match = re.match(r'"(.*?)":\s*"(.*?)"', line.strip())
                    if match:
                        key, value = match.groups()
                        parsed_value = parse_value(value)
                        if parsed_value is not None:  # 确保解析成功
                            data[key].append(parsed_value)
    
    # 计算每个键的最小值和最大值
    results = {}
    for key, values in data.items():
        results[key] = (min(values), max(values))
    
    return results

def save_results(results, output_file):
    """将结果保存到文件"""
    with open(output_file, "w", encoding="utf-8") as f:
        for key, (min_val, max_val) in sorted(results.items()):
            f.write(f"{key} {min_val} {max_val}\n")

if __name__ == "__main__":
    folder_path = "E:\Deep Learning\datasets\RAW\ELD/CanonEOS700D"  # 修改为你的txt文件夹路径
    output_file = "metarange.txt"
    results = process_txt_files(folder_path)
    save_results(results, output_file)
    print(f"结果已保存到 {output_file}")

import os
import subprocess

# 定义需要提取的元数据键
required_keys = [
    "ExposureTime",
    "ISO",
    "FNumber",
    "BrightnessValue",
    "FocalLength",
    "BlueBalance",
    "RedBalance",
    "LightValue",
    "ColorMatrix"
]

def extract_metadata(folder_path):
    # 检查文件夹是否存在
    if not os.path.isdir(folder_path):
        print(f"指定的文件夹路径不存在：{folder_path}")
        return

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件扩展名是否为 RAW 图像格式
        if filename.lower().endswith(('.cr2', '.nef', '.arw', '.dng', '.raw')):
            # 构建完整的文件路径
            file_path = os.path.join(folder_path, filename)
            # 构建输出的 TXT 文件名
            output_txt = os.path.join(folder_path, os.path.splitext(filename)[0] + '.txt')

            # 调用 exiftool 命令
            try:
                # 使用 subprocess 运行 exiftool 命令
                result = subprocess.run(
                    ['exiftool', '-s', '-m', file_path],  # 获取元数据
                    capture_output=True,  # 捕获输出
                    text=True  # 输出为文本格式
                )
                if result.returncode == 0:
                    # 将 exiftool 的输出解析为字典
                    metadata = {}
                    for line in result.stdout.splitlines():
                        if ':' in line:  # 确保行中包含冒号
                            parts = line.split(':', 1)
                            if len(parts) == 2:
                                key, value = parts
                                key = key.strip()
                                value = value.strip()
                                metadata[key] = value

                    # 筛选出需要的键值对
                    filtered_metadata = {key: metadata[key] for key in required_keys if key in metadata}

                    # 将筛选后的元数据写入 TXT 文件
                    with open(output_txt, 'w', encoding='utf-8') as txt_file:
                        for key, value in filtered_metadata.items():
                            txt_file.write(f'"{key}": "{value}"\n')
                    print(f"元数据已保存到：{output_txt}")
                else:
                    print(f"处理文件 {filename} 时出错：{result.stderr}")
            except Exception as e:
                print(f"运行 exiftool 时出错：{e}")

if __name__ == "__main__":
    # 指定文件夹路径
    folder_path = "E:\Deep Learning\datasets\RAW\ELD\CanonEOS70D"
    extract_metadata(folder_path)
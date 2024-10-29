import os
import shutil
import exifread
import re
def rename_images(base_path):
    for i in range(1, 11):
        scene_dir = os.path.join(base_path, f'scene-{i}')
        if os.path.exists(scene_dir):
            images = [f for f in os.listdir(scene_dir) if f.startswith('IMG')]
            images.sort()
            for j, img in enumerate(images, start=1):
                new_name = f'scene{i}_0{j:03d}{os.path.splitext(img)[1]}'
                old_name = os.path.join(scene_dir, img)
                new_name_path = os.path.join(scene_dir, new_name)
                os.rename(old_name, new_name_path)

def move_images_and_cleanup(path):
    for i in range(1, 11):
        scene_dir = os.path.join(path, f'scene-{i}')
        if os.path.exists(scene_dir):
            for filename in os.listdir(scene_dir):
                file_path = os.path.join(scene_dir, filename)
                if os.path.isfile(file_path):
                    shutil.move(file_path, path)
            shutil.rmtree(scene_dir)

def metainfo(rawpath):
    with open(rawpath, 'rb') as f:
        tags = exifread.process_file(f)
        _, suffix = os.path.splitext(os.path.basename(rawpath))

        if suffix == '.dng':
            expo = eval(str(tags['Image ExposureTime']))
            iso = eval(str(tags['Image ISOSpeedRatings']))
        else:
            expo = eval(str(tags['EXIF ExposureTime']))
            iso = eval(str(tags['EXIF ISOSpeedRatings']))
    return iso, expo

def generate_pairs(camera, data_dir):
    images = [f for f in os.listdir(data_dir)]
    images.sort()
    for image in images: #scene1_0001.CR2'
        filename, suffix = os.path.splitext(os.path.basename(image))#scene1_0001 .CR2
        input_path = os.path.join(data_dir,image)
        id = int(filename[-4:])
        if id in [1, 6, 11, 16]:
            continue
        elif id in [2, 3]:
            pattern = r'(\d{4})'
            target_path = re.sub(pattern, '0001', input_path)
            iso, expo = metainfo(target_path)
            target_expo = iso * expo
            iso, expo = metainfo(input_path)
            ratio = target_expo / (iso * expo)
        elif id in [4, 5, 7, 8]:
            pattern = r'(\d{4})'
            target_path = re.sub(pattern, '0006', input_path)
            iso, expo = metainfo(target_path)
            target_expo = iso * expo
            iso, expo = metainfo(input_path)
            ratio = target_expo / (iso * expo)
        elif id in [9, 10, 12, 13]:
            pattern = r'(\d{4})'
            target_path = re.sub(pattern, '0011', input_path)
            iso, expo = metainfo(target_path)
            target_expo = iso * expo
            iso, expo = metainfo(input_path)
            ratio = target_expo / (iso * expo)
        elif id in [14, 15]:
            pattern = r'(\d{4})'
            target_path = re.sub(pattern, '0016', input_path)
            iso, expo = metainfo(target_path)
            target_expo = iso * expo
            iso, expo = metainfo(input_path)
            ratio = target_expo / (iso * expo)
        with open(f'{camera}.txt', 'a') as f:
            f.write(f"{input_path[input_path.index('scene'):]} {target_path[target_path.index('scene'):]} {ratio}\n")
        with open(f'{camera}_{ratio}.txt', 'a') as f:
            f.write(f"{input_path[input_path.index('scene'):]} {target_path[target_path.index('scene'):]} {int(ratio)}\n")

if __name__ == '__main__':
    data_dir = 'E:\Deep Learning\datasets\RAW\ELD'
    cameras = ['CanonEOS70D', 'CanonEOS700D', 'NikonD850', 'SonyA7S2']
    for camera in cameras:
        generate_pairs(camera, os.path.join(data_dir, camera))
        #path = os.path.join(datadir, camera)
        #rename_images(path)
        #move_images_and_cleanup(path)
    
    
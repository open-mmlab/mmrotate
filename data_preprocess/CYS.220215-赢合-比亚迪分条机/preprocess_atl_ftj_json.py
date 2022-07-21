import logging
from pathlib import Path
from sonic_ai.pipelines.utils_labelme import load_labelme_datasets, labelme2coco, save_json
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)

category_map = {
    x.split(' ')[0]: x.strip().split(' ')[1].strip()
    for x in open(
         '/data2/5-标注数据/0-分条机-CYS.220215-赢合-比亚迪分条机/label.ini',
        encoding='utf-8').readlines()
}
category_list = list(category_map.values())
category_list = sorted(set(category_list), key=category_list.index)
output_path = f'/data/xys/data/byd_ftj'

print(category_map)
print(category_list)
print(output_path)

dataset_path_list = ['/data2/5-标注数据/0-分条机-CYS.220215-赢合-比亚迪分条机']

data_list = load_labelme_datasets(dataset_path_list, category_map)
train_list, valid_list = train_test_split(
    data_list, test_size=0.1, random_state=42)
train_coco = labelme2coco(train_list, category_map)
valid_coco = labelme2coco(valid_list, category_map)
save_json(train_coco, Path(output_path, 'train.json'))
save_json(valid_coco, Path(output_path, 'valid.json'))

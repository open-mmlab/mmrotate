import shutil

import tqdm
from mmdet.datasets.pipelines.compose import Compose

from sonic_ai.pipelines.init_pipeline import *

data_root = '/data/xys/data/byd_ftj/'

label_path = '/data2/5-标注数据/0-分条机-CYS.220215-赢合-比亚迪分条机/label.ini'
dataset_path_list = ['/data2/5-标注数据/0-分条机-CYS.220215-赢合-比亚迪分条机']


def create_text_and_img(dir_name, start, end):
    ann_file = data_root + dir_name + '/labelTxt'
    img_prefix = data_root + dir_name + '/images'

    os.makedirs(ann_file, exist_ok=True)
    os.makedirs(img_prefix, exist_ok=True)

    init_pipeline = [
        dict(type='LoadCategoryList', ignore_labels=['屏蔽']),
        dict(type='LoadPathList'),
        dict(type='SplitData', start=start, end=end, key='json_path_list'),
        dict(type='LoadJsonDataList'),
        dict(type='LoadLabelmeDataset'),
    ]

    data = dict(
        dataset_path_list=dataset_path_list,
        label_path=label_path,
    )
    compose = Compose(init_pipeline)
    compose(data)

    json_data_list = data['json_data_list']
    category_map = data['category_map']

    print('搬运图片和生成标签文件中')
    for idx, json_data in tqdm(enumerate(json_data_list),
                               total=len(json_data_list)):
        text_name = Path(Path(json_data['imagePath']).name).with_suffix('.txt')
        text = open(f"{ann_file}/{text_name}", "w+")
        lines = []
        for shape in json_data['shapes']:
            cont = shape['points']
            rect = cv2.minAreaRect(np.array(cont).astype(int))
            poly = cv2.boxPoints(rect)

            label = shape['label']
            difficulty = 0
            if label not in category_map or category_map[label] == '屏蔽':
                difficulty = 1
            else:
                label = category_map[label]

            line = ""
            for x, y in poly:
                line += f"{x} {y} "
            line += f"{label} {difficulty}\n"
            lines.append(line)
        text.writelines(lines)
        text.close()

        src = json_data['imagePath']
        dst = f"{img_prefix}/{Path(json_data['imagePath']).name}"
        shutil.copy(src, dst)

    print(f"text数量{len(os.listdir(ann_file))}")
    print(f'图片数量{len(os.listdir(img_prefix))}')


print('生成训练集中')
create_text_and_img(dir_name='train', start=0, end=0.8)
print('生成验证集中')
create_text_and_img(dir_name='test', start=0.8, end=1)

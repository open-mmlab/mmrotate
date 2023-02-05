import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Test (and eval) a model')

    parser.add_argument(
        '--mongo',
        default='3060',
        type=str,
        help='dump predictions to a pickle file for offline evaluation')

    # parser.add_argument(
    #     '--prefix',
    #     type=str,
    #     default='work_dirs/headet/configs-ic19-rbb-rbb/'
    #     help='dump prediction')

    parser.add_argument(
        '--prefix',
        default='work_dirs/headet/configs-ic19-obb-obb/',
        type=str,
        help='dump predictions to a pickle file for offline evaluation')
    

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()

    mongo = args.mongo
    prefix = args.prefix

    groups = os.listdir(prefix)

    for i, group in enumerate(groups):

        print(f'group {i} of {len(groups)}: {group}')

        group_configs = os.listdir(f'{prefix}{group}')

        for j, config in enumerate(group_configs):

            print(f'config {j} of {len(group_configs)}: {config}')

            pths = os.listdir(f'{prefix}{group}/{config}')
            pths = [pth for pth in pths if pth.endswith('.pth')]

            sorted(pths, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            
            try:
                cmd = f'python projects/headet/tools/test.py {prefix}{group}/{config}/{config}.py {prefix}{group}/{config}/{pths[-1]} --mongo {mongo}'
                os.system(cmd)
            except Exception as e:
                print(e)

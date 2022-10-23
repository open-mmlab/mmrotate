# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmengine import Config


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert benchmark model list to script')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--port', type=int, default=29666, help='dist port')
    parser.add_argument(
        '--run', action='store_true', help='run script directly')
    parser.add_argument(
        '--out', type=str, help='path to save model benchmark script')

    args = parser.parse_args()
    return args


def process_model_info(model_info):
    config = model_info['config'].strip()
    checkpoint = model_info['checkpoint'].strip()
    return dict(config=config, checkpoint=checkpoint)


def create_test_bash_info(commands, model_test_dict, port, script_name):
    config = model_test_dict['config']
    checkpoint = model_test_dict['checkpoint']

    echo_info = f' \necho \'{config}\' &'
    commands.append(echo_info)
    commands.append('\n')

    command_info = f'python {script_name} '

    command_info += f'{config} '
    command_info += f'$CHECKPOINT_DIR/{checkpoint} '

    command_info += f'--cfg-option env_cfg.dist_cfg.port={port} '
    command_info += ' &'

    commands.append(command_info)


def main():
    args = parse_args()
    if args.out:
        out_suffix = args.out.split('.')[-1]
        assert args.out.endswith('.sh'), \
            f'Expected out file path suffix is .sh, but get .{out_suffix}'
    assert args.out or args.run, \
        ('Please specify at least one operation (save/run/ the '
         'script) with the argument "--out" or "--run"')

    commands = []

    checkpoint_root = 'CHECKPOINT_DIR=$1 '
    commands.append(checkpoint_root)
    commands.append('\n')

    script_name = osp.join('tools', 'test.py')
    port = args.port

    cfg = Config.fromfile(args.config)
    for model_key in cfg:
        model_infos = cfg[model_key]
        if not isinstance(model_infos, list):
            model_infos = [model_infos]
        for model_info in model_infos:
            print('processing: ', model_info['config'])
            model_test_dict = process_model_info(model_info)
            create_test_bash_info(commands, model_test_dict, port, script_name)
            port += 1

    command_str = ''.join(commands)
    if args.out:
        with open(args.out, 'w') as f:
            f.write(command_str)
    if args.run:
        os.system(command_str)


if __name__ == '__main__':
    main()

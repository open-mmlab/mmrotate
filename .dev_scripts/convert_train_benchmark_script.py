# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert benchmark model json to script')
    parser.add_argument(
        'txt_path', type=str, help='txt path output by benchmark_filter')
    parser.add_argument(
        '--run', action='store_true', help='run script directly')
    parser.add_argument(
        '--out', type=str, help='path to save model benchmark script')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.out:
        out_suffix = args.out.split('.')[-1]
        assert args.out.endswith('.sh'), \
            f'Expected out file path suffix is .sh, but get .{out_suffix}'
    assert args.out or args.run, \
        ('Please specify at least one operation (save/run/ the '
         'script) with the argument "--out" or "--run"')

    root_name = './tools'
    train_script_name = osp.join(root_name, 'train.py')

    commands = []

    with open(args.txt_path, 'r') as f:
        model_cfgs = f.readlines()
        for i, cfg in enumerate(model_cfgs):
            cfg = cfg.strip()
            if len(cfg) == 0:
                continue
            # print cfg name
            echo_info = f'echo \'{cfg}\' &'
            commands.append(echo_info)
            commands.append('\n')

            command_info = f'python {train_script_name} '
            command_info += f'{cfg} '

            command_info += '--cfg-options default_hooks.checkpoint.' \
                            'max_keep_ckpts=1 '
            command_info += '&'

            commands.append(command_info)

            if i < len(model_cfgs):
                commands.append('\n')

        command_str = ''.join(commands)
        if args.out:
            with open(args.out, 'w') as f:
                f.write(command_str)
        if args.run:
            os.system(command_str)


if __name__ == '__main__':
    main()

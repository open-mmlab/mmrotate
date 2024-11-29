import argparse
import mmcv
import matplotlib.pyplot as plt
import numpy as np

from mmrotate.core.evaluation.eval_map import eval_rbbox_map

from mmcv import Config
from mmrotate.datasets import build_dataset


def plot_pr_curve(config, prediction_path, iou_thr=0.5):
    # If you need to compare with other models, you can start a loop here.
    cfg = Config.fromfile(config)
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

    dataset = build_dataset(cfg.data.test)
    annotations = [dataset.get_ann_info(i) for i in range(len(dataset))]
    # print(dataset.load_annotations(cfg.data.test.ann_file))

    pkl_results = mmcv.load(prediction_path)

    _, results = eval_rbbox_map(pkl_results, annotations, iou_thr=iou_thr)

    precisions = results[0]['precision']
    recalls = results[0]['recall']
    print(precisions)

    recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
    inds = np.searchsorted(recalls, recThrs, side='left')
    q = np.zeros((len(recThrs),))
    try:
        for ri, pi in enumerate(inds):
            q[ri] = precisions[pi]
    except:
        pass
    precisions = np.array(q)
    x = np.arange(0.0, 1.01, 0.01)
    pr_array = precisions
    plt.plot(x, pr_array, label='iou=%0.2f' % iou_thr)

    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.01)
    plt.grid(True)
    plt.legend(loc='lower left')
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet eval image prediction result for each')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        'prediction_path', help='prediction path where test pkl result')
    parser.add_argument(
        '--iou_thr', help='eval iou_thr', type=float, default=0.5)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    mmcv.check_file_exist(args.prediction_path)
    plot_pr_curve(args.config, args.prediction_path, args.iou_thr)


if __name__ == '__main__':
    main()

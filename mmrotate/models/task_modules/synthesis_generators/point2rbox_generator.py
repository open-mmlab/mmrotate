# Copyright (c) OpenMMLab. All rights reserved.
import os

import cv2
import numpy as np
import torch
import torchvision
from mmcv.ops import nms_rotated


def get_pattern_fill(w, h):
    w, h = int(w), int(h)
    p = np.ones((h, w))
    cv2.line(p, (0, 0), (0, h - 1), 0.01, 1)
    cv2.line(p, (0, 0), (w - 1, 0), 0.01, 1)
    cv2.line(p, (w - 1, h - 1), (0, h - 1), 0.01, 1)
    cv2.line(p, (w - 1, h - 1), (w - 1, 0), 0.01, 1)
    return p


def get_pattern_line(w, h):
    w, h = int(w), int(h)
    p = np.ones((h, w))
    interval_range = [3, 6]
    xn = np.random.randint(*interval_range)
    yn = np.random.randint(*interval_range)
    for i in range(xn):
        x = np.intp(np.round((w - 1) * i / (xn - 1)))
        cv2.line(p, (x, 0), (x, h), 0.5, 1)
    for i in range(yn):
        y = np.intp(np.round((h - 1) * i / (yn - 1)))
        cv2.line(p, (0, y), (w, y), 0.5, 1)
    return torch.from_numpy(p)


def get_pattern_rose(w, h):
    w, h = int(w), int(h)
    p = np.ones((h, w), dtype=float)
    t = np.float32(range(100))
    omega_range = [2, 4]
    xn = np.random.randint(*omega_range)
    x = np.sin(t / 99 * 2 * np.pi) * np.cos(
        t / 100 * 2 * np.pi * xn) * w / 2 + w / 2
    y = np.cos(t / 99 * 2 * np.pi) * np.cos(
        t / 100 * 2 * np.pi * 2) * h / 2 + h / 2
    xy = np.stack((x, y), -1)
    cv2.polylines(p, np.int_([xy]), True, 0.5, 1)
    return torch.from_numpy(p)


def get_pattern_li(w, h):
    w, h = int(w), int(h)
    p = np.ones((h, w), dtype=float)
    t = np.float32(range(100))
    s = np.random.rand() * 8
    s2 = np.random.rand() * 0.5 + 0.1
    r = (np.abs(np.cos(t / 99 * 4 * np.pi))**s) * (1 - s2) + s2
    x = r * np.sin(t / 99 * 2 * np.pi) * w / 2 + w / 2
    y = r * np.cos(t / 99 * 2 * np.pi) * h / 2 + h / 2
    xy = np.stack((x, y), -1)
    cv2.fillPoly(p, np.int_([xy]), 1)
    cv2.polylines(p, np.int_([xy]), True, 0.5, 1)
    return torch.from_numpy(p)


def obb2xyxy(obb):
    w = obb[:, 2]
    h = obb[:, 3]
    a = obb[:, 4]
    cosa = torch.cos(a).abs()
    sina = torch.sin(a).abs()
    dw = cosa * w + sina * h
    dh = sina * w + cosa * h
    dx = obb[..., 0]
    dy = obb[..., 1]
    x1 = dx - dw / 2
    y1 = dy - dh / 2
    x2 = dx + dw / 2
    y2 = dy + dh / 2
    return torch.stack((x1, y1, x2, y2), -1)


def plot_one_rotated_box(img,
                         obb,
                         color=[0.0, 0.0, 128],
                         label=None,
                         line_thickness=None):
    width, height, theta = obb[2], obb[3], obb[4] / np.pi * 180
    if theta < 0:
        width, height, theta = height, width, theta + 90
    rect = [(obb[0], obb[1]), (width, height), theta]
    poly = np.intp(np.round(
        cv2.boxPoints(rect)))  # [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    cv2.drawContours(
        image=img, contours=[poly], contourIdx=-1, color=color, thickness=2)


def get_pattern_gaussian(w, h, device):
    w, h = int(w), int(h)
    y, x = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing='ij')
    y = (y - h / 2) / (h / 2)
    x = (x - w / 2) / (w / 2)
    ox, oy = torch.randn(2, device=device).clip(-3, 3) * 0.15
    sx, sy = torch.rand(2, device=device) + 0.3
    z = torch.exp(-((x - ox) * sx)**2 - ((y - oy) * sy)**2) * 0.9 + 0.1
    return z


def generate_sythesis(img, bb_occupied, sca_fact, pattern, prior_size,
                      dense_cls, imgsize):
    device = img.device
    cen_range = [50, imgsize - 50]

    base_scale = (torch.randn(1, device=device) * 0.4).clamp(-1, 1) * sca_fact
    base_scale = torch.exp(base_scale)

    bb_occupied = bb_occupied.clone()
    bb_occupied[:, 2] = prior_size[bb_occupied[:, 6].long(), 0] * 0.7
    bb_occupied[:, 3] = prior_size[bb_occupied[:, 6].long(), 0] * 0.7
    bb_occupied[:, 4] = 0

    bb = []
    palette = [
        torch.zeros(0, 6, device=device) for _ in range(len(prior_size))
    ]
    adjboost = 2
    for b in bb_occupied:
        x, y = torch.rand(
            2, device=device) * (cen_range[1] - cen_range[0]) + cen_range[0]
        dw = prior_size[b[6].long(), 2]
        w = (torch.randn(1, device=device) * 0.4).clamp(-1, 1) * dw
        w = base_scale * torch.exp(w)
        dr = prior_size[b[6].long(), 3]
        r = (torch.randn(1, device=device) * 0.4).clamp(-1, 1) * dr
        h = w * torch.exp(r)
        w *= prior_size[b[6].long(), 0]
        h *= prior_size[b[6].long(), 1]
        a = torch.rand(1, device=device) * torch.pi
        x = x.clip(0.71 * w, imgsize - 1 - 0.71 * w)
        y = y.clip(0.71 * h, imgsize - 1 - 0.71 * h)
        bb.append([x, y, w, h, a, (w * h) / imgsize / imgsize + 0.1, b[6]])

        bx = torch.clip(b[0:2], 16, imgsize - 1 - 16).long()
        nbr0 = img[:, bx[1] - 2:bx[1] + 3, bx[0] - 2:bx[0] + 3].reshape(3, -1)
        nbr1 = img[:, bx[1] - 16:bx[1] + 17,
                   bx[0] - 16:bx[0] + 17].reshape(3, -1)
        c0 = nbr0.mean(1)
        c1 = nbr1[:, (nbr1.mean(0) - c0.mean()).abs().max(0)[1]]
        c = torch.cat((c0, c1), 0)
        palette[b[6].long()] = torch.cat((palette[b[6].long()], c[None]), 0)

        if np.random.random() < 0.2 and adjboost > 0:
            adjboost -= 1
            if b[6].long() in dense_cls:
                itv, dev = torch.rand(
                    1, device=device) * 4 + 2, torch.rand(
                        1, device=device) * 8 - 4
                ofx = (h + itv) * torch.sin(-a) + dev * torch.cos(a)
                ofy = (h + itv) * torch.cos(a) + dev * torch.sin(a)
                for k in range(1, 6):
                    bb.append([
                        x + k * ofx, y + k * ofy, w, h, a,
                        (w * h) / imgsize / imgsize + 0.1 - 0.001 * k, b[6]
                    ])
            else:
                itv, dev = torch.rand(
                    1, device=device) * 40 + 10, torch.rand(
                        1, device=device) * 0
                ofx = (h + itv) * torch.sin(-a) + dev * torch.cos(a)
                ofy = (h + itv) * torch.cos(a) + dev * torch.sin(a)
                for k in range(1, 4):
                    bb.append([
                        x + k * ofx, y + k * ofy, w, h, a,
                        (w * h) / imgsize / imgsize + 0.1 - 0.001 * k, b[6]
                    ])

    bb = torch.tensor(bb)
    bb = torch.cat((bb_occupied, bb), 0)
    _, keep = nms_rotated(bb[:, 0:5], bb[:, 5], 0.05)
    bb = bb[keep]
    bb = bb[bb[:, 5] < 1]

    xyxy = obb2xyxy(bb)
    mask = torch.logical_and(
        xyxy.min(-1)[0] >= 0,
        xyxy.max(-1)[0] <= imgsize - 1)
    bb, xyxy = bb[mask], xyxy[mask]

    for i in range(len(bb)):
        cx, cy, w, h, t, s, c = bb[i, :7]
        ox, oy = torch.floor(xyxy[i, 0:2]).long()
        ex, ey = torch.ceil(xyxy[i, 2:4]).long()
        sx, sy = ex - ox, ey - oy
        theta = torch.tensor(
            [[1 / w * torch.cos(t), 1 / w * torch.sin(t), 0],
             [1 / h * torch.sin(-t), 1 / h * torch.cos(t), 0]],
            dtype=torch.float,
            device=device)
        theta[:, :2] @= torch.tensor([[sx, 0], [0, sy]],
                                     dtype=torch.float,
                                     device=device)
        grid = torch.nn.functional.affine_grid(
            theta[None], (1, 1, sy, sx), align_corners=False)
        p = pattern[c.long()]
        p = p[np.random.randint(0, len(p))][None].clone()
        trans = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
        ])
        if np.random.random() < 0.2:
            p *= get_pattern_line(p.shape[2], p.shape[1])
        if np.random.random() < 0.2:
            p *= get_pattern_rose(p.shape[2], p.shape[1])
        if np.random.random() < 0.2:
            p *= get_pattern_li(p.shape[2], p.shape[1])
        p = trans(p)
        p = torch.nn.functional.grid_sample(
            p[None], grid, align_corners=False, mode='nearest')[0]
        if np.random.random() < 0.9:
            a = get_pattern_gaussian(sx, sy, device) * (p != 0)
        else:
            a = (p != 0).float()
        pal = palette[c.long()]
        color = pal[torch.randint(0, len(pal), (1, ))][0]
        p = p * color[:3, None, None] + (1 - p) * color[3:, None, None]

        img[:, oy:oy + sy,
            ox:ox + sx] = (1 - a) * img[:, oy:oy + sy, ox:ox + sx] + a * p

    return img, bb


def load_basic_pattern(path, set_rc=True, set_sk=True):
    with open(os.path.join(path, 'properties.txt')) as f:
        prior = eval(f.read())
    prior_size = torch.tensor(list(prior.values()))
    pattern = []
    for i in range(len(prior_size)):
        p = []
        if set_rc:
            img = get_pattern_fill(*prior_size[i, (0, 1)])
            p.append(torch.from_numpy(img).float())
        if set_sk:
            if os.path.exists(os.path.join(path, f'{i}.png')):
                img = cv2.imread(os.path.join(path, f'{i}.png'), 0)
                img = img / 255
            else:
                img = get_pattern_fill(*prior_size[i, (0, 1)])
            p.append(torch.from_numpy(img).float())
        pattern.append(p)
    return pattern, prior_size


if __name__ == '__main__':
    pattern, prior_size = load_basic_pattern(
        '../../../../data/basic_patterns/dota')

    img = cv2.imread('../../../../data/basic_patterns/test/P2805.png') - 127.0
    img = torch.from_numpy(img).permute(2, 0, 1).float().contiguous()
    bb = torch.tensor(((318, 84, 0, 0, 0.5, 1, 4), ) * 10)

    import time
    c = time.time()
    # img, bb1 = generate_sythesis(img, bb, 1)
    # print(time.time() - c)
    # bb1[:, 5] = 1
    c = time.time()
    img, bb2 = generate_sythesis(img, bb, 0.5, pattern, prior_size, (4, 5, 6),
                                 1024)
    print(time.time() - c)

    img = img.permute(1, 2, 0).contiguous().cpu().numpy()
    # for b in bb1.cpu().numpy():
    #     plot_one_rotated_box(img, b, color=[0, 100, 0])
    # for b in bb2.cpu().numpy():
    #     plot_one_rotated_box(img, b, color=[0, 0, 100])
    cv2.imshow('', (img + 127) / 256)
    # cv2.imwrite('s1.png', (img + 127))
    cv2.waitKey()

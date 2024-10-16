from turtledemo.forest import start

import cv2
from Bidirectional_a_star import BidirectionalAStar
# from predict import timer
import operator
from PIL import ImageDraw, Image
from scipy.interpolate import splprep, splev
import numpy as np

def smooth_curve(points):
    x = [p[0] for p in points]#mask为空则point为空，mask不通，则point只有2个
    y = [p[1] for p in points]
    tck, u = splprep([x, y], s=0, k=3)
    x_new, y_new = splev(np.linspace(0, 1, 50), tck)
    return list(zip(x_new, y_new))


def draw_path_on_image(image, path, color=(255, 255, 255), radius=1):
    # 生成平滑曲线点
    smooth_path = smooth_curve(path)

    # 检查图像类型并进行转换
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL Image or numpy array")

    draw = ImageDraw.Draw(image)
    for point in smooth_path:
        # 绘制每个点为一个圆
        draw.ellipse([point[1] - radius, point[0] - radius,
                      point[1] + radius, point[0] + radius],
                     fill=color)
    return image

def Down_Sample(XX_numpy, n):
    """
    下采样
    """
    XX_numpy = cv2.resize(XX_numpy, (0, 0), fx=n, fy=n, interpolation=cv2.INTER_NEAREST)  # 栅格化，n=0.1,缩小10倍
    return XX_numpy


# def find_final(map, map_h, map_w):
#     """
#     找最远端路面重点的
#     """
#     h = map_h
#     w = map_w
#     # 认为长度小于N分之总宽度即为结束
#     N = 10
#     white_line_threshold = int(w / N)
#     count_line = 0
#     bian = 0
#     start_x = 0
#     end_x = w
#     route = []
#     for j in range(0, int(h)):  # 从 (0, 0) 开始
#         for i in range(0, w):
#             if map[j, i] == 1:  # 当前像素为白色区域
#                 if count_line == 0:  # 新段的开始
#                     start_x = i
#                 count_line += 1  # 计数
#             elif count_line > 0:  # 遇到黑色区域，表示白色区域结束
#                 end_x = i
#                 if count_line > white_line_threshold:
#                     mid_x = int((start_x + end_x) / 2)
#                     route.append((mid_x, j))  # 保持 y 坐标
#                 count_line = 0  # 重置计数
#             # print("route:", route)
#         # 存一个就行
#         if len(route) > 0:
#             break
#         else:
#             continue
#         count_line = 0
#         bian = 0
#         start_x = 0
#         end_x = w
#     return (route)
###li
# def find_final(map, map_h, map_w):
#     """
#     在可变尺寸的图像中，找到一个位于白色区域的终点坐标，
#     该坐标尽量位于白色区域高度的三分之一处（自底向上），
#     且横坐标尽量接近图像的中间。
#     """
#     h = map_h
#     w = map_w
#
#     # 获取整个图像的白色区域（可通行区域）的坐标
#     white_coords = np.column_stack(np.where(map == 1))
#
#     if white_coords.size == 0:
#         # 如果没有白色区域，返回 None 或处理异常
#         return None
#
#     # 获取白色区域的最底部和最顶部的行号
#     min_y = np.min(white_coords[:, 0])  # 最顶部的白色像素行号
#     max_y = np.max(white_coords[:, 0])  # 最底部的白色像素行号
#
#     # 计算白色区域高度的三分之一位置（自底向上）
#     white_height = max_y - min_y
#     target_y = max_y - white_height // 3
#
#     # 限制 target_y 在图像范围内
#     target_y = max(0, min(target_y, h - 1))
#
#     # 在 target_y 行上，找到所有在白色区域内的像素
#     candidate_x = np.where(map[target_y, :] == 1)[0]
#
#     if candidate_x.size == 0:
#         # 如果目标行没有白色像素，向上或向下搜索最近的白色行
#         offset = 1
#         found = False
#         while not found and (target_y - offset >= 0 or target_y + offset < h):
#             # 向上搜索
#             if target_y - offset >= 0:
#                 candidate_x = np.where(map[target_y - offset, :] == 1)[0]
#                 if candidate_x.size > 0:
#                     target_y = target_y - offset
#                     found = True
#                     break
#             # 向下搜索
#             if target_y + offset < h:
#                 candidate_x = np.where(map[target_y + offset, :] == 1)[0]
#                 if candidate_x.size > 0:
#                     target_y = target_y + offset
#                     found = True
#                     break
#             offset += 1
#
#         if not found:
#             # 如果仍未找到，返回 None 或处理异常
#             return None
#
#     # 在候选的 x 坐标中，选择最接近图像中间的一个
#     center_x = w // 2
#     distances = np.abs(candidate_x - center_x)
#     min_distance_index = np.argmin(distances)
#     target_x = candidate_x[min_distance_index]
#
#     # 返回终点坐标，注意 (y, x) 的顺序
#     return (target_y, target_x)
def find_final(map, map_h, map_w, white_threshold=20):
    """
    在可变尺寸的图像中，找到一个位于白色区域的终点坐标，
    该坐标尽量位于白色区域高度的三分之一处（自底向上），
    且横坐标尽量接近图像的中间。

    Args:
    - map: 二值化图像，0 表示不可通行区域，1 表示可通行区域
    - map_h: 图像的高度
    - map_w: 图像的宽度
    - white_threshold: 用于过滤噪点的白色像素最小阈值

    Returns:
    - (y, x): 终点坐标
    """
    h = map_h
    w = map_w

    # 获取整个图像的白色区域（可通行区域）的坐标
    white_coords = np.column_stack(np.where(map == 1))

    if white_coords.size == 0:
        # 如果没有白色区域，返回 None 或处理异常
        return None

    # 获取每一行中白色像素的数量，并根据阈值筛选出有效的白色行
    white_pixel_counts = np.sum(map == 1, axis=1)
    valid_rows = np.where(white_pixel_counts > white_threshold)[0]  # 只保留白色像素数量超过阈值的行

    if valid_rows.size == 0:
        # 如果没有满足条件的白色行，返回 None
        return None

    # 获取白色区域的最底部和最顶部的有效行号
    min_y = np.min(valid_rows)  # 最顶部的有效白色像素行号
    max_y = np.max(valid_rows)  # 最底部的有效白色像素行号

    # 计算白色区域高度的三分之一位置（自底向上）
    white_height = max_y - min_y
    target_y = max_y - white_height // 3

    # 限制 target_y 在图像范围内
    target_y = max(0, min(target_y, h - 1))

    # 在 target_y 行上，找到所有在白色区域内的像素
    candidate_x = np.where(map[target_y, :] == 1)[0]

    if candidate_x.size == 0:
        # 如果目标行没有白色像素，向上或向下搜索最近的白色行
        offset = 1
        found = False
        while not found and (target_y - offset >= 0 or target_y + offset < h):
            # 向上搜索
            if target_y - offset >= 0:
                candidate_x = np.where(map[target_y - offset, :] == 1)[0]
                if candidate_x.size > 0:
                    target_y = target_y - offset
                    found = True
                    break
            # 向下搜索
            if target_y + offset < h:
                candidate_x = np.where(map[target_y + offset, :] == 1)[0]
                if candidate_x.size > 0:
                    target_y = target_y + offset
                    found = True
                    break
            offset += 1

        if not found:
            # 如果仍未找到，返回 None 或处理异常
            return None

    # 在候选的 x 坐标中，选择最靠近 target_y 对应白色区域中间的横坐标
    min_x, max_x = np.min(candidate_x), np.max(candidate_x)
    target_x = (min_x + max_x) // 2  # 选择该行白色区域的中间位置

    # 返回终点坐标，注意 (y, x) 的顺序
    return (target_y, target_x)


###


def find_end_point_bak(passable, img_height):
    relax = 10.5
    end_y = img_height * 2 / 3 - relax
    end_x_left = None
    end_x_right = None
    is_passable = False

    for [x, y] in passable:
        if y == end_y:
            if end_x_left is None or end_x_right is None:
                end_x_left = end_x_right = x
            else:
                end_x_right = (x if x < end_x_right else end_x_right)
                end_x_left = (x if x > end_x_left else end_x_left)
        if y >= img_height - relax:
            is_passable = True

    if end_x_left is not None and end_x_right is not None:
        return ([(end_x_left + end_x_right) / 2, end_y] if is_passable else None)
    else:
        return None


def find_end_point(passable, img_height):
    relax = 10.5
    line = [0, 10, 20, 40, 60]

    for l in line:
        point_x = []
        length = 0
        is_passable = None
        end_y1 = img_height * 2 / 3 - relax + l
        # end_y1 = img_height * 1 / 3 - relax + l

        for [x, y] in passable:
            if y == end_y1:
                point_x.append(x)
            if y >= img_height - relax:
                is_passable = True
        # print(end_y1, point_x, is_passable)
        point_x.sort()
        # print('point',len(point_x))

        if len(point_x) % 2 == 1:
            continue

        if len(point_x) % 2 == 0 and len(point_x) != 0:
            for i in range(int(len(point_x) / 2)):
                if (point_x[2 * i + 1] - point_x[2 * i]) >= length:
                    length = point_x[2 * i + 1] - point_x[2 * i]
                    mid_x = (point_x[2 * i + 1] + point_x[2 * i]) / 2
                else:
                    length = length
                    mid_x = mid_x
            return ([mid_x, end_y1] if is_passable else None)

    return None


def img_patch(map, map_h, map_w):
    """
    为识别结果中中间漏识别的点做补充
    把一片区域中被包围的没有识别到的点补充进去
    将被包围的0变成255
    """

    r = map_h
    c = map_w

    def bfs(map, i, j):
        if 0 <= i < r and 0 <= j < c and map[i][j] == 0:
            map[i][j] = 500
            bfs(map, i, j + 1)
            bfs(map, i, j - 1)
            bfs(map, i + 1, j)
            bfs(map, i - 1, j)

    for i in range(r):
        for j in range(c):
            if (i == 0 or i == r - 1 or j == 0 or j == c - 1) and map[i][j] == 0:
                bfs(map, i, j)

    for i in range(r):
        for j in range(c):
            map[i][j] = 0 if map[i][j] == 500 else 255

    return map


def pathplan(XX_numpy, end_point: list = None):
    # XX = torch.zeros(int(h), int(w), 1, dtype=torch.long, device="cuda:0")
    # XX_numpy = (XX * 255).byte().cpu().numpy()  # 把tensor转numpy,必须加cpu()
    # XX_numpy = np.concatenate((XX_numpy, XX_numpy, XX_numpy), axis=-1)
    """
    找出路径中点并规划路径
    """
    # with timer("pathplanning"):
    gain = 10
    XX_numpy = Down_Sample(XX_numpy, 1 / gain)

    # planning_map = XX_numpy[:, :, 0]
    planning_map = XX_numpy
    map_w = planning_map.shape[1]
    map_h = planning_map.shape[0]

        # 修复噪点
        # for i in range(map_w):
        #     for j in range(map_h):
        #         if planning_map[j][i] < 10:
        #             planning_map[j][i] = 0
        #         if planning_map[j][i] > 248:
        #             planning_map[j][i] = 255

        # 补充图像中坏点
        # planning_map = img_patch(planning_map, map_h, map_w)

        # 找终点
    if end_point is None:
        goal = find_final(planning_map, map_h, map_w)#高宽
        print("end_point is None")
    else:
        goal = end_point
        # goal = (int(goal[0] / gain), int(goal[1] / gain))
        # # goal = (int(goal[1] / gain), int(goal[0] / gain))

        # 找路径
    if goal:
        # start = (int(map_w/2) ,map_h-1)
        start = (map_h - 1, int(map_w / 2))
        # print(start, goal)
        bastar = BidirectionalAStar(start, goal, XX_numpy, "euclidean")
        path, visited_fore, visited_back = bastar.searching()
        if path:
            def temp(turple):
                return (turple[0] * gain, turple[1] * gain)

            path = list(map(temp, path))
            return path  ###返回的路径类型为 List[Tuple[int, int]]

    return None
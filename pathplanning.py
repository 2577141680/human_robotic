from turtledemo.forest import start

import cv2
from Bidirectional_a_star import BidirectionalAStar
# from predict import timer
import operator


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
def find_final(map, map_h, map_w):
    """
    找最远端路面重点的
    """
    h = map_h
    w = map_w
    N = 10
    white_line_threshold = int(w / N)
    count_line = 0
    route = []
    start_x = start_y = 0
    end_x = end_y = 0

    for j in range(h):  # 遍历整个高度
        for i in range(w):  # 遍历整个宽度
            if map[j, i] == 1:  # 当前像素为白色区域
                if count_line == 0:  # 新段的开始
                    start_x = i
                    start_y = j  # 记录白色区域的开始行
                count_line += 1  # 计数
            elif count_line > 0:  # 遇到黑色区域，表示白色区域结束
                end_x = i
                end_y = j  # 记录白色区域的结束行

                if count_line > white_line_threshold:
                    # 计算纵向三分之一的位置
                    height = end_y - start_y
                    one_third_height = height // 3
                    mid_y = end_y - one_third_height  # 自底向上三分之一

                    # 确保 mid_y 在白色区域内
                    if mid_y > h*2//3 and map[mid_y, (start_x + end_x) // 2] == 1:
                        mid_x = (start_x + end_x) // 2  # 横坐标在白色区域中间
                        route.append((mid_x, mid_y))  # 记录坐标

                count_line = 0  # 重置计数

        # 存一个就行
        if len(route) > 0:
            break

    return route


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
        goal = find_final(planning_map, map_h, map_w)[0]#高宽
        print("end_point is None")
    else:
        goal = end_point
        # goal = (int(goal[0] / gain), int(goal[1] / gain))
        # # goal = (int(goal[1] / gain), int(goal[0] / gain))

        # 找路径
    if goal:
        start = (int(map_w / 2), map_h - 2)
        # start = (map_h - 1, int(map_w / 2))
        # print(start, goal)
        bastar = BidirectionalAStar(start, goal, XX_numpy, "euclidean")
        path, visited_fore, visited_back = bastar.searching()
        if path:
            def temp(turple):
                return (turple[0] * gain, turple[1] * gain)

            path = list(map(temp, path))
            return path  ###返回的路径类型为 List[Tuple[int, int]]

    return None
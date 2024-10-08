# -*- coding: utf-8 -*-

import copy
from collections import namedtuple
import time

import numpy as np

Rectangle = namedtuple("Rectangle", "top bottom left right")


class Projection:
    """ Projection direction """
    Xy = 0
    Xz = 1
    Yx = 2
    Yz = 3
    Zx = 4
    Zy = 5
    

def obs_to_list(observation, container_size):
    if not isinstance(observation, np.ndarray):
        obs_info = observation.cpu().numpy()
    else:
        obs_info = observation

    obs_info = obs_info.reshape((4, -1))
    new_box_l = int(obs_info[1][0])
    new_box_w = int(obs_info[2][0])
    new_box_h = int(obs_info[3][0])

    plain = obs_info[0].reshape((container_size[0], container_size[1]))

    return plain, (new_box_l, new_box_w, new_box_h)


def find_rectangles(arr: list, height: int) -> list:
    """
    detect rectangles from 2d list (height map)
    Args:
        arr: 2d list
    Returns:
        rectangles: list of Rectangle
    """
    # Deeply copy the array so that it can be modified safely
    arr = [row[:] for row in arr]

    rectangles = []

    for top, row in enumerate(arr):
        start = 0

        # Look for rectangles whose top row is here
        while True:
            try:
                left = row.index(0, start)
            except ValueError:
                break

            # Set start to one past the last 0 in the contiguous line of 0s
            try:
                start = row.index(1, left)
            except ValueError:
                start = len(row)

            right = start - 1

            # if (  # Width == 1
            #         left == right):  # or
            #     # There are 0s above
            #     # top > 0 and not all(arr[top - 1][left:right + 1])):
            #     continue

            bottom = top + 1
            while (bottom < len(arr) and
                   # No extra zeroes on the sides
                   # (left == 0 or arr[bottom][left - 1]) and
                   # (right == len(row) - 1 or arr[bottom][right + 1]) and
                   # All zeroes in the row
                   not any(arr[bottom][left:right + 1])):
                bottom += 1

            # The loop ends when bottom has gone too far, so backtrack
            bottom -= 1

            # if (  # Height == 1
            #         bottom == top):  # or
            #     # There are 0s beneath
            #     # (bottom < len(arr) - 1 and
            #     #  not all(arr[bottom + 1][left:right + 1]))):
            #     continue

            # rectangles.append(Rectangle(top, bottom, left, right))
            rectangles.append([bottom - top + 1, right - left + 1, height, top, left, 0])

            # Remove the rectangle so that it doesn't affect future searches
            for i in range(top, bottom + 1):
                arr[i][left:right + 1] = [1] * (right + 1 - left)

    return rectangles


def extract_items_from_heightmap(observation: np.ndarray) -> list:
    """
    Args:
        observation: heightmap 2d array

    Returns:
        item_list: list of items extracted from current height map
    """
    # time_start = time.time()

    height_arr = copy.deepcopy(observation)
    height_value = np.unique(height_arr)
    height_value_without_zero = height_value[np.nonzero(height_value)]

    rectangles = []  # a rectangle represents a item
    item_list = []  # extracted items

    # distinguish objects by height
    for height in height_value_without_zero:
        state_height = height_arr - height
        state_height[np.nonzero(state_height)] = 1
        state_height_list = state_height.tolist()

        rectangles = find_rectangles(state_height_list, height)
        # rectangles = sweep(state_height_list)
        # print("rectangles: ", rectangles)
        
        # for rect in rectangles:
        #     length = rect.bottom - rect.top + 1
        #     width = rect.right - rect.left + 1
    
        #     item = [length, width, height, rect.top, rect.left, 0]
        #     item_list.append(item)
        item_list.extend(rectangles)  # (length, width, height, x, y, z)

    # time_end = time.time()
    # print("time cost: ", (time_end - time_start) * 1000, "ms")
    return item_list


def can_take_projection(new_item, placed_item, ep_dir: int, proj_dir: int) -> bool:
    
    """
    function returning true if an EP(generation direction: ep_dir) of item k can be projected(projection direction:
    proj_dir) on the item i.
    :param new_item:
    :param placed_item:
    :param ep_dir: (number 0,1,2 corresponding to x, y, z), extreme point generation direction
    :param proj_dir: extreme point projection direction
    :return: bool
    """
    new_dim = new_item[:3]
    placed_dim = placed_item[:3]

    remain_dir = 3 - ep_dir - proj_dir
    epsilon = 0.0
    proj_flag = True

    if placed_item[-3:][proj_dir] + placed_dim[proj_dir] > new_item[-3:][proj_dir] - epsilon:
        # i.e. piece is further from axis in projection direction
        proj_flag = False
        return proj_flag

    if placed_item[-3:][ep_dir] > new_item[-3:][ep_dir] + new_dim[ep_dir] - epsilon:
        # i.e. piece too far
        proj_flag = False
        return proj_flag

    if placed_item[-3:][ep_dir] + placed_dim[ep_dir] < new_item[-3:][ep_dir] + new_dim[ep_dir] + epsilon:
        # i.e. piece not far enough
        proj_flag = False
        return proj_flag

    if placed_item[-3:][remain_dir] > new_item[-3:][remain_dir] - epsilon:
        # i.e. piece too far
        proj_flag = False
        return proj_flag

    if placed_item[-3:][remain_dir] + placed_dim[remain_dir] < new_item[-3:][remain_dir] + epsilon:
        # i.e. piece not far enough
        proj_flag = False
        return proj_flag

    return proj_flag



if __name__ == "__main__":
    # test find_rectangles
    state = [[15, 15, 15, 15, 15, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 18, 0, 0, 0, 0],
             [15, 15, 15, 15, 15, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 18, 0, 0, 0, 0],
             [15, 15, 15, 15, 15, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 18, 0, 0, 0, 0],
             [15, 15, 15, 15, 15, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 18, 0, 0, 0, 0],
             [15, 15, 15, 15, 15, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 18, 0, 0, 0, 0],
             [15, 15, 15, 15, 15, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 18, 0, 0, 0, 0],
             [15, 15, 15, 15, 15, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 18, 0, 0, 0, 0],
             [15, 15, 15, 15, 15, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 18, 0, 0, 0, 0],
             [24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 15, 15, 15, 15, 15, 5, 0, 0, 0, 0],
             [24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 15, 15, 15, 15, 15, 5, 0, 0, 0, 0],
             [24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 15, 15, 15, 15, 15, 5, 0, 0, 0, 0],
             [24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 15, 15, 15, 15, 15, 5, 0, 0, 0, 0],
             [24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 15, 15, 15, 15, 15, 5, 0, 0, 0, 0],
             [24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 15, 15, 15, 15, 15, 0, 0, 0, 0, 0],
             [24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 15, 15, 15, 15, 15, 0, 0, 0, 0, 0],
             [24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 15, 15, 15, 15, 15, 0, 0, 0, 0, 0],
             [22, 22, 22, 22, 22, 12, 12, 12, 12, 12, 15, 15, 15, 15, 15, 15, 0, 0, 0, 0],
             [22, 22, 22, 22, 22, 12, 12, 12, 12, 12, 15, 15, 15, 15, 15, 15, 0, 0, 0, 0],
             [22, 22, 22, 22, 22, 12, 12, 12, 12, 12, 15, 15, 15, 15, 15, 15, 0, 0, 0, 0],
             [22, 22, 22, 22, 22, 12, 12, 12, 12, 12, 15, 15, 15, 15, 15, 15, 0, 0, 0, 0],
             [22, 22, 22, 22, 22, 12, 12, 12, 12, 12, 15, 15, 15, 15, 15, 15, 0, 0, 0, 0],
             [22, 22, 22, 22, 22, 12, 12, 12, 12, 12, 10, 10, 10, 10, 10, 0, 0, 0, 0, 0],
             [22, 22, 22, 22, 22, 12, 12, 12, 12, 12, 10, 10, 10, 10, 10, 0, 0, 0, 0, 0],
             [22, 22, 22, 22, 22, 12, 12, 12, 12, 12, 10, 10, 10, 10, 10, 0, 0, 0, 0, 0],
             ]

    rectangles = extract_items_from_heightmap(np.array(state))
    print(rectangles)

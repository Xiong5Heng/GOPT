import copy
import itertools
from functools import reduce
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.spatial import ConvexHull
from matplotlib.path import Path

from .ems import compute_ems
from .utils import *
from .box import Box


class Container(object):
    def __init__(self, length=10, width=10, height=10, rotation=True):
        self.dimension = np.array([length, width, height])
        self.heightmap = np.zeros(shape=(length, width), dtype=np.int32)
        self.can_rotate = rotation
        # packed box list
        self.boxes = []
        # record rotation information
        self.rot_flags = []
        self.height = height
        self.candidates = [[0, 0, 0]]

    def print_heightmap(self):
        print("container heightmap: \n", self.heightmap)

    def get_heightmap(self):
        """
        get the heightmap for the ideal situation
        Returns:

        """
        plain = np.zeros(shape=self.dimension[:2], dtype=np.int32)
        for box in self.boxes:
            plain = self.update_heightmap(plain, box)
        return plain

    def update_heightmap_vision(self, vision):
        """
        TODO
        Args:
            vision:

        Returns:

        """
        self.heightmap = vision

    @staticmethod
    def update_heightmap(plain, box):
        """
        update heightmap
        Args:
            plain:
            box:

        Returns:

        """
        plain = copy.deepcopy(plain)
        le = box.pos_x
        ri = box.pos_x + box.size_x
        up = box.pos_y
        do = box.pos_y + box.size_y
        max_h = np.max(plain[le:ri, up:do])
        max_h = max(max_h, box.pos_z + box.size_z)
        plain[le:ri, up:do] = max_h
        return plain

    def get_box_list(self):
        vec = list()
        for box in self.boxes:
            vec += box.standardize()
        return vec

    def get_plain(self):
        return copy.deepcopy(self.heightmap)

    def get_action_space(self):
        return self.dimension[0] * self.dimension[1]

    def get_action_mask(self, next_box, scheme="heightmap"):
        action_mask = np.zeros(shape=(self.dimension[0], self.dimension[1]), dtype=np.int32)

        if scheme == "heightmap":
            candidates_xy, extra_corner_xy = self.candidate_from_heightmap(next_box, self.can_rotate)

            for xy in candidates_xy:
                if self.check_box(next_box, xy) > -1:
                    action_mask[xy[0], xy[1]] = 1
            for xy in extra_corner_xy[:3]:
                if self.check_box(next_box, xy) > -1:
                    action_mask[xy[0], xy[1]] = 1

            if self.can_rotate:
                rotated_box = [next_box[1], next_box[0], next_box[2]]
                action_mask_rot = np.zeros_like(action_mask)

                for xy in candidates_xy:
                    if self.check_box(rotated_box, xy) > -1:
                        action_mask_rot[xy[0], xy[1]] = 1
                for xy in extra_corner_xy[-3:]:
                    if self.check_box(rotated_box, xy) > -1:
                        action_mask_rot[xy[0], xy[1]] = 1

                action_mask = np.hstack((action_mask.reshape((-1,)), action_mask_rot.reshape((-1,))))

        elif scheme == "EP":
            candidates_xy, extra_corner_xy = self.candidate_from_EP(next_box, self.can_rotate)
            # extra_corner_xy = []
            for xy in candidates_xy:
                if self.check_box(next_box, xy) > -1:
                    action_mask[xy[0], xy[1]] = 1
            for xy in extra_corner_xy[:3]:
                if self.check_box(next_box, xy) > -1:
                    action_mask[xy[0], xy[1]] = 1

            if self.can_rotate:
                rotated_box = [next_box[1], next_box[0], next_box[2]]
                action_mask_rot = np.zeros_like(action_mask)

                for xy in candidates_xy:
                    if self.check_box(rotated_box, xy) > -1:
                        action_mask_rot[xy[0], xy[1]] = 1
                for xy in extra_corner_xy[-3:]:
                    if self.check_box(rotated_box, xy) > -1:
                        action_mask_rot[xy[0], xy[1]] = 1

                action_mask = np.hstack((action_mask.reshape((-1,)), action_mask_rot.reshape((-1,))))

        elif scheme == "FC":
            x_list = list(range(self.dimension[0]))
            y_list = list(range(self.dimension[1]))
            candidates_xy = list(itertools.product(x_list, y_list))

            for xy in candidates_xy:
                if self.check_box(next_box, xy) > -1:
                    action_mask[xy[0], xy[1]] = 1

            if self.can_rotate:
                rotated_box = [next_box[1], next_box[0], next_box[2]]
                action_mask_rot = np.zeros_like(action_mask)

                for xy in candidates_xy:
                    if self.check_box(rotated_box, xy) > -1:
                        action_mask_rot[xy[0], xy[1]] = 1
                
                action_mask = np.hstack((action_mask.reshape((-1,)), action_mask_rot.reshape((-1,))))

            # assert False, 'No FC implementation'
        else:
            assert False, 'Wrong candidate generation scheme'

        # if all actions are invalid, set all mask is 1 and perform any action to end this episode
        if action_mask.sum() == 0:
            action_mask[:] = 1

        return action_mask.reshape(-1).tolist()

    def check_box(self, box_size, pos_xy, benchmark=False):
        """
            check
            1. whether cross the border
            2. check stability
        Args:
            box_size:
            pos_xy:

        Returns:

        """
        if pos_xy[0] + box_size[0] > self.dimension[0] or pos_xy[1] + box_size[1] > self.dimension[1]:
            return -1

        pos_z = np.max(self.heightmap[pos_xy[0]:pos_xy[0] + box_size[0], pos_xy[1]:pos_xy[1] + box_size[1]])

        # whether cross the broder
        if pos_z + box_size[2] > self.dimension[2]:
            return -1
        
        # check stability
        if benchmark:
            # zhao AAAI2021 paper
            rec = self.heightmap[pos_xy[0]:pos_xy[0] + box_size[0], pos_xy[1]:pos_xy[1] + box_size[1]]
            r00 = rec[0, 0]
            r10 = rec[box_size[0] - 1, 0]
            r01 = rec[0, box_size[1] - 1]
            r11 = rec[box_size[0] - 1, box_size[1] - 1]
            rm = max(r00, r10, r01, r11)
            sc = int(r00 == rm) + int(r10 == rm) + int(r01 == rm) + int(r11 == rm)
            # at least 3 support point
            if sc < 3:
                return -1
            # check area and corner
            max_area = np.sum(rec == pos_z)
            area = box_size[0] * box_size[1]
            # 
            if max_area / area > 0.95:
                return pos_z
            if rm == pos_z and sc == 3 and max_area/area > 0.85:
                return pos_z
            if rm == pos_z and sc == 4 and max_area/area > 0.50:
                return pos_z
        else:
            if self.is_stable(box_size, [pos_xy[0], pos_xy[1], pos_z]):
                return pos_z

        return -1

    def check_box_ems(self, box_size, ems, benchmark=False):
        """
            check
            1. whether cross the border
            2. check stability
        Args:
            box_size:
            pos_xy:

        Returns:

        """
        if ems[3] - ems[0] < box_size[0] or ems[4] - ems[1] < box_size[1] or ems[5] - ems[2] < box_size[2]:
            return -1

        if ems[0] + box_size[0] > self.dimension[0] or ems[1] + box_size[1] > self.dimension[1]:
            return -1

        pos_z = np.max(self.heightmap[ems[0]:ems[0] + box_size[0], ems[1]:ems[1] + box_size[1]])

        # whether cross the broder
        if pos_z + box_size[2] > self.dimension[2]:
            return -1
        
        # check stability
        if self.is_stable(box_size, [ems[0], ems[1], pos_z]):
            return pos_z

        return -1

    def is_stable(self, dimension, position) -> bool:
        """
            check stability for 3D packing
        Args:
            dimension:
            position:

        Returns:

        """
        def on_segment(P1, P2, Q):
            if ((Q[0] - P1[0]) * (P2[1] - P1[1]) == (P2[0] - P1[0]) * (Q[1] - P1[1]) and
                min(P1[0], P2[0]) <= Q[0] <= max(P1[0], P2[0]) and
                min(P1[1], P2[1]) <= Q[1] <= max(P1[1], P2[1])):
                return True
            else:
                return False

        # item on the ground of the bin
        if position[2] == 0:
            return True

        # calculate barycentric coordinates, -1 means coordinate indices start at zero
        x_1 = position[0]
        x_2 = x_1 + dimension[0] - 1
        y_1 = position[1]
        y_2 = y_1 + dimension[1] - 1
        z = position[2] - 1
        obj_center = ((x_1 + x_2) / 2, (y_1 + y_2) / 2)

        # valid points right under this object
        points = []
        for x in range(x_1, x_2 + 1):
            for y in range(y_1, y_2 + 1):
                if self.heightmap[x][y] == (z + 1):
                    points.append([x, y])

        # the support area is more than half of the bottom surface of the item
        if len(points) > dimension[0] * dimension[1] * 0.5:
            return True
        
        if len(points) == 0 or len(points) == 1: 
            return False
        elif len(points) == 2: # whether the center lies on the line of the two points
            return on_segment(points[0], points[1], obj_center)
        else:
            # calculate the convex hull of the points
            points = np.array(points)
            try:
                convex_hull = ConvexHull(points)
            except:
                # error means co-lines
                start_p = min(points, key=lambda p: [p[0], p[1]])
                end_p = max(points, key=lambda p: [p[0], p[1]])
                return on_segment(start_p, end_p, obj_center)

            hull_path = Path(points[convex_hull.vertices])

            return hull_path.contains_point(obj_center)

    def get_volume_ratio(self):
        vo = reduce(lambda x, y: x + y, [box.size_x * box.size_y * box.size_z for box in self.boxes], 0.0)
        mx = self.dimension[0] * self.dimension[1] * self.dimension[2]
        ratio = vo / mx
        assert ratio <= 1.0
        return ratio

    # 1d index -> 2d plain coordinate
    def idx_to_position(self, idx):
        """
        TODO
        Args:
            idx:

        Returns:

        """
        lx = idx // self.dimension[1]
        ly = idx % self.dimension[1]
        return lx, ly

    def position_to_index(self, position):
        assert len(position) == 2
        assert position[0] >= 0 and position[1] >= 0
        assert position[0] < self.dimension[0] and position[1] < self.dimension[1]
        return position[0] * self.dimension[1] + position[1]

    def place_box(self, box_size, pos, rot_flag):
        """ place box in the position (index), then update heightmap
        :param box_size:
        :param idx:
        :param rot_flag:
        :return:
        """
        if not rot_flag:
            size_x = box_size[0]
            size_y = box_size[1]
        else:
            size_x = box_size[1]
            size_y = box_size[0]
        size_z = box_size[2]
        plain = self.heightmap
        new_h = self.check_box([size_x, size_y, size_z], [pos[0], pos[1]])
        if new_h != -1:
            self.boxes.append(Box(size_x, size_y, size_z, pos[0], pos[1], pos[2]))  # record rotated box
            self.rot_flags.append(rot_flag)
            self.heightmap = self.update_heightmap(plain, self.boxes[-1])
            self.height = max(self.height, pos[2] + size_z)
            return True
        return False

    def candidate_from_heightmap(self, next_box, max_n) -> list:
        """
        get the x and y coordinates of candidates
        Args:
            next_box:
            can_rotate:

        Returns:

        """
        heightmap = copy.deepcopy(self.heightmap)

        corner_list = []
        # hm_diff: height differences of neighbor columns, padding 0 in the front
        # x coordinate
        # heightmap: [r0, r1, r2, r3, r4, r5, ..., rn]
        # insert: [r0, r0, r1, r2, r3, r4, r5, ..., rn]
        hm_diff_x = np.insert(heightmap, 0, heightmap[0, :], axis=0)
        # delete: [r0, r0, r1, r2, r3, r4, ..., rn-1]
        hm_diff_x = np.delete(hm_diff_x, len(hm_diff_x) - 1, axis=0)
        # hm_diff_x: [0, r1-r0, r2-r1, r3-r2, r4-r3, r5-r4, rn-r(n-1)]
        hm_diff_x = heightmap - hm_diff_x

        # y coordinate
        hm_diff_y = np.insert(heightmap, 0, heightmap[:, 0], axis=1)
        hm_diff_y = np.delete(hm_diff_y, len(hm_diff_y.T) - 1, axis=1)
        # hm_diff_y: [0, c1-c0, c2-c1, c3-c2, c4-c3, c5-c4, cn-c(n-1)]
        hm_diff_y = heightmap - hm_diff_y

        # get the xy coordinates of all left-deep-bottom corners
        corner_x_list = np.array(np.nonzero(hm_diff_x)).T.tolist()
        corner_y_list = np.array(np.nonzero(hm_diff_y)).T.tolist()

        corner_xy_list = []
        corner_xy_list.append([0, 0])

        for xy in corner_x_list:
            x, y = xy
            if y != 0 and [x, y - 1] in corner_x_list:
                # if heightmap[x, y] == heightmap[x, y - 1] and hm_diff_x[x, y] == hm_diff_x[x, y - 1]:
                if heightmap[x, y] == heightmap[x, y - 1]:
                    continue
            corner_xy_list.append(xy)
        for xy in corner_y_list:
            x, y = xy
            if x != 0 and [x - 1, y] in corner_y_list:
                # if heightmap[x, y] == heightmap[x - 1, y] and hm_diff_x[x, y] == hm_diff_x[x - 1, y]:
                if heightmap[x, y] == heightmap[x - 1, y]:
                    continue
            if xy not in corner_xy_list:
                corner_xy_list.append(xy)

        candidate_x, candidate_y = zip(*corner_xy_list)
        # remove duplicate elements
        candidate_x = list(set(candidate_x))
        candidate_y = list(set(candidate_y))

        # get corner_list
        corner_list = list(itertools.product(candidate_x, candidate_y))
        candidates = [] 

        for xy in corner_list:
            z = self.check_box(next_box, xy)
            if z > -1:
                # candidates.append([xy[0], xy[1], z, 0])
                candidates.append([xy[0], xy[1], z, xy[0] + next_box[0], xy[1] + next_box[1], z + next_box[2]])
        
        if self.can_rotate:
            rotated_box = [next_box[1], next_box[0], next_box[2]]
            for xy in corner_list:
                z = self.check_box(rotated_box, xy)
                if z > -1:
                    # candidates.append([xy[0], xy[1], z, 1])
                    candidates.append([xy[0], xy[1], z, xy[0] + rotated_box[0], xy[1] + rotated_box[1], z + rotated_box[2]])

        # sort by z, y coordinate, then x
        candidates.sort(key=lambda x: [x[2], x[1], x[0]])

        if len(candidates) > max_n:
            candidates = candidates[:max_n]
        self.candidates = candidates
        return np.array(candidates)

    def candidate_from_EP(self, next_box, max_n) -> list:
        """
        calculate extreme points from items extracted from current heightmap
        Args:
            new_item:

        Returns:

        """
        heightmap = copy.deepcopy(self.heightmap)
        items_in = extract_items_from_heightmap(heightmap)
        new_eps = []
        new_eps.append([0, 0, 0])

        for k in range(len(items_in)):
            items_in_copy = copy.deepcopy(items_in)
            item_new = items_in_copy[k]
            new_dim = item_new[:3]
            new_pos = item_new[-3:]

            items_in_copy.pop(k)
            item_fitted = items_in_copy

            # add xoy, xoz, yoz planes for easy projection
            item_fitted.append([self.dimension[0], self.dimension[1], 0, 0, 0, 0])
            item_fitted.append([self.dimension[0], 0, self.dimension[2], 0, 0, 0])
            item_fitted.append([0, self.dimension[1], self.dimension[2], 0, 0, 0])

            max_bounds = [-1, -1, -1, -1, -1, -1]

            for i in range(len(item_fitted)):
                fitted_dim = item_fitted[i][:3]
                fitted_pos = item_fitted[i][-3:]
                project_x = fitted_dim[0] + fitted_pos[0]
                project_y = fitted_dim[1] + fitted_pos[1]
                project_z = fitted_dim[2] + fitted_pos[2]

                # Xy - new_eps[0]
                if can_take_projection(item_new, item_fitted[i], 0, 1) and project_y > max_bounds[Projection.Xy]:
                    new_eps.append([new_pos[0] + new_dim[0], project_y, new_pos[2]])
                    max_bounds[Projection.Xy] = project_y

                # Xz - new_eps[1]
                if can_take_projection(item_new, item_fitted[i], 0, 2) and project_z > max_bounds[Projection.Xz]:
                    new_eps.append([new_pos[0] + new_dim[0], new_pos[1], project_z])
                    max_bounds[Projection.Xz] = project_z

                # Yx - new_eps[2]
                if can_take_projection(item_new, item_fitted[i], 1, 0) and project_x > max_bounds[Projection.Yx]:
                    new_eps.append([project_x, new_pos[1] + new_dim[1], new_pos[2]])
                    max_bounds[Projection.Yx] = project_x

                # Yz - new_eps[3]
                if can_take_projection(item_new, item_fitted[i], 1, 2) and project_z > max_bounds[Projection.Yz]:
                    new_eps.append([new_pos[0], new_pos[1] + new_dim[1], project_z])
                    max_bounds[Projection.Yz] = project_z

                # Zx - new_eps[4]
                if can_take_projection(item_new, item_fitted[i], 2, 0) and project_x > max_bounds[Projection.Zx]:
                    new_eps.append([project_x, new_pos[1], new_pos[2] + new_dim[2]])
                    max_bounds[Projection.Zx] = project_x

                # Zy - new_eps[5]
                if can_take_projection(item_new, item_fitted[i], 2, 1) and project_y > max_bounds[Projection.Zy]:
                    new_eps.append([new_pos[0], project_y, new_pos[2] + new_dim[2]])
                    max_bounds[Projection.Zy] = project_y

        new_eps = [ep for ep in new_eps if not (ep[0] == self.dimension[0] or 
                                                ep[1] == self.dimension[1] or 
                                                ep[2] == self.dimension[2])]

        # only need x, y
        new_eps = np.array(new_eps, dtype=np.int32)

        # remove duplicates
        new_eps = np.unique(new_eps, axis=0)
        candidates = new_eps.tolist()
        candidates.sort(key=lambda x: [x[2], x[1], x[0]])
        mask = np.zeros((2, max_n), dtype=np.int8)

        if len(candidates) > max_n:
            candidates = candidates[:max_n]

        for id, ep in enumerate(candidates):
            z = self.check_box(next_box, ep)
            if z > -1 and z == ep[2]:
                mask[0, id] = 1 
        if self.can_rotate:
            rotated_box = [next_box[1], next_box[0], next_box[2]]
            for id, ep in enumerate(candidates):
                z = self.check_box(rotated_box, ep)
                if z > -1 and z == ep[2]:
                    mask[1, id] = 1 

        self.candidates = candidates
        return np.array(candidates), mask
    
    def candidate_from_EMS(self, 
        next_box, 
        max_n
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ calculate Empty Maximum Space from items extracted from current heightmap

        Args:
            next_box (_type_): _description_
            max_n (_type_): _description_

        Returns:
            list: _description_
        """
        heightmap = copy.deepcopy(self.heightmap)
        # left-bottom & right-top pos [bx, by, bz, tx, ty, tz], 
        # dimension: [tx-bx, ty-by, tz-bz],
        all_ems = compute_ems(heightmap, container_h=self.dimension[2])  

        candidates = all_ems
        mask = np.zeros((2, max_n), dtype=np.int8)
        
        # sort by z, y coordinate, then x
        candidates.sort(key=lambda x: [x[2], x[1], x[0]])

        if len(candidates) > max_n:
            candidates = candidates[:max_n]
        
        for id, ems in enumerate(candidates):
            if self.check_box_ems(next_box, ems) > -1:
                mask[0, id] = 1
        if self.can_rotate:
            rotated_box = [next_box[1], next_box[0], next_box[2]]
            for id, ems in enumerate(candidates):
                if self.check_box_ems(rotated_box, ems) > -1:
                    mask[1, id] = 1
        
        self.candidates = candidates
        return np.array(candidates), mask

    def candidate_from_FC(self, next_box) -> list:
        """
        calculate extreme points from items extracted from current heightmap
        Args:
            new_item:

        Returns:

        """
        candidates = []

        for x in range(self.dimension[0]):
            for y in range(self.dimension[1]):
                candidates.append([x, y, self.heightmap[x, y]])

        mask = np.zeros((2, self.dimension[0]*self.dimension[1]), dtype=np.int8)

        for id, xyz in enumerate(candidates):
            z = self.check_box(next_box, xyz)
            if z > -1 and z == xyz[2]:
                mask[0, id] = 1 
        if self.can_rotate:
            rotated_box = [next_box[1], next_box[0], next_box[2]]
            for id, xyz in enumerate(candidates):
                z = self.check_box(rotated_box, xyz)
                if z > -1 and z == xyz[2]:
                    mask[1, id] = 1 

        self.candidates = candidates
        return np.array(candidates), mask


if __name__ == '__main__':
    container = Container(3, 4, 10)
    container.heightmap = np.array([[5, 1, 4, 4], 
                                    [1, 5, 4, 1],
                                    [4, 4, 4, 1]])
    # container.print_heightmap()
    # next = [3, 2, 2]
    # mask = container.get_action_mask(next, True)

    # print(mask.reshape((-1, 10, 8)))
    print(container.place_box([3, 3, 3], [0, 0, 5], 0))

    # print(container.candidate_from_EMS([2, 2, 2], 10))

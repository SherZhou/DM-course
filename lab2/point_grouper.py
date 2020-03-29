# Group 聚类
import sys
import numpy as np
import lab2.mean_shift_utils as ms_utils

# 密度阈值
GROUP_DISTANCE_TOLERANCE = 0.08


class PointGrouper(object):
    # 通过Shifted_Point索引，将点分配到簇，或创建新簇
    def group_points(self, points):
        # group_assignment：shifted Point index[]；
        # groups：shifted Point[]
        group_assignment = []
        groups = []
        group_index = 0
        for point in points:
            nearest_group_index = self._determine_nearest_group(point, groups)
            if nearest_group_index is None:
                # create new group
                groups.append([point])
                group_assignment.append(group_index)
                group_index += 1
            else:
                group_assignment.append(nearest_group_index)
                groups[nearest_group_index].append(point)
        a=np.array(group_assignment)
        print("cluster:",np.unique(a,return_counts=True))
        return a

    # 将点发分到有最小距离的簇
    # 若有最小距离的Shifted_Point在groups中，返回该Shifted_Point在groups中索引，否则返回None
    def _determine_nearest_group(self, point, groups):
        nearest_group_index = None
        index = 0
        for group in groups:
            distance_to_group = self._distance_to_group(point, group)
            if distance_to_group < GROUP_DISTANCE_TOLERANCE:
                nearest_group_index = index
            index += 1
        return nearest_group_index

    # 到已有Shifted_point的最小距离
    def _distance_to_group(self, point, group):
        min_distance = sys.float_info.max
        for pt in group:
            dist = ms_utils.euclidean_dist(point, pt)
            if dist < min_distance:
                min_distance = dist
        return min_distance

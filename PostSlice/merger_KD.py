import numpy as np
from scipy.spatial import cKDTree
import numpy as np
import scipy.sparse as sp

def merge_mask_KD(masks: sp.csr_matrix, COMs: np.ndarray, timestamps: np.ndarray, max_dist=0.0, valid_arr=None):
    groups_kd = group_points_without_chaining(COMs, max_dist=max_dist)

    row_ind_all = np.hstack([iter_base*np.ones(groups_kd[iter_base].size) for iter_base in range(len(groups_kd))])
    col_ind_all = np.hstack([groups_kd[iter_base] for iter_base in range(len(groups_kd))])

    transfer_mat = sp.csr_matrix(([1]*row_ind_all.size, (row_ind_all, col_ind_all)), (len(groups_kd), masks.shape[0]))
    masks_merged = transfer_mat.dot(masks)
    timestamps_merged = [timestamps[row.nonzero()[1]] for row in transfer_mat]

    return masks_merged, timestamps_merged


def group_points_without_chaining(COMs, max_dist):
    n = COMs.shape[0]
    visited = np.zeros(n, dtype=bool)
    groups = []
    tree = cKDTree(COMs)
    
    for i in range(n):
        if not visited[i]:
            # 查询距离当前点小于max_dist的所有点（包括自身）
            neighbors = tree.query_ball_point(COMs[i], max_dist)
            neighbors = np.array(neighbors)
            # 筛选未访问的点
            unvisited = neighbors[~visited[neighbors]]
            if len(unvisited) > 0:
                # 记录当前组
                groups.append(unvisited)
                # 标记为已访问
                visited[unvisited] = True
    return groups
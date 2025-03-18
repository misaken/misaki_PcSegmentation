#######################################################################
## 以下剛体変換求める関係
## 回転行列は、ロール・ピッチ・ヨーの順番(x, y, z)でかけている
#######################################################################

import sys, pdb, torch
import numpy as np
from numpy.linalg import svd


def gen_T(r,t):
    rx, ry, rz = np.radians(r)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])

    R = Rz @ Ry @ Rx
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array(t)
    
    return T

#pyplotの座標の表現が行ベクトルで各点を表しているので、列ベクトルに修正した後変換したものをもとの形で返す。
def rigid_transform(T, X):
    # Tは同次座標, Xはpyplotで描画する前提の点集合。
    n_data = X.shape[0]
    X_T = X.T
    ones_vector = np.ones([1, n_data])
    X_ex = np.concatenate([X_T, ones_vector], 0)
    transformed = T @ X_ex
    return transformed[:-1, ].T

#pyplotの座標の表現が行ベクトルで各点を表しているので、列ベクトルに修正した後変換したものをもとの形で返す。
def rigid_transform_batch(T, X):
    batch_size = T.shape[0]
    # Tは同次座標, Xはpyplotで描画する前提の点集合。
    n_data = X.shape[0]
    X_T = X.T
    ones_vector = np.ones([1, n_data])
    X_ex = np.concatenate([X_T, ones_vector], 0) # 同次座標系にする
    X_ex = X_ex.reshape([1, 4, n_data]).repeat(batch_size, axis=0)
    transformed = (T @ X_ex)[:, :-1, :].transpose(0, 2, 1)
    return transformed

def svd2T(X, Y):
    """ 剛体変換を推定
    Args:
        X, Y(ndarray): (num_points, 3)の配列
    Returns:
        ndarray: 剛体変換T.(4, 4)の同次座標系
    """
    # pdb.set_trace()
    X = X.T
    Y = Y.T
    sample_size = X.shape[1]
    X_centroid = (np.sum(X, axis=1) / sample_size).reshape(3,1)
    Y_centroid = (np.sum(Y, axis=1) / sample_size).reshape(3,1)
    X_, Y_ = X - X_centroid, Y - Y_centroid
    A = X_ @ Y_.T
    u, s, v = svd(A)
    
    # R = v @ u.T
    R = v.T @ u.T
    detR = np.linalg.det(R)
    #print(f"det(R) = {detR}  s={s} v={v}")
    #回転行列の行列式が-1の場合は反射になっているので、その時の対処
    #特異値の一つが0の場合は、その対応するvの符号を反転する
    if -1.10000<=detR<=0.98888:
        if sum(s <= 1.0e-10) != 1: # 正しい剛体変換が得られない場合はinlierが少なくなるだけなので放置
            print("can not find R")
        else:
            reflect_index = np.where(s <= 1.0e-10)
            v[reflect_index] = v[reflect_index] * -1
            R = v.T @ u.T
    t = (Y_centroid - R @ X_centroid).reshape(3, )

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


def svd2T_batch(X:np.ndarray, Y: np.ndarray):
    """
    バッチを用いて特異値分解で剛体変換を求める
    Args:
        X,Y: (batch_size, 3, 3)
    Return:
        np.ndarray: (batch_size, 4, 4)
    """
    X = X.transpose(0, 2, 1)
    Y = Y.transpose(0, 2, 1)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    X = torch.tensor(X).to(device)
    Y = torch.tensor(Y).to(device)
    sample_size = X.shape[2]
    batch_size = X.shape[0]
    X_centroid = (torch.sum(X, dim=2) / sample_size).reshape(batch_size, 3, 1)
    Y_centroid = (torch.sum(Y, dim=2) / sample_size).reshape(batch_size, 3, 1)
    X_, Y_ = X - X_centroid, Y - Y_centroid
    A = torch.matmul(X_, Y_.transpose(2, 1))
    A = A.to(device)
    #u, s, v = svd(A)
    u, s, vT = torch.linalg.svd(A, full_matrices=True)
    # pdb.set_trace()
    
    # R = v.T @ u.T
    R = torch.bmm(vT.transpose(1, 2), u.transpose(1, 2))
    detR = torch.linalg.det(R)
    #print(f"det(R) = {detR}  s={s} vT={vT}")
    #回転行列の行列式が-1の場合は反射になっているので、その時の対処
    #特異値の一つが0の場合は、その対応するvの符号を反転する
    detR_int = torch.round(detR).to(torch.int)
    reflect_index = torch.where(detR_int == -1)[0]
    error_n = 0
    for i in reflect_index:
        if sum(s[i] <= 1.0e-10) != 1:
            # print("can not find R")
            error_n += 1
        else:
            sig_change_index = torch.where(s[i] <= 1.0e-10)
            vT[sig_change_index] = vT[sig_change_index] * -1
            # R = vT.T @ u.T
    R = torch.bmm(vT.transpose(1, 2), u.transpose(1, 2))
    t = (Y_centroid - R @ X_centroid).reshape(batch_size, 3)
    # t = (Y_centroid - R @ X_centroid)
    T = torch.stack([torch.eye(4) for _ in range(batch_size)])
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    print(f"could not find R : {error_n}") if error_n != 0 else None

    return T.numpy()


def calc_f_norm(A, B):
    return np.sqrt(((A-B)**2).sum())


# 基準点からの距離の近さの上位数割の範囲からサンプリング
def sampling_by_distance(pc, zero_idx, sample_size):
    random_state = np.random.RandomState(0)
    p_center_idx = random_state.choice(zero_idx)
    d = np.linalg.norm(pc - pc[p_center_idx], axis=1)
    n_points = pc.shape[0]
    near_idx = np.argsort(d)[:int(n_points*0.1)]
    sample_idx = np.append(random_state.choice(near_idx, sample_size-1, replace=False), p_center_idx)
    return sample_idx

def sampling_by_distance_batch(pc, zero_idx, batch_size, sample_size):
    random_state = np.random.RandomState(0)
    n_points = pc.shape[0]
    p_center_idx = random_state.choice(zero_idx, size=batch_size)
    d = np.linalg.norm(pc - pc[p_center_idx].reshape([batch_size, 1, 3]), axis=2) # d:(batch_size, n_data)
    near_idx = np.argsort(d)[:, :int(n_points*0.3)]
    sample_idx = np.append(np.array([random_state.choice(row, size=sample_size-1, replace=False) for row in near_idx]), p_center_idx.reshape([batch_size, 1]), axis=1)
    return sample_idx
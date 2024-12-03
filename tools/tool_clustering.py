#####################################
## クラスタリング関係
####################################

from sklearn.cluster import DBSCAN
import numpy as np
import sys, pdb, pickle


def dbscan(X):
    eps = 0.1
    min_samples = 5

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(X)

    labels = dbscan.labels_
    print("クラスタラベル:", labels)
    return labels


class MyKMeans():
    def __init__(self, metrics, max_iter=100, random_seed=0):
        self.max_iter = max_iter
        self.random_state = np.random.RandomState(random_seed)
        self.metrics = metrics
        self.distances = None # aicを求めるために用いる.クラスタが決定された際の距離
        self.centroids = None
        self.centroids_labelVec = None
        self.centroids_spatial = None
    
    def custom_distance_old(self, X, y):
        if self.metrics == "l0":
            distance = (X != y).sum(axis=1, dtype=float)
        elif self.metrics == "l2":
            distance = ((X - y)**2).sum(axis=1, dtype=float)
        else:
            sys.exit("Could not fid such a metrics. Select another one.")
        return distance
    
    def custom_distance(self, labels, labels_xentroids, points, points_centroids):
        # distances = (labels != labels_xentroids.reshape([labels_xentroids.shape[0], 1, labels.shape[1]])).sum(axis=2, dtype=float)
        # ((points - points_centroids.reshape([points_centroids.shape[0], 1, points.shape[1]]))**2).sum(axis=2, dtype=float)
        pass

    def calc_distance(self, X, idx=None, Y=None, metrics=None, preprocessing=None):
        """行列Xと、X[idx]の各点との距離を求める
        Args:
            X(ndarray): (n_samples, n_features)の行列
            idx(list): Xのどのサンプルとの距離を計算するかを指定。X内のデータとの距離を求めるときに利用。この場合は、自身との距離はnp.infになる。idxかYのいずれか。
            Y(ndarray): (n_samples, n_features)の行列。Yのサンプルサイズごとに、Xとの距離を求める。idxかYのいずれか。
            metrics(str): 距離尺度を指定。Noneの場合はinitで渡された距離尺度が用いられる。
            preprocessing(str): 前処理を指定。(normalize, None)。各クラスタ中心との距離ごとに行う。(全データで行うのとでどんな違いある？)
        Return:
            ndarray: (idxもしくはYのサンプルサイズ, Xのサンプルサイズ)の行列。各行は、Xの全データと、X[ixd]もしくはYの各サンプルとの距離をもつ。
        """
        if metrics is None: # metricsが指定されていない場合は、initでのmetricsを指定
            metrics = self.metrics
        
        if idx is None and Y is None:
            print("距離の比較対象がありません。idxかYを与えてください。")
            sys.exit(1)
        elif idx is not None and Y is not None:
            print("idxとYの両方が渡されています。どちらか一方のみにしてください。")
            sys.exit(1)
        elif idx is not None:
            sample_size = len(idx)
            if metrics == "l0":
                distances = (X != X[idx].reshape([len(idx), 1, X.shape[1]])).sum(axis=2, dtype=float)
            elif metrics == "l2":
                distances = ((X - X[idx].reshape([len(idx), 1, X.shape[1]]))**2).sum(axis=2, dtype=float)
            else:
                print("Could not fid such a metrics. Select another one.")
                sys.exit(1)
            # distances[np.arange(len(idx)), idx] = np.inf
        elif Y is not None:
            sample_size = Y.shape[0]
            if metrics == "l0":
                distances = (X != Y.reshape([Y.shape[0], 1, Y.shape[1]])).sum(axis=2, dtype=float)
            elif metrics == "l2":
                distances = ((X - Y.reshape([Y.shape[0], 1, Y.shape[1]]))**2).sum(axis=2, dtype=float)
            else:
                print("Could not fid such a metrics. Select another one.")
                sys.exit(1)
        
        if preprocessing == "normalization":
            # range_distances = np.ma.masked_where(distances==np.inf, distances).max(axis=1) - np.ma.masked_where(distances==np.inf, distances).min(axis=1) # 各行(クラスタ中心との距離)ごとに正規化する
            # range_distances = np.ma.masked_where(distances==np.inf, distances).max() - np.ma.masked_where(distances==np.inf, distances).min() # 全体で正規化
            # range_distances = range_distances.data.reshape([sample_size, 1]) # 各行で正規化の場合のみ必要
            range_distances = distances.max() - distances.min()
            distances = (distances - distances.min()) / range_distances
            # norms = np.linalg.norm(X, axis=0, keepdims=True)
            # X_normalized = X / norms

        if idx is not None:
            distances[np.arange(len(idx)), idx] = np.inf
        return distances

    def fit(self, k, X, labelVec_weight, pc=None):
        """calc labels
        Args:
            k(int): クラスタ数
            X(ndarray): (n_samples, n_features)
            labelVec_weight(float): l0l2のときのみ.ラベル遷移ベクトルと空間的距離を結合する際のラベル側の割合.
            pc(ndarray): default None. (n_samples, 3)。metricsがl0l2のときのみ。三次元点群座標。
        Return:
            ndarray: labels per samples. (n_samples, )
        """
        if self.metrics == "l0l2" and pc is None:
            print("metricsが'l0l2'の場合は、引数pcに三次元空間座標を与えてください")
            sys.exit(1)
        n_samples = X.shape[0]
        n_features = X.shape[1]
        alpha_labelVec = labelVec_weight # クラスタリングでラベル遷移ベクトルと三次元空間座標の両方を用いるときの重み
        alpha_spatial = 1.0 - alpha_labelVec

        # k-means++で初期値を決定
        centroids_idx = []
        centroids_idx.append(self.random_state.randint(n_samples)) # １つ目のクラスタ中心をランダムに選択
        if self.metrics == "l0" or self.metrics == "l2":
            for i in range(1, k):
                # distances = np.empty((0, n_samples), dtype=float)
                # for j in range(i): # 各クラスタ中心と、サンプル点との距離の行列を作成。(num_cluster, num_samples)の形状。
                #     distance_tmp = self.custom_distance_old(X, X[centroids_idx[j]])
                #     distance_tmp[centroids_idx[j]] = np.inf # クラスタ中心自身との距離が最小になるのを防ぐために、無限にしておく
                #     distances = np.vstack([distances, distance_tmp])
                distances = self.calc_distance(X=X, idx=centroids_idx)
                min_distances = distances.min(axis=0) # 各点の最近傍クラスタ中心までの距離
                mask = np.ones(n_samples, dtype=bool) # クラスタ中心以外のインデックスマスク
                mask[centroids_idx[:i]] = False
                pr = min_distances[mask] / min_distances[mask].sum() # 次に取る初期値の確率分布
                centroids_idx.append(int(self.random_state.choice(np.arange(n_samples)[mask], size=1, replace=False, p=pr)[0]))
        elif self.metrics == "l0l2":
            for i in range(1, k):
                labelVec_distances = self.calc_distance(X=X, idx=centroids_idx, metrics="l0", preprocessing="normalization")
                min_label_distances = labelVec_distances.min(axis=0)
                spatial_distances = self.calc_distance(X=pc, idx=centroids_idx, metrics="l2", preprocessing="normalization")
                min_spatial_distances = spatial_distances.min(axis=0)

                min_distances = alpha_labelVec*min_label_distances + alpha_spatial*min_spatial_distances

                mask = np.ones(n_samples, dtype=bool) # クラスタ中心以外のインデックスマスク
                mask[centroids_idx[:i]] = False
                
                pr = min_distances[mask] / min_distances[mask].sum()
                centroids_idx.append(int(self.random_state.choice(np.arange(n_samples)[mask], size=1, replace=False, p=pr)[0]))
        else:
            print("そのようなmetricsは無効です")
            sys.exit(1)
        
        # 初期クラスタ中心をランダムで選択
        # centroids_idx = self.random_state.choice(n_samples, size=self.n_clusters).tolist()

        # クラスタ中心の初期値を描画 テストなので適当 点群のパスは引数Xに応じて修正して
        # ndarray2video = NDARRAY2VIDEO("./data/pointclouds/bodyHands_REGISTRATIONS_A01/A01_pc_array.pkl", "./", dir_rm=False)
        # ndarray2video.create(frame_idx=[0], points_idx=centroids_idx, file_names=["first_cluster_centroids"], create_video=False)
        
        # クラスタの更新
        past_labels = None
        labels = np.zeros(n_samples) - 1

        if self.metrics == "l0" or self.metrics == "l2":
            centroids = X[centroids_idx].astype(float)
        elif self.metrics == "l0l2":
            centroids_labelVec = X[centroids_idx].astype(float)
            centroids_spatial = pc[centroids_idx].astype(float)
        # centroids = X[centroids_idx]
        
        iter_cnt = 0
        while (past_labels != labels).sum() != 0 and iter_cnt < self.max_iter:
            past_labels = labels

            if self.metrics == "l0" or self.metrics == "l2":
                distances = self.calc_distance(X=X, Y=centroids)
                # ユークリッド距離であれば1番目で問題ないが、l2だと大量に同じ距離が出てくるので二番目。そんなに時間もかからないし、とりあえず2にしておく。(収束しない問題はあるが、、、)
                # 1. 距離が最小のクラスタ中心に各点を割り当てる。
                # labels = distances.argmin(axis=0)
                
                # 2. 同じ距離の点があるときはランダムに割り振るようにする。
                tmp_labels = []
                for c in range(distances.shape[1]):
                    tmp_labels.append(self.random_state.choice(np.where((distances == distances.min(axis=0))[:, c])[0]))
                labels = np.array(tmp_labels)

                # クラスタ中心を再計算
                for c in range(k):
                    cluster_mask = (labels == c)
                    centroid = X[cluster_mask].sum(axis=0) / cluster_mask.sum()
                    centroids[c] = centroid
                if self.metrics == "l0":
                    centroids = np.round(centroids) # 要素が異なる個数をカウントするので、小数だとまずい。全部ことなる要素になってしまう。偶数丸めで一桁に(型はfloat)。
                self.centroids = centroids
            
            elif self.metrics == "l0l2":
                # 各クラスタ中心との距離を求める
                labelVec_distances = self.calc_distance(X=X, Y=centroids_labelVec, metrics="l0", preprocessing="normalization")
                spatial_distances = self.calc_distance(X=pc, Y=centroids_spatial, metrics="l2", preprocessing="normalization")
                distances = (alpha_labelVec * labelVec_distances) + (alpha_spatial * spatial_distances)
                # 1. 距離が最小のクラスタ中心に各点を割り当てる。
                # labels = distances.argmin(axis=0)
                
                # 2. 同じ距離の点があるときはランダムに割り振るようにする。
                tmp_labels = []
                for c in range(distances.shape[1]):
                    # print(np.where((distances == distances.min(axis=0))[:, c])[0])
                    tmp_labels.append(self.random_state.choice(np.where((distances == distances.min(axis=0))[:, c])[0]))
                labels = np.array(tmp_labels)
                
                # pdb.set_trace()
                # クラスタ中心を再計算
                for c in range(k):
                    cluster_mask = (labels == c)
                    # centroid = X[cluster_mask].sum(axis=0) / int(cluster_mask.sum())
                    # centroids[c] = centroid
                    centroid_labelVec = X[cluster_mask].sum(axis=0) / cluster_mask.sum()
                    centroids_labelVec[c] = centroid_labelVec
                    centroid_spatial = pc[cluster_mask].sum(axis=0) / cluster_mask.sum()
                    centroids_spatial[c] = centroid_spatial
                centroids_labelVec = np.round(centroids_labelVec) # ラベル遷移ベクトルの距離がl0なので、クラスタ中心も整数にする。
                self.centroids_labelVec = centroids_labelVec
                self.centroid_spatial = centroid_spatial
                print(centroids_labelVec)
                print(centroids_spatial)
            
            print(iter_cnt, np.unique(labels))
            iter_cnt += 1
        self.distances = distances
        return labels
    
    def calc_aic(self, range_k, X, labelVec_weight, pc=None):
        """ クラスタ数kまでのaicを求める.これは距離尺度をl0l2の場合のみ対応
        Args:
            range_k(list): aicを求めるkの範囲.[start_k, end_k]のリスト
            others: fitメソッドと同様
        """
        X_dim = X.shape[1]
        pc_dim = pc.shape[1]
        for k in range(range_k[0], range_k[1]):
            labels = self.fit(k=k, X=X, labelVec_weight=labelVec_weight, pc=pc)
            for i in range(k):
                cluster_size = (labels == i).sum()
                # ラベル遷移ベクトルの尤度
                pdb.set_trace()
                labelVec_distances = (X[labels==i] != self.centroids_labelVec[i].reshape([1, -1])).sum(axis=1, dtype=float)
                range_labelVec_distances = labelVec_distances.max() - labelVec_distances.min() # 全体で正規化
                labelVec_distances = (labelVec_distances - labelVec_distances.min()) / range_labelVec_distances
                # 三次元座標の尤度


class MyKMedoids():
    def __init__(self, metrics, max_iter=100, random_seed=0):
        self.max_iter = max_iter
        self.random_state = np.random.RandomState(random_seed)
        self.metrics = metrics
        self.distances = None # aicを求めるために用いる.クラスタが決定された際の距離
        self.centroids = None
        self.centroids_labelVec = None
        self.centroids_spatial = None
    
    def calc_distance(self, X, idx=None, Y=None, metrics=None, preprocessing=None):
        """行列Xと、X[idx]の各点との距離を求める
        Args:
            X(ndarray): (n_samples, n_features)の行列
            idx(list): Xのどのサンプルとの距離を計算するかを指定。X内のデータとの距離を求めるときに利用。この場合は、自身との距離はnp.infになる。idxかYのいずれか。
            Y(ndarray): (n_samples, n_features)の行列。Yのサンプルサイズごとに、Xとの距離を求める。idxかYのいずれか。
            metrics(str): 距離尺度を指定。Noneの場合はinitで渡された距離尺度が用いられる。
            preprocessing(str): 前処理を指定。(normalize, None)。各クラスタ中心との距離ごとに行う。(全データで行うのとでどんな違いある？)
        Return:
            ndarray: (idxもしくはYのサンプルサイズ, Xのサンプルサイズ)の行列。各行は、Xの全データと、X[ixd]もしくはYの各サンプルとの距離をもつ。
        """
        if metrics is None: # metricsが指定されていない場合は、initでのmetricsを指定
            metrics = self.metrics
        
        if idx is None and Y is None:
            print("距離の比較対象がありません。idxかYを与えてください。")
            sys.exit(1)
        elif idx is not None and Y is not None:
            print("idxとYの両方が渡されています。どちらか一方のみにしてください。")
            sys.exit(1)
        elif idx is not None:
            sample_size = len(idx)
            if metrics == "l0":
                distances = (X != X[idx].reshape([len(idx), 1, X.shape[1]])).sum(axis=2, dtype=float)
            elif metrics == "l2":
                distances = ((X - X[idx].reshape([len(idx), 1, X.shape[1]]))**2).sum(axis=2, dtype=float)
            else:
                print("Could not fid such a metrics. Select another one.")
                sys.exit(1)
            # distances[np.arange(len(idx)), idx] = np.inf
        elif Y is not None:
            sample_size = Y.shape[0]
            if metrics == "l0":
                distances = (X != Y.reshape([Y.shape[0], 1, Y.shape[1]])).sum(axis=2, dtype=float)
            elif metrics == "l2":
                distances = ((X - Y.reshape([Y.shape[0], 1, Y.shape[1]]))**2).sum(axis=2, dtype=float)
            else:
                print("Could not fid such a metrics. Select another one.")
                sys.exit(1)
        
        if preprocessing == "normalization":
            # range_distances = np.ma.masked_where(distances==np.inf, distances).max(axis=1) - np.ma.masked_where(distances==np.inf, distances).min(axis=1) # 各行(クラスタ中心との距離)ごとに正規化する
            # range_distances = np.ma.masked_where(distances==np.inf, distances).max() - np.ma.masked_where(distances==np.inf, distances).min() # 全体で正規化
            # range_distances = range_distances.data.reshape([sample_size, 1]) # 各行で正規化の場合のみ必要
            range_distances = distances.max() - distances.min()
            distances = (distances - distances.min()) / range_distances
            # norms = np.linalg.norm(X, axis=0, keepdims=True)
            # X_normalized = X / norms

        if idx is not None:
            distances[np.arange(len(idx)), idx] = np.inf
        return distances

    def fit(self, k, X, labelVec_weight=None, pc=None):
        """calc labels
        Args:
            k(int): クラスタ数
            X(ndarray): (n_samples, n_features)
            labelVec_weight(float): default None. l0l2のときのみ.ラベル遷移ベクトルと空間的距離を結合する際のラベル側の割合.
            pc(ndarray): default None. (n_samples, 3)。metricsがl0l2のときのみ。三次元点群座標。
        Return:
            ndarray: labels per samples. (n_samples, )
        """
        if self.metrics == "l0l2" and pc is None:
            print("metricsが'l0l2'の場合は、引数pcに三次元空間座標を与えてください")
            sys.exit(1)
        n_samples = X.shape[0]
        n_features = X.shape[1]

        if labelVec_weight is not None:
            alpha_labelVec = labelVec_weight # クラスタリングでラベル遷移ベクトルと三次元空間座標の両方を用いるときの重み
            alpha_spatial = 1.0 - alpha_labelVec

        # k-means++で初期値を決定
        centroids_idx = []
        centroids_idx.append(self.random_state.randint(n_samples)) # １つ目のクラスタ中心をランダムに選択
        if self.metrics == "l0" or self.metrics == "l2":
            for i in range(1, k):
                # distances = np.empty((0, n_samples), dtype=float)
                # for j in range(i): # 各クラスタ中心と、サンプル点との距離の行列を作成。(num_cluster, num_samples)の形状。
                #     distance_tmp = self.custom_distance_old(X, X[centroids_idx[j]])
                #     distance_tmp[centroids_idx[j]] = np.inf # クラスタ中心自身との距離が最小になるのを防ぐために、無限にしておく
                #     distances = np.vstack([distances, distance_tmp])
                distances = self.calc_distance(X=X, idx=centroids_idx)
                min_distances = distances.min(axis=0) # 各点の最近傍クラスタ中心までの距離
                mask = np.ones(n_samples, dtype=bool) # クラスタ中心以外のインデックスマスク
                mask[centroids_idx[:i]] = False
                pr = min_distances[mask] / min_distances[mask].sum() # 次に取る初期値の確率分布
                centroids_idx.append(int(self.random_state.choice(np.arange(n_samples)[mask], size=1, replace=False, p=pr)[0]))
        elif self.metrics == "l0l2":
            for i in range(1, k):
                labelVec_distances = self.calc_distance(X=X, idx=centroids_idx, metrics="l0", preprocessing="normalization")
                min_label_distances = labelVec_distances.min(axis=0)
                spatial_distances = self.calc_distance(X=pc, idx=centroids_idx, metrics="l2", preprocessing="normalization")
                min_spatial_distances = spatial_distances.min(axis=0)

                min_distances = alpha_labelVec*min_label_distances + alpha_spatial*min_spatial_distances

                mask = np.ones(n_samples, dtype=bool) # クラスタ中心以外のインデックスマスク
                mask[centroids_idx[:i]] = False
                
                pr = min_distances[mask] / min_distances[mask].sum()
                centroids_idx.append(int(self.random_state.choice(np.arange(n_samples)[mask], size=1, replace=False, p=pr)[0]))
        else:
            print("そのようなmetricsは無効です")
            sys.exit(1)
        
        # 初期クラスタ中心をランダムで選択
        # centroids_idx = self.random_state.choice(n_samples, size=self.n_clusters).tolist()

        # クラスタ中心の初期値を描画 テストなので適当 点群のパスは引数Xに応じて修正して
        # ndarray2video = NDARRAY2VIDEO("./data/pointclouds/bodyHands_REGISTRATIONS_A01/A01_pc_array.pkl", "./", dir_rm=False)
        # ndarray2video.create(frame_idx=[0], points_idx=centroids_idx, file_names=["first_cluster_centroids"], create_video=False)
        
        # クラスタの更新
        past_labels = None
        labels = np.zeros(n_samples) - 1

        if self.metrics == "l0" or self.metrics == "l2":
            centroids = X[centroids_idx].astype(float)
        elif self.metrics == "l0l2":
            centroids_labelVec = X[centroids_idx].astype(float)
            centroids_spatial = pc[centroids_idx].astype(float)
        # centroids = X[centroids_idx]
        
        iter_cnt = 0
        while (past_labels != labels).sum() != 0 and iter_cnt < self.max_iter:
            past_labels = labels

            if self.metrics == "l0" or self.metrics == "l2":
                distances = self.calc_distance(X=X, Y=centroids)
                # ユークリッド距離であれば1番目で問題ないが、l2だと大量に同じ距離が出てくるので二番目。そんなに時間もかからないし、とりあえず2にしておく。(収束しない問題はあるが、、、)
                # 1. 距離が最小のクラスタ中心に各点を割り当てる。
                # labels = distances.argmin(axis=0)
                
                # 2. 同じ距離の点があるときはランダムに割り振るようにする。
                tmp_labels = []
                for c in range(distances.shape[1]):
                    tmp_labels.append(self.random_state.choice(np.where((distances == distances.min(axis=0))[:, c])[0]))
                labels = np.array(tmp_labels)

                # クラスタ中心を再計算
                for c in range(k):
                    cluster_mask = (labels == c)
                    centroid = np.median(X[cluster_mask], axis=0).astype(int)
                    centroids[c] = centroid
                if self.metrics == "l0":
                    centroids = np.round(centroids) # 要素が異なる個数をカウントするので、小数だとまずい。全部ことなる要素になってしまう。偶数丸めで一桁に(型はfloat)。
                self.centroids = centroids
            
            elif self.metrics == "l0l2":
                # 各クラスタ中心との距離を求める
                labelVec_distances = self.calc_distance(X=X, Y=centroids_labelVec, metrics="l0", preprocessing="normalization")
                spatial_distances = self.calc_distance(X=pc, Y=centroids_spatial, metrics="l2", preprocessing="normalization")
                distances = (alpha_labelVec * labelVec_distances) + (alpha_spatial * spatial_distances)
                # 1. 距離が最小のクラスタ中心に各点を割り当てる。
                # labels = distances.argmin(axis=0)
                
                # 2. 同じ距離の点があるときはランダムに割り振るようにする。
                tmp_labels = []
                for c in range(distances.shape[1]):
                    # print(np.where((distances == distances.min(axis=0))[:, c])[0])
                    tmp_labels.append(self.random_state.choice(np.where((distances == distances.min(axis=0))[:, c])[0]))
                labels = np.array(tmp_labels)
                
                # pdb.set_trace()
                # クラスタ中心を再計算
                for c in range(k):
                    cluster_mask = (labels == c)
                    # centroid = X[cluster_mask].sum(axis=0) / int(cluster_mask.sum())
                    # centroids[c] = centroid
                    centroid_labelVec = X[cluster_mask].sum(axis=0) / cluster_mask.sum()
                    centroids_labelVec[c] = centroid_labelVec
                    centroid_spatial = pc[cluster_mask].sum(axis=0) / cluster_mask.sum()
                    centroids_spatial[c] = centroid_spatial
                centroids_labelVec = np.round(centroids_labelVec) # ラベル遷移ベクトルの距離がl0なので、クラスタ中心も整数にする。
                self.centroids_labelVec = centroids_labelVec
                self.centroid_spatial = centroid_spatial
                print(centroids_labelVec)
                print(centroids_spatial)
            
            print(iter_cnt, np.unique(labels))
            iter_cnt += 1
        self.distances = distances
        return labels
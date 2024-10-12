# Standard Library
import random
from copy import deepcopy

# Third Party
import numpy as np

# Local
from ..utilities import laplacian_matrix


##############
# MATH HELPERS
##############


def eigenvector_matrix(L, n):
    eigvals, eigvecs = np.linalg.eig(L)
    sorted_eigs = sorted(zip(eigvals, eigvecs.T), key=lambda e: e[0])

    n_eigvecs = []
    for index, (eigval, eigvec) in enumerate(sorted_eigs):
        if not index:
            continue
        elif index == n:
            break

        n_eigvecs.append(eigvec)

    return np.vstack(n_eigvecs).T


#################
# K-MEANS HELPERS
#################


def init_communities(num_nodes, k, cluster_size):
    # Инициализация кластеров с фиксированным количеством элементов
    # all_indices = list(range(num_nodes))
    # random.shuffle(all_indices)  # Перемешиваем индексы
    # communities = [{} for _ in range(k)]  # Создаем пустые кластеры
    return [{i} for i in random.sample(range(num_nodes), k)]


def calc_centroids(V, communities):
    centroids = []
    for community in communities:
        centroid = V[list(community)].mean(axis=0)
        centroids.append(centroid)

    C = np.vstack(centroids)
    return C


def update_assignments(V, C, communities, cluster_size):
    num_nodes = len(V)
    # Используем список для отслеживания количества узлов в каждом кластере
    cluster_sizes = [len(comm) for comm in communities]
    
    # Множество для отслеживания уже назначенных узлов
    assigned_nodes = set().union(*communities)

    for i in range(num_nodes):
        # Пропускаем узел, если он уже был назначен
        if i in assigned_nodes:
            continue

        best_sim, best_comm_index = -1, None

        # Для каждого узла находим ближайший центроид, учитывая, что кластер не переполнен
        for c_i in range(len(C)):
            if cluster_sizes[c_i] >= cluster_size:
                # Пропускаем кластер, если он уже достиг максимального размера
                continue

            cosine_sim = np.dot(V[i], C[c_i])
            cosine_sim /= np.linalg.norm(V[i]) * np.linalg.norm(C[c_i])

            if cosine_sim > best_sim:
                best_sim = cosine_sim
                best_comm_index = c_i

        # Назначаем узел в лучший кластер
        if best_comm_index is not None:
            communities[best_comm_index].add(i)
            cluster_sizes[best_comm_index] += 1
            assigned_nodes.add(i)  # Добавляем узел в множество назначенных

    return communities



###### 
# MAIN
######


def spectral_clustering(adj_matrix: np.ndarray, k: int, cluster_size: int) -> list:
    L = laplacian_matrix(adj_matrix)
    V = eigenvector_matrix(L, k)

    # Инициализация кластеров
    communities = init_communities(len(adj_matrix), k, cluster_size)
    
    while True:
        C = calc_centroids(V, communities)
        updated_communities = update_assignments(V, C, deepcopy(communities), cluster_size)

        if updated_communities == communities:
            break

        communities = deepcopy(updated_communities)

    return communities

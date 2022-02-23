# This file is released under MIT licence, see file LICENSE.
# Author(s):       Theo Lacombe
#
# Copyright (C) 2021 Inria

import numpy as np
import utils.utils_graphs as ug
import gudhi as gd
import warnings
try:
    import matplotlib.pyplot as plt
except:
    ImportError('matplotlib not installed. Plot functions not available.')


def diagram_from_simplex_tree(st, mode, dim=0):
    '''
    Given a gudhi.SimplexTree, compute the H0 persistence diagram and flip it. The intuition is that if we build the
    SimplexTree using negative values, we actually return the superlevel sets PD.
    We remove the infinite point(s) and only return the second coordinates, so that the returned PD is 1D.
    :param st: gudhi.SimplexTree
    :return: a one-dimensional diagram represented by a numpy.array.
    '''
    st.compute_persistence(min_persistence=-1.)
    dgm0 = st.persistence_intervals_in_dimension(0)[:, 1]
    if mode == "superlevel":
        dgm0 = - dgm0[np.where(np.isfinite(dgm0))]
    elif mode == "sublevel":
        dgm0 = dgm0[np.where(np.isfinite(dgm0))]
    if dim==0:
        return dgm0
    elif dim==1:
        dgm1 = st.persistence_intervals_in_dimension(1)[:,0]
        return dgm0, dgm1


def dgm_per_layers_from_graphs(graphs_per_layers):
    '''
    Given a list of SimplexTree (representing graphs, typically from build_graph_from_deep_model),
    compute the corresponding persistence diagrams using superlevel set filtration (equiv. maximum spanning tree).

    Note: we add negative wheights and then flip the diagram to implicitely perform superlevel sets persistence.
    These persistence diagrams are 1-dimensional and have the same cardinality.

    :param graphs_per_layers: list of gudhi simplexTree.
    :return: list of persistence diagrams, represented as 1D numpy.array.
    '''

    # [G.compute_persistence(min_persistence=-1.) for G in graphs_per_layers]
    # dgms = [G.persistence_intervals_in_dimension(0)[:,1] for G in graphs_per_layers]
    # dgms = [- dgm[np.where(np.isfinite(dgm))] for dgm in dgms]
    dgms = [diagram_from_simplex_tree(G, mode="superlevel") for G in graphs_per_layers]
    return dgms


def diags_from_graphs(graphs):
    '''
    :param graphs: list of list of SimplexTree (representing graphs)
    :return: list of list of persistence diagrams
    '''
    return [dgm_per_layers_from_graphs(gpl) for gpl in graphs]


def diag_from_numpy_array(A, dim=0):
    '''
    Turn a graph, encoded by its numpy adjacency matrix, into the corresponding persistence diagram.
    We use the Rips filtration of negative weight values (then flip the diagram), so that it is equivalent to
    superlevel set filtration.
    :param A: A numpy adjacency matrix of size (n x n) representing a graph.
    :return: A numpy array representing the corresponding persistence diagram (size n-1).
    '''
    n = A.shape[0]
    # assert np.min(A) >= 0
    if np.min(A) < 0:
        warnings.warn('Warning: The distance matrix contains negative values!')
    rc = gd.RipsComplex(distance_matrix=-A)  # take negative values of A for superlevel set filtration
    if dim == 0:
        st = rc.create_simplex_tree(max_dimension=1)
        # must do this somewhat dirty trick to ensure the filtration to work properly (because of the negative entries
        # in the matrix).
        for x in st.get_simplices():
            if len(x[0]) == 1:
                st.assign_filtration(x[0], -np.inf)
    elif dim == 1:
        st = rc.create_simplex_tree(max_dimension=2)
        for x in st.get_simplices():
            if len(x[0]) == 1:
                st.assign_filtration(x[0], -np.inf)
    else:
        raise ValueError('dim = %s not allowed. dim must be 0 or 1.' %dim)
    dgm = diagram_from_simplex_tree(st, mode="superlevel", dim=dim)
    # dgm = diagram_from_simplex_tree(st, mode="sublevel", dim=dim)
    return dgm


def diag_from_point_cloud(X, mode="sublevel"):
    '''
    Given a point cloud, compute its H0-Rips (**minimum** spanning tree). Returns 1D diagram, without infinite points.
    :param X: numpy.array of size (n x d).
    :return: one dimensional PD of size (n-1).
    '''
    rc = gd.RipsComplex(points=X)
    st = rc.create_simplex_tree(max_dimension=1)
    dgm = diagram_from_simplex_tree(st, mode=mode)
    return dgm


def wasserstein_barycenter_1D(u, p=2):
    '''
    Assume N array with the same number of points K, returns the naive 1D barycenter with support of size K
    '''
    s = np.sort(u)
    if p==2:
        return np.mean(s, axis=0)
    elif p==1:
        return np.median(s, axis=0)
    else:
        raise ValueError()


def wasserstein_distance_1D(a, b, p=2., average=True, thresh=None):
    '''
    Given two sets of points with the same cardinality, compute the 1D Wasserstein distance between them.

    :param a: First 1D set of points.
    :param b: Second 1D set of points.
    :param p: Wasserstein exponent, in [1, +np.inf].
    :param average: if True, the Wasserstein distance is divided by the cardinality of the point clouds (normalization).
    :return: 1D Wasserstein distance between the two sets of points.
    '''
    n = a.shape[0]
    assert b.shape[0] == n

    ## Note: overflows can occur in the following. This should be fixed!

    if thresh is not None:
        a = a[:thresh]
        b = b[:thresh]

    if np.isinf(p):
        res = np.max(np.abs(np.sort(a) - np.sort(b)))
    else:
        res = np.sum(np.abs(np.sort(a) - np.sort(b))**p)**(1./p)

    # print('a:\n', a)
    # print('b:\n', b)
    # print('res / n :', res / n)

    if average:
        return res / n
    else:
        return res

def total_persistence(diag, p=2., average=True, thresh=None):
    '''
    Given a set of points, compute the 1D Wasserstein distance to the set of the same cardinality, consisting only of 0 entries.

    :param diag: 1D set of points.
    :param p: Wasserstein exponent, in [1, +np.inf].
    :param average: if True, the Wasserstein distance is divided by the cardinality of the point clouds (normalization).
    :return: 1D Wasserstein distance between diag and the 0-diagram.
    '''
    diag_0 = np.zeros(np.shape(diag))
    return wasserstein_distance_1D(diag, diag_0, p=p, average=average, thresh=thresh)


def barycenters_of_set_from_deep_model(model, x, layers_id=None):
    '''
    Given a model, a (sub)set of (training) observations and a subset of layers to consider, compute the corresponding
    barycenters (one for each layer). Note that, for a given layer, the cardinality of the barycenter is the same
    as the one of the diagrams corresponding to this layer.

    :param model: a tensorflow sequential model.
    :param x: Set of observations (usually belonging to a same class in the training set).
    :param layers_id: layers for which we compute a barycenter. If `None`, all (fully-connected) layers are used.
    :return: list of list of barycenters (1D numpy.array).
    '''
    graphs = ug.build_graphs_from_deep_model(model, x, layers_id=layers_id)
    diags = diags_from_graphs(graphs)
    nlayer = len(diags[0])  # number of layer (with weight matrix) that we use from our model.
    point_clouds_per_layer = [np.array([diag[ell] for diag in diags]) for ell in range(nlayer)]

    wbarys = [wasserstein_barycenter_1D(pc) for pc in point_clouds_per_layer]

    return wbarys

def mean_adjacency_matrices_from_deep_model(model, x, layers_id=None):
    '''
    Given a model, a (sub)set of (training) observations and a subset of layers to consider, compute the corresponding
    averaged adjacency matrices (one for each layer).

    :param model: a tensorflow sequential model.
    :param x: Set of observations (usually belonging to a same class in the training set).
    :param layers_id: layers for which we compute an averaged adjacency matrix. If `None`, all (fully-connected) layers are used.
    :return: list of list of averaged adjacency matrices (1D numpy.array).
    '''
    adjacency_matrices = ug.build_adjacency_matrices_from_deep_model(model, x, layers_id=layers_id)

    return np.mean(adjacency_matrices, axis=0)


def topological_uncertainty(model, x, all_barycenters,
                            layers_id=None, aggregation='mean', p=2., normalize_wasserstein=True, all_classes=False):
    '''

    :param model: tensorflow sequential model.
    :param x: set of observations for which we want to compute Topological Uncertainty. Must be valid entries for `model`.
    :param all_barycenters: list of list of persistence diagrams; all_barycenter[label_id][ell] represents the
                            barycenter corresponding to the `label_id`-th class and the `layer_id[ell]`-th layer.
                            Typically obtained calling `barycenters_of_set_from_deep_model`.
    :param layers_id: layers for which we compute Topological Uncertainty. Must be the same as the one used to compute
    barycenters first.
    :param aggregation: How TU-per-layer is aggregated. Default: `mean`, i.e. averaging through layers.
                        If `max`,  maximum TU over layers is taken into account;
                        If `None`, return an array representing the TU for all layers considered.
    :param p: Wasserstein exponent used to compute distances. Default is p=2., consistent with Fréchet mean computation.
    :param normalize_wasserstein: If True, divide the distance between diagrams by their cardinality (helps making things
                                  comparable layer-wise).
    :return: Topological uncertainty values for all observations. If `aggregation` is `None`, it is a
             (nb_obs x nb_layers) numpy.array.
             Otherwise, it is a (nb_obs) numpy.array.
    '''
    nlayer = len(all_barycenters[0])  # number of layers used
    predicted_classes = np.argmax(model.predict(x), axis=-1)
    num_classes = model.predict(x).shape[-1]
    classes = range(num_classes)
    graphs = ug.build_graphs_from_deep_model(model, x, layers_id=layers_id)
    diags = diags_from_graphs(graphs)

    if not all_classes:
        res = np.array([[wasserstein_distance_1D(diags_per_layer[ell], all_barycenters[predicted_class][ell],
                                             p=p, average=normalize_wasserstein) for ell in range(nlayer)]
                     for (diags_per_layer, predicted_class) in zip(diags, predicted_classes)])

        if aggregation=='mean':
            return np.mean(res,axis=1)
        elif aggregation=='max':
            return np.max(res,axis=1)
        elif aggregation is None:
            return res
        else:
            raise ValueError('aggregation=%s is not valid. Set it to mean (default) or max.' %aggregation)

    else:
        res = np.array([[[wasserstein_distance_1D(diags_per_layer[ell], all_barycenters[_class][ell],
                                                 p=p, average=normalize_wasserstein)
                                                 for ell in range(nlayer)] for (diags_per_layer, _class) in zip(diags, np.full(predicted_classes.shape, cl))] for cl in classes])

        if aggregation == 'mean':
            return np.transpose(np.array([np.mean(r, axis=1) for r in res]))
        elif aggregation == 'max':
            return np.transpose(np.array([np.max(r, axis=1) for r in res]))
        elif aggregation is None:
            return res
        else:
            raise ValueError('aggregation=%s is not valid. Set it to mean (default) or max.' % aggregation)


def topological_difference(model, x, all_mean_adjacency_matrices,
                            layers_id=None, aggregation="mean", p=2.,
                           normalize_wasserstein=True, absolute_value=True, all_classes=False):
    '''

    :param model: tensorflow sequential model.
    :param x: set of observations for which we want to compute Topological Uncertainty. Must be valid entries for `model`.
    :param all_mean_adjacency_matrices: list of list of adjacency matrices; all_mean_adjacency_matrices[label_id][ell] represents the
                            mean adjacency matrix corresponding to the `label_id`-th class and the `layer_id[ell]`-th layer.
                            Typically obtained by calling `mean_adjacency_matrices_from_deep_model`.
    :param layers_id: layers for which we compute Topological Difference. Must be the same as the one used to compute
    mean adjacency matrices first.
    :param aggregation: How TU-per-layer is aggregated. Default: `mean`, i.e. averaging through layers.
                        If `max`,  maximum TU over layers is taken into account;
                        If `None`, return an array representing the TU for all layers considered.
    :param p: Wasserstein exponent used to compute distances. Default is p=2., consistent with Fréchet mean computation.
    :param normalize_wasserstein: If True, divide the distance between diagrams by their cardinality (helps making things
                                  comparable layer-wise).
    :param absolute_value: Boolean flag, indicating whether taking the absolute value of the difference of adjacency matrices or not.
    :return: Topological Difference values for all observations. If `aggregation` is `None`, it is a
             (nb_obs x nb_layers) numpy.array.
             Otherwise, it is a (nb_obs) numpy.array.
    '''
    def a_val(absolute_value):
        if absolute_value:
            return lambda x: np.abs(x)
        else:
            return lambda x: x

    abs_val_fct = a_val(absolute_value)

    nlayer = len(all_mean_adjacency_matrices[0])  # number of layers used
    predicted_classes = np.argmax(model.predict(x), axis=-1)
    num_predicted_classes = np.shape(predicted_classes)[0]
    num_classes = model.predict(x).shape[-1]
    classes = range(num_classes)
    adjacency_matrices = ug.build_adjacency_matrices_from_deep_model(model, x, layers_id=layers_id)

    if not all_classes:
        matrix_diff = [[abs_val_fct(A[ell] - all_mean_adjacency_matrices[predicted_class][ell]) for ell in range(nlayer)]
                        for (A, predicted_class) in zip(adjacency_matrices, predicted_classes)]

        diags = [[diag_from_numpy_array(B) for B in matrix_diff[idx]] for idx in range(len(matrix_diff))]

        res = np.array([[total_persistence(d, p=p) for d in diags[idx]] for idx in range(len(diags))])

        if aggregation=='mean':
            return np.mean(res,axis=1)
        elif aggregation=='max':
            return np.max(res,axis=1)
        elif aggregation is None:
            return res
        else:
            raise ValueError('aggregation=%s is not valid. Set it to mean (default) or max.' %aggregation)

    else:
        matrix_diff = [
            [
             [
             abs_val_fct(A[ell] - all_mean_adjacency_matrices[_class][ell]) for ell in range(nlayer)
             ]
            for (A, _class) in zip(adjacency_matrices, np.full(predicted_classes.shape, cl))
            ]
            for cl in classes
            ]

        # diags = [[[diag_from_numpy_array(B) for B in matrix_diff[i][idx]] for idx in range(len(matrix_diff[i]))] for i in range(len(matrix_diff))]
        #
        # res = np.array([np.array([[total_persistence(d, p=p) for d in diags[i][idx]] for idx in range(len(diags[i]))]) for i in range(len(diags))])
        # print('matrix diff shape: ', np.array(matrix_diff).shape)

        diags = [[[diag_from_numpy_array(B) for B in matrix_diff[cl][pcl]] for pcl in range(num_predicted_classes)] for cl in classes]# for ell in range(nlayer)])

        # print('diags\n', np.array(diags).shape)

        res = np.array([np.array([[total_persistence(d, p=p) for d in diags[cl][pcl]] for pcl in range(num_predicted_classes)]) for cl in classes])# for cl in classes])

        if aggregation == 'mean':
            return np.transpose(np.array([np.mean(r, axis=-1) for r in res])), diags, adjacency_matrices, matrix_diff# return np.transpose(np.array([np.mean(r, axis=1) for r in res])), diags, adjacency_matrices, matrix_diff
        elif aggregation == 'max':
            return np.transpose(np.array([np.max(r, axis=1) for r in res]))
        elif aggregation is None:
            return res
        else:
            raise ValueError('aggregation=%s is not valid. Set it to mean (default) or max.' % aggregation)

def plot_1d_diagram(dgm, ax=None, color='blue', xlim=None):
    n = dgm.shape
    m, M = np.min(dgm), np.max(dgm)
    v = 0.1 * (M - m)
    if ax is None:
        fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.plot((dgm,dgm), (np.zeros(n), 0.3 * np.ones(n)), color=color)
    ax.plot(xlim, (0,0), color='black')
    ax.set_xticks([m,M])
    ax.set_xticklabels([m, M])
    ax.scatter(dgm, 0.3 * np.ones(n), marker='x', color=color)
    #ax.annotate(np.round(m,1), (m,0.35))
    #ax.annotate(np.round(M,1), (M,0.35))
    ax.set_ylim(0,1)
    #ax.set_yticks([])
    if xlim is not None:
        ax.set_xlim(xlim)
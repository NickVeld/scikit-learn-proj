"""Spectral Embedding"""

# Author: Gael Varoquaux <gael.varoquaux@normalesup.org>
#         Wei LI <kuantkid@gmail.com>
# License: BSD 3 clause

import warnings
import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import lobpcg
from ..base import BaseEstimator
from ..externals import six
from ..utils import check_random_state, check_array, check_symmetric
from ..utils.extmath import _deterministic_vector_sign_flip
from ..utils.graph import graph_laplacian
from ..utils.sparsetools import connected_components
from ..utils.arpack import eigsh
from ..metrics.pairwise import rbf_kernel
from ..neighbors import kneighbors_graph

from math import sqrt

def _graph_connected_component(graph, node_id):
    """Find the largest graph connected components that contains one
    given node

    Parameters
    ----------
    graph : array-like, shape: (n_samples, n_samples)
        adjacency matrix of the graph, non-zero weight means an edge
        between the nodes

    node_id : int
        The index of the query node of the graph

    Returns
    -------
    connected_components_matrix : array-like, shape: (n_samples,)
        An array of bool value indicating the indexes of the nodes
        belonging to the largest connected components of the given query
        node
    """
    n_node = graph.shape[0]
    if sparse.issparse(graph):
        # speed up row-wise access to boolean connection mask
        graph = graph.tocsr()
    connected_nodes = np.zeros(n_node, dtype=np.bool)
    nodes_to_explore = np.zeros(n_node, dtype=np.bool)
    nodes_to_explore[node_id] = True
    for _ in range(n_node):
        last_num_component = connected_nodes.sum()
        np.logical_or(connected_nodes, nodes_to_explore, out=connected_nodes)
        if last_num_component >= connected_nodes.sum():
            break
        indices = np.where(nodes_to_explore)[0]
        nodes_to_explore.fill(False)
        for i in indices:
            if sparse.issparse(graph):
                neighbors = graph[i].toarray().ravel()
            else:
                neighbors = graph[i]
            np.logical_or(nodes_to_explore, neighbors, out=nodes_to_explore)
    return connected_nodes


def _graph_is_connected(graph):
    """ Return whether the graph is connected (True) or Not (False)

    Parameters
    ----------
    graph : array-like or sparse matrix, shape: (n_samples, n_samples)
        adjacency matrix of the graph, non-zero weight means an edge
        between the nodes

    Returns
    -------
    is_connected : bool
        True means the graph is fully connected and False means not
    """
    if sparse.isspmatrix(graph):
        # sparse graph, find all the connected components
        n_connected_components, _ = connected_components(graph)
        return n_connected_components == 1
    else:
        # dense graph, find all connected components start from node 0
        return _graph_connected_component(graph, 0).sum() == graph.shape[0]


def _set_diag(laplacian, value, norm_laplacian):
    """Set the diagonal of the laplacian matrix and convert it to a
    sparse format well suited for eigenvalue decomposition

    Parameters
    ----------
    laplacian : array or sparse matrix
        The graph laplacian
    value : float
        The value of the diagonal
    norm_laplacian : bool
        Whether the value of the diagonal should be changed or not

    Returns
    -------
    laplacian : array or sparse matrix
        An array of matrix in a form that is well suited to fast
        eigenvalue decomposition, depending on the band width of the
        matrix.
    """
    n_nodes = laplacian.shape[0]
    # We need all entries in the diagonal to values
    if not sparse.isspmatrix(laplacian):
        if norm_laplacian:
            laplacian.flat[::n_nodes + 1] = value
    else:
        laplacian = laplacian.tocoo()
        if norm_laplacian:
            diag_idx = (laplacian.row == laplacian.col)
            laplacian.data[diag_idx] = value
        # If the matrix has a small number of diagonals (as in the
        # case of structured matrices coming from images), the
        # dia format might be best suited for matvec products:
        n_diags = np.unique(laplacian.row - laplacian.col).size
        if n_diags <= 7:
            # 3 or less outer diagonals on each side
            laplacian = laplacian.todia()
        else:
            # csr has the fastest matvec and is thus best suited to
            # arpack
            laplacian = laplacian.tocsr()
    return laplacian


def spectral_embedding(adjacency, n_components=8, eigen_solver=None,
                       random_state=None, eigen_tol=0.0,
                       norm_laplacian=True, drop_first=True, return_eigenvalues = False):
    """Project the sample on the first eigenvectors of the graph Laplacian.

    The adjacency matrix is used to compute a normalized graph Laplacian
    whose spectrum (especially the eigenvectors associated to the
    smallest eigenvalues) has an interpretation in terms of minimal
    number of cuts necessary to split the graph into comparably sized
    components.

    This embedding can also 'work' even if the ``adjacency`` variable is
    not strictly the adjacency matrix of a graph but more generally
    an affinity or similarity matrix between samples (for instance the
    heat kernel of a euclidean distance matrix or a k-NN matrix).

    However care must taken to always make the affinity matrix symmetric
    so that the eigenvector decomposition works as expected.

    Read more in the :ref:`User Guide <spectral_embedding>`.

    Parameters
    ----------
    adjacency : array-like or sparse matrix, shape: (n_samples, n_samples)
        The adjacency matrix of the graph to embed.

    n_components : integer, optional, default 8
        The dimension of the projection subspace.

    eigen_solver : {None, 'arpack', 'lobpcg', or 'amg'}, default None
        The eigenvalue decomposition strategy to use. AMG requires pyamg
        to be installed. It can be faster on very large, sparse problems,
        but may also lead to instabilities.

    random_state : int seed, RandomState instance, or None (default)
        A pseudo random number generator used for the initialization of the
        lobpcg eigenvectors decomposition when eigen_solver == 'amg'.
        By default, arpack is used.

    eigen_tol : float, optional, default=0.0
        Stopping criterion for eigendecomposition of the Laplacian matrix
        when using arpack eigen_solver.

    drop_first : bool, optional, default=True
        Whether to drop the first eigenvector. For spectral embedding, this
        should be True as the first eigenvector should be constant vector for
        connected graph, but for spectral clustering, this should be kept as
        False to retain the first eigenvector.

    norm_laplacian : bool, optional, default=True
        If True, then compute normalized Laplacian.

    Returns
    -------
    embedding : array, shape=(n_samples, n_components)
        The reduced samples.

    Notes
    -----
    Spectral embedding is most useful when the graph has one connected
    component. If there graph has many components, the first few eigenvectors
    will simply uncover the connected components of the graph.

    References
    ----------
    * https://en.wikipedia.org/wiki/LOBPCG

    * Toward the Optimal Preconditioned Eigensolver: Locally Optimal
      Block Preconditioned Conjugate Gradient Method
      Andrew V. Knyazev
      http://dx.doi.org/10.1137%2FS1064827500366124
    """
    adjacency = check_symmetric(adjacency)

    try:
        from pyamg import smoothed_aggregation_solver
    except ImportError:
        if eigen_solver == "amg":
            raise ValueError("The eigen_solver was set to 'amg', but pyamg is "
                             "not available.")

    if eigen_solver is None:
        eigen_solver = 'arpack'
    elif eigen_solver not in ('arpack', 'lobpcg', 'amg'):
        raise ValueError("Unknown value for eigen_solver: '%s'."
                         "Should be 'amg', 'arpack', or 'lobpcg'"
                         % eigen_solver)

    random_state = check_random_state(random_state)

    n_nodes = adjacency.shape[0]
    # Whether to drop the first eigenvector
    if drop_first:
        n_components = n_components + 1

    if not _graph_is_connected(adjacency):
        warnings.warn("Graph is not fully connected, spectral embedding"
                      " may not work as expected.")

    laplacian, dd = graph_laplacian(adjacency,
                                    normed=norm_laplacian, return_diag=True)
    if (eigen_solver == 'arpack' or eigen_solver != 'lobpcg' and
       (not sparse.isspmatrix(laplacian) or n_nodes < 5 * n_components)):
        # lobpcg used with eigen_solver='amg' has bugs for low number of nodes
        # for details see the source code in scipy:
        # https://github.com/scipy/scipy/blob/v0.11.0/scipy/sparse/linalg/eigen
        # /lobpcg/lobpcg.py#L237
        # or matlab:
        # http://www.mathworks.com/matlabcentral/fileexchange/48-lobpcg-m
        laplacian = _set_diag(laplacian, 1, norm_laplacian)

        # Here we'll use shift-invert mode for fast eigenvalues
        # (see http://docs.scipy.org/doc/scipy/reference/tutorial/arpack.html
        #  for a short explanation of what this means)
        # Because the normalized Laplacian has eigenvalues between 0 and 2,
        # I - L has eigenvalues between -1 and 1.  ARPACK is most efficient
        # when finding eigenvalues of largest magnitude (keyword which='LM')
        # and when these eigenvalues are very large compared to the rest.
        # For very large, very sparse graphs, I - L can have many, many
        # eigenvalues very near 1.0.  This leads to slow convergence.  So
        # instead, we'll use ARPACK's shift-invert mode, asking for the
        # eigenvalues near 1.0.  This effectively spreads-out the spectrum
        # near 1.0 and leads to much faster convergence: potentially an
        # orders-of-magnitude speedup over simply using keyword which='LA'
        # in standard mode.
        try:
            # We are computing the opposite of the laplacian inplace so as
            # to spare a memory allocation of a possibly very large array
            laplacian *= -1
            v0 = random_state.uniform(-1, 1, laplacian.shape[0])
            lambdas, diffusion_map = eigsh(laplacian, k=n_components,
                                           sigma=1.0, which='LM',
                                           tol=eigen_tol, v0=v0)
            embedding = diffusion_map.T[n_components::-1] * dd
        except RuntimeError:
            # When submatrices are exactly singular, an LU decomposition
            # in arpack fails. We fallback to lobpcg
            eigen_solver = "lobpcg"
            # Revert the laplacian to its opposite to have lobpcg work
            laplacian *= -1

    if eigen_solver == 'amg':
        # Use AMG to get a preconditioner and speed up the eigenvalue
        # problem.
        if not sparse.issparse(laplacian):
            warnings.warn("AMG works better for sparse matrices")
        # lobpcg needs double precision floats
        laplacian = check_array(laplacian, dtype=np.float64,
                                accept_sparse=True)
        laplacian = _set_diag(laplacian, 1, norm_laplacian)
        ml = smoothed_aggregation_solver(check_array(laplacian, 'csr'))
        M = ml.aspreconditioner()
        X = random_state.rand(laplacian.shape[0], n_components + 1)
        X[:, 0] = dd.ravel()
        lambdas, diffusion_map = lobpcg(laplacian, X, M=M, tol=1.e-12,
                                        largest=False)
        embedding = diffusion_map.T * dd
        if embedding.shape[0] == 1:
            raise ValueError

    elif eigen_solver == "lobpcg":
        # lobpcg needs double precision floats
        laplacian = check_array(laplacian, dtype=np.float64,
                                accept_sparse=True)
        if n_nodes < 5 * n_components + 1:
            # see note above under arpack why lobpcg has problems with small
            # number of nodes
            # lobpcg will fallback to eigh, so we short circuit it
            if sparse.isspmatrix(laplacian):
                laplacian = laplacian.toarray()
            lambdas, diffusion_map = eigh(laplacian)
            embedding = diffusion_map.T[:n_components] * dd
        else:
            laplacian = _set_diag(laplacian, 1, norm_laplacian)
            # We increase the number of eigenvectors requested, as lobpcg
            # doesn't behave well in low dimension
            X = random_state.rand(laplacian.shape[0], n_components + 1)
            X[:, 0] = dd.ravel()
            lambdas, diffusion_map = lobpcg(laplacian, X, tol=1e-15,
                                            largest=False, maxiter=2000)
            embedding = diffusion_map.T[:n_components] * dd
            if embedding.shape[0] == 1:
                raise ValueError

    embedding = _deterministic_vector_sign_flip(embedding)
    if return_eigenvalues:
        if drop_first:
            return embedding[1:n_components].T, lambdas
        else:
            return embedding[:n_components].T, lambdas
    else:
        if drop_first:
            return embedding[1:n_components].T
        else:
            return embedding[:n_components].T


class SpectralEmbedding(BaseEstimator):
    """Spectral embedding for non-linear dimensionality reduction.

    Forms an affinity matrix given by the specified function and
    applies spectral decomposition to the corresponding graph laplacian.
    The resulting transformation is given by the value of the
    eigenvectors for each data point.

    Read more in the :ref:`User Guide <spectral_embedding>`.

    Parameters
    -----------
    n_components : integer, default: 2
        The dimension of the projected subspace.

    eigen_solver : {None, 'arpack', 'lobpcg', or 'amg'}
        The eigenvalue decomposition strategy to use. AMG requires pyamg
        to be installed. It can be faster on very large, sparse problems,
        but may also lead to instabilities.

    random_state : int seed, RandomState instance, or None, default : None
        A pseudo random number generator used for the initialization of the
        lobpcg eigenvectors decomposition when eigen_solver == 'amg'.

    affinity : string or callable, default : "nearest_neighbors"
        How to construct the affinity matrix.
         - 'nearest_neighbors' : construct affinity matrix by knn graph
         - 'rbf' : construct affinity matrix by rbf kernel
         - 'precomputed' : interpret X as precomputed affinity matrix
         - callable : use passed in function as affinity
           the function takes in data matrix (n_samples, n_features)
           and return affinity matrix (n_samples, n_samples).

    gamma : float, optional, default : 1/n_features
        Kernel coefficient for rbf kernel.

    n_neighbors : int, default : max(n_samples/10 , 1)
        Number of nearest neighbors for nearest_neighbors graph building.

    n_jobs : int, optional (default = 1)
        The number of parallel jobs to run.
        If ``-1``, then the number of jobs is set to the number of CPU cores.

    Attributes
    ----------

    embedding_ : array, shape = (n_samples, n_components)
        Spectral embedding of the training matrix.

    affinity_matrix_ : array, shape = (n_samples, n_samples)
        Affinity_matrix constructed from samples or precomputed.

    References
    ----------

    - A Tutorial on Spectral Clustering, 2007
      Ulrike von Luxburg
      http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.165.9323

    - On Spectral Clustering: Analysis and an algorithm, 2011
      Andrew Y. Ng, Michael I. Jordan, Yair Weiss
      http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.19.8100

    - Normalized cuts and image segmentation, 2000
      Jianbo Shi, Jitendra Malik
      http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.160.2324
    """

    def __init__(self, n_components=2, affinity="nearest_neighbors",
                 gamma=None, random_state=None, eigen_solver=None,
                 n_neighbors=None, n_jobs=1):
        self.n_components = n_components
        self.affinity = affinity
        self.gamma = gamma
        self.random_state = random_state
        self.eigen_solver = eigen_solver
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

    @property
    def _pairwise(self):
        return self.affinity == "precomputed"

    def _get_affinity_matrix(self, X, Y=None):
        """Calculate the affinity matrix from data
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples
            and n_features is the number of features.

            If affinity is "precomputed"
            X : array-like, shape (n_samples, n_samples),
            Interpret X as precomputed adjacency graph computed from
            samples.

        Returns
        -------
        affinity_matrix, shape (n_samples, n_samples)
        """
        if self.affinity == 'precomputed':
            self.affinity_matrix_ = X
            return self.affinity_matrix_
        if self.affinity == 'nearest_neighbors':
            if sparse.issparse(X):
                warnings.warn("Nearest neighbors affinity currently does "
                              "not support sparse input, falling back to "
                              "rbf affinity")
                self.affinity = "rbf"
            else:
                self.n_neighbors_ = (self.n_neighbors
                                     if self.n_neighbors is not None
                                     else max(int(X.shape[0] / 10), 1))
                self.affinity_matrix_ = kneighbors_graph(X, self.n_neighbors_,
                                                         include_self=True,
                                                         n_jobs=self.n_jobs)
                # currently only symmetric affinity_matrix supported
                self.affinity_matrix_ = 0.5 * (self.affinity_matrix_ +
                                               self.affinity_matrix_.T)
                return self.affinity_matrix_
        if self.affinity == 'rbf':
            self.gamma_ = (self.gamma
                           if self.gamma is not None else 1.0 / X.shape[1])
            self.affinity_matrix_ = rbf_kernel(X, gamma=self.gamma_)
            return self.affinity_matrix_
        self.affinity_matrix_ = self.affinity(X)
        return self.affinity_matrix_

    def fit(self, X, y=None):
        """Fit the model from data in X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples
            and n_features is the number of features.

            If affinity is "precomputed"
            X : array-like, shape (n_samples, n_samples),
            Interpret X as precomputed adjacency graph computed from
            samples.

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        X = check_array(X, ensure_min_samples=2, estimator=self)

        random_state = check_random_state(self.random_state)
        if isinstance(self.affinity, six.string_types):
            if self.affinity not in set(("nearest_neighbors", "rbf",
                                         "precomputed")):
                raise ValueError(("%s is not a valid affinity. Expected "
                                  "'precomputed', 'rbf', 'nearest_neighbors' "
                                  "or a callable.") % self.affinity)
        elif not callable(self.affinity):
            raise ValueError(("'affinity' is expected to be an affinity "
                              "name or a callable. Got: %s") % self.affinity)

        affinity_matrix = self._get_affinity_matrix(X)
        #print(affinity_matrix)
        #self.e_over_affinity_matrix = np.average(affinity_matrix, axis=0)

        self.embedding_, self.eigenvalues = spectral_embedding(affinity_matrix,
                                             n_components=self.n_components,
                                             eigen_solver=self.eigen_solver,
                                             random_state=random_state,
                                             return_eigenvalues=True)
        self.empirical_data = X
        return self

    def fit_transform(self, X, y=None):
        """Fit the model from data in X and transform X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples
            and n_features is the number of features.

            If affinity is "precomputed"
            X : array-like, shape (n_samples, n_samples),
            Interpret X as precomputed adjacency graph computed from
            samples.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        self.fit(X)
        return self.embedding_

    def transform(self, X):
        """
        Transform new points into embedding space.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        X_new : array, shape = [n_samples, n_components]

        Notes
        -----

        """
        X = check_array(X)

        # n = X.shape[0]
        # M = np.zeros((n, n))
        # for i in range(n):
        #     for j in range(i + 1, n):
        #         M[j][i] = M[i][j] = metric(X[i], X[j])
        # M_rows_aver = np.average(M, axis=1)
        # M_aver = np.average(M_rows_aver)
        # print(M)
        # print(M_rows_aver)
        # print(M_aver)
        # for i in range(n):
        #     for j in range(i, n):
        #         M[j][i] = M[i][j] = M[i][j]/pow(M_rows_aver[i] * M_rows_aver[j], 0.5)
        # eigenvectors = np.linalg.eig(M)[1]
        # print(M)
        # print(eigenvectors)
        old_data_n_samples = self.empirical_data.shape[0]
        new_data_n_samples = X.shape[0]
        K = kneighbors_graph(np.concatenate((self.empirical_data, X)), self.n_neighbors_,
                                                         include_self=True,
                                                         n_jobs=self.n_jobs)
        # currently only symmetric affinity_matrix supported
        K = ((K + K.T) * 0.5).tolil()
        e_over_K = K[:old_data_n_samples, :].mean(axis=0)
        X_new = np.zeros((new_data_n_samples, self.n_components))
        for j in range(old_data_n_samples + new_data_n_samples):
            for k in range(j):
                K[k,j] /= sqrt(e_over_K[0,k] * e_over_K[0,j])
                K[j,k] = K[k,j]       
        for i in range(new_data_n_samples):
            for k in range(self.n_components):
                for j in range(old_data_n_samples):
                    X_new[i,k] += self.embedding_[j,k] * K[i,j]
                X_new[i,k] /= sqrt(old_data_n_samples) * self.eigenvalues[k]
        return X_new

def metric(vector1, vector2):
    if (len(vector1) != len(vector2)):
        raise ValueError("metric's operands have different sizes!")
    return np.linalg.norm(vector1 - vector2)

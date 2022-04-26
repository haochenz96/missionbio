import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import umap
from hdbscan import HDBSCAN
from missionbio.h5.constants import BARCODE, ID, SAMPLE
from missionbio.h5.data.assay import Assay as H5_Assay
from plotly.subplots import make_subplots
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from missionbio.mosaic.constants import (
    COLORS,
    LABEL,
    ORG_BARCODE,
    PALETTE,
    PCA_LABEL,
    READS,
    SCALED_LABEL,
    UMAP_LABEL,
)
from missionbio.mosaic.plotting import plt, require_seaborn, sns
from missionbio.mosaic.utils import clipped_values, get_indexes


class _Assay(H5_Assay):
    """
    Abstract class for all assays

    Each of :class:`missionbio.mosaic.dna.Dna`, :class:`missionbio.mosaic.cnv.Cnv`,
    and :class:`missionbio.mosaic.protein.Protein` are inherited from this base class,
    hence all of them have the following functionality:

    `_Assay` objects can be filtered using Python's slice notation. It requires two
    keys (barcodes,  ids). Both could be a list of positions, a Boolean list,
    or a list of the respective values.

    Load the sample.

        >>> import missionbio.mosaic.io as mio
        >>> sample = mio.load('/path/to/h5')

    Select the first 250 cells (these aren't necessarily the cells
    with the highest reads, they're arbitrary  cells) and the dna
    variants obtained from filtering.

        >>> select_bars = sample.dna.barcodes()[: 250]
        >>> select_vars = sample.dna.filter_variants()

    Slice the dna assay for the selected cells and variants.

        >>> filtered_dna = sample.dna[select_bars, select_vars]

    Note: This example is for DNA, but the same can be done for
    any `_Assay` object.

    .. rubric:: Modifying data
    .. autosummary::

       add_layer
       add_row_attr
       add_col_attr
       add_metadata
       del_layer
       del_row_attr
       del_col_attr
       del_metadata

    .. rubric:: Selecting data
    .. autosummary::

       barcodes
       clustered_barcodes
       ids
       clustered_ids
       drop
       get_palette
       set_palette
       get_labels
       set_labels
       set_selected_labels
       rename_labels
       get_attribute

    .. rubric:: Tertiary analysis methods
    .. autosummary::

       scale_data
       run_pca
       run_umap
       run_lda
       cluster
       group_clusters

    .. rubric:: Visualizations
    .. autosummary::

       scatterplot
       heatmap
       violinplot
       ridgeplot
       stripplot
       read_distribution
       feature_scatter
       fishplot
       barplot

    .. rubric:: Statistics
    .. autosummary::

       feature_signature
    """

    def __init__(self, *args, **kwargs):
        """
        Calls missionbio.h5.data.assay.__init__.
        """
        super().__init__(*args, **kwargs)
        self.selected_bars = {}
        self.selected_ids = np.array([])
        self.title = ", ".join(self.metadata[SAMPLE].flatten())

        self._rename_barcodes()

        pal = None
        if PALETTE in self.metadata:
            pal = self.get_palette()

        labs = self.get_labels()
        self.set_labels(labs)

        if pal is not None and set(pal.keys()) == set(labs):
            self.set_palette(pal)

    # --------- Interactivity

    def __getitem__(self, key):
        """
        Implemented for convenience when analyzing
        multiple samples in a Jupyter Notebook.

        Parameters
        ----------
        key : 2-tuple
            Both elements of the tuple must be an
            array of values, indices or bools. The first
            element is used to filter barcodes and the
            second element is used to filter ids.

        Returns
        -------
        :class:`missionbio.mosaic.assay._Assay`
            A copy of itself with the relevant
            barcodes and ids selected.

        Raises
        ------
        KeyError
            When the tuple provided does not match the
            expected format.
        """
        def _isstrlike(val):
            if (isinstance(val, list) and isinstance(val[0], str)) or (isinstance(val, np.ndarray) and isinstance(val[0], str)):
                return True
            return False

        if isinstance(key, tuple):
            assay = deepcopy(self)
            key_row, key_col = key

            if _isstrlike(key_row):
                key_row = get_indexes(assay.barcodes(), key_row)

            if _isstrlike(key_col):
                key_col = get_indexes(assay.ids(), key_col)

            if key_row is not None:
                assay.select_rows(key_row)

            if key_col is not None:
                assay.select_columns(key_col)
        else:
            raise KeyError(f"Expected tuple, got {type(key)}")

        return assay

    def _rename_barcodes(self):
        """
        Appends the name of the sample to the barcode
        in case of multi sample files. This is required to
        avoid duplicate barcodes from colliding.

        In case two different samples have duplicate
        barcodes, then there are high chances of
        misidentification of the cell signature. Moreover,
        many plots and analyses assume that barcodes
        are unique. They fail in case of duplicate
        barcodes. This is the easiset workaround to tackle
        both the situations.

        The original barcodes are stored in the org_barcode
        row attribute.
        """

        if self.metadata[SAMPLE].shape[0] > 1:
            msg = 'Multiple samples detected. Appending the sample name to the barcodes to avoid duplicate barcodes.'
            msg += f' The original barcodes are stored in the {ORG_BARCODE} row attribute'

            warnings.warn(msg)

            self.add_row_attr(ORG_BARCODE, self.barcodes())
            self.add_row_attr(BARCODE, self.barcodes() + '-' + self.row_attrs[SAMPLE])

    def drop(self, values, value_type=None):
        """
        Drops the given values from the assay.

        Parameters
        ----------
        values : list-like
            The values that are to be dropped,
            which could be barcodes or ids.
        value_type : str
            Accepted values are `id`, `barcode`, and
            `None`. When `None`, the type is guessed
            based on the values.

        Returns
        -------
        :class:`missionbio.mosaic.assay._Assay`
            A copy of itself with the relevant
            ids removed.

        Raises
        ------
        ValueError
            Raise when the type of the value (barcode or id)
            is not determinable, and value_type is expected.
        """

        remaining_bars = None
        remaining_ids = None

        def check_complete(array):
            nonlocal values
            if not np.isin(values, array).all():
                raise ValueError('Not all the given values are present in the assay.')

        if value_type == 'id':
            check_complete(self.ids())
            remaining_ids = ~np.isin(self.ids(), values)
        elif value_type == 'barcode':
            check_complete(self.barcodes())
            remaining_bars = ~np.isin(self.barcodes(), values)
        elif value_type is None:
            if np.isin(values, self.ids()).any():
                check_complete(self.ids())
                remaining_ids = ~np.isin(self.ids(), values)
            elif np.isin(values, self.barcodes()).any():
                check_complete(self.barcodes())
                remaining_bars = ~np.isin(self.barcodes(), values)
            else:
                raise ValueError('Not all the given values are present in the assay.')
        else:
            raise ValueError('value_type must be `id`, `barcode`, or `None`.')

        return self[remaining_bars, remaining_ids]

    def del_layer(self, key):
        """
        Delete a layer from the assay layers.

        Parameters
        ----------
        key : str
            The name of the layer.
        """
        if key in self.layers:
            del self.layers[key]

    def del_row_attr(self, key):
        """
        Delete an attribute from the row_attrs.

        Parameters
        ----------
        key : str
            The name of the row attribute.
        """
        if key in self.row_attrs:
            del self.row_attrs[key]

    def del_col_attr(self, key):
        """
        Delete an attribute from the col_attrs.

        Parameters
        ----------
        key : str
            The name of the column attribute.
        """
        if key in self.col_attrs:
            del self.col_attrs[key]

    def del_metadata(self, key):
        """
        Delete an attribute from the metadata.

        Parameters
        ----------
        key : str
            The name of the metadata.
        """
        if key in self.metadata:
            del self.metadata[key]

    def get_attribute(self, attribute, features=None, constraint=None):
        """
        Retrieve any attribute in the assay

        Returns an np.array which could either be
        a row attribute, column attribute, layer
        or the passed attribute itself.

        Parameters
        ----------
        attribute : str / np.ndarray
            The attribute to be searched for in the assay.

        features : str
            In case the attribute is a layer or col, then
            the subset of features to select

        constraint : str
            One of the following is accepted.
                1. None
                    No contraint.
                2. 'row'
                    The first dimension must be equal
                    to the number of cells.
                3. 'col'
                    The second dimension must be equal
                    to the number of ids or the number of
                    features given in the input.
                4. 'row+col'
                    The dimension must be exactly
                    (number of cells, number of ids).
                    The layers have this shape.

        Returns
        -------
        pd.DataFrame
            The array of the attribute with the given name
            found in the assay layer, row attributes,
            or col attributes in that order. If a constraint
            is provided, the array is reshaped appropriately
            if possible. In case `constraint` is provided then
            the columns and index are named based on the
            barcodes, ids, or a range depending on where
            the given attribute is found.

        Raises
        ------
        ValueError
            When the attribute is not found or
            when the constraint is not satisfied.
        TypeError
            When the attribute is neither a str
            not an np.ndarray

        Notes
        -----
        In case the constraint can reshape the array
        into the expected shape then no error will be raised.
        Eg. An assay with 100 barcodes and 10 ids has a shape
        (100, 10). When the attribute 'barcode' is fetched
        constrained by 'col' it will not raise an error,
        but rather return a 10x10 dataframe of barcodes.

        >>> assay.shape
        (100, 10)
        >>> attr = assay.get_attribute('barcode', constrain='col')
        >>> attr.shape
        (10, 10)

        Possible expected behavior
        >>> assay.shape
        (100, 11)
        >>> attr = assay.get_attribute('barcode', constrain='col')
        ValueError - 'The given attribute does not have the expected
        shape nor could be reshaped appropriately.'
        """

        # Find where the attribute is
        if isinstance(attribute, str):
            title = attribute

            if attribute in self.row_attrs and attribute in self.layers:
                warnings.warn(f'{attribute} found in the row attributes and the layers both.'
                              f' Defaulting to the {attribute} in the layers.')

            if attribute in self.col_attrs and attribute in self.layers:
                warnings.warn(f'{attribute} found in the col attributes and the layers both.'
                              f' Defaulting to the {attribute} in the layers.')

            if attribute in self.col_attrs and attribute in self.row_attrs:
                warnings.warn(f'{attribute} found in the col attributes and the row attributes both.'
                              f' Defaulting to the {attribute} in the row attributes.')

            if attribute in self.layers:
                attribute = self.layers[attribute]
                if features is None:
                    features = self.ids()
                elif np.isin(features, self.ids()).all():
                    inds = [np.where(self.ids() == f)[0][0] for f in features]
                    attribute = attribute[:, inds]
                else:
                    raise ValueError('Not all features are found in the arrays ids')
            elif attribute in self.row_attrs:
                attribute = self.row_attrs[attribute]
            elif attribute in self.col_attrs:
                attribute = self.col_attrs[attribute]
                if features is None:
                    features = self.ids()
                elif np.isin(features, self.ids()).all():
                    inds = [np.where(self.ids() == f)[0][0] for f in features]
                    attribute = attribute[inds, :]
                else:
                    raise ValueError('Not all features are found in the arrays ids')
            else:
                raise ValueError(f'{attribute} not found in either the row attributes, col attributes or the layers')
        elif isinstance(attribute, np.ndarray):
            title = '-'
        else:
            raise TypeError("'attribute' must be of type str or np.ndarray")

        # Constraint the attribute
        attribute = attribute.copy()
        err_msg = 'The given attribute does not have the expected shape nor could be reshaped appropriately.'
        if constraint == 'row':
            try:
                attribute = attribute.reshape((self.shape[0], -1))
            except ValueError:
                raise ValueError(f'{err_msg} Given {attribute.shape}, expected ({self.shape[0]}, *).')
        elif constraint == 'col':
            try:
                if features is None:
                    attribute = attribute.reshape((-1, self.shape[1]))
                else:
                    attribute = attribute.reshape((-1, len(features)))
            except ValueError:
                raise ValueError(f'{err_msg} Given {attribute.shape}, expected (*, {self.shape[1]}).')
        elif constraint == 'row+col':
            try:
                if features is None:
                    attribute = attribute.reshape(self.shape)
                else:
                    attribute = attribute.reshape((self.shape[0], len(features)))
            except ValueError:
                raise ValueError(f'{err_msg} Given {attribute.shape}, expected {self.shape}.')
        elif constraint is not None:
            raise TypeError(f'Constraint must be one of None, "row", "col", "row+col". Given {constraint}')

        # Mark the indices and columns appropriately
        attribute = pd.DataFrame(attribute)
        if constraint is not None:
            if features is None:
                attribute.columns = (np.arange(attribute.shape[1]) + 1).astype(str)
            elif len(features) == attribute.shape[1]:
                attribute.columns = features
            else:
                raise ValueError(f"Expected {attribute.shape[1]} features, given {len(features)}")

            if 'row' in constraint:
                attribute.index = self.barcodes()

        attribute.index.name = title

        return attribute

    def barcodes(self, label=None):
        """
        Get the list of barcodes.

        Parameters
        ----------
        label : list
            The list of labels whose barcodes are to be retrieved.
            If `None`, then all barcodes are returned.

        Returns
        -------
        numpy.ndarray
             An array of barcode ids.
        """

        bars = np.array(self.row_attrs[BARCODE])

        if isinstance(label, str):
            label = np.array([label])

        if label is None:
            return bars
        else:
            return bars[get_indexes(self.get_labels(), label)]

    def clustered_barcodes(self, orderby=None, splitby=LABEL, override=False):
        """
        Hierarchical clustering of barcodes.

        If the layer is not provided, the barcodes are
        returned according to the labels.

        If a layer is provided, then further hierarchical
        clustering is performed based on the values
        in this layer for each label.

        Parameters
        ----------
        orderby : str / np.ndarray
            An attribute with at least one dimension equal
            to the number of barcodes. Uses :meth:`_Assay.get_attribute`
            to retrieve the values constrained by `row`.
        splitby : bool
            Whether to order the barcodes based on the given
            labels or not. Only applicable when labels `orderby`
            is `None`. Uses :meth:`_Assay.get_attribute` to
            retrieve the values constrained by `row`. The shape
            must be equal to `(#cells)`
        override : bool
            Continue clustering even if the number of `id`
            is greater than 1,000.

        Returns
        -------
        numpy.ndarray
            Ordered barcodes

        Raises
        ------
        ValueError
            When the shape of orderby is not appropriate

        Exception
            When the number of `id` in the assay is greater
            than 1,000. This can take a long time
            to process, hence denied.

        Warning
            If the number of `id` is greater than 1,000 and
            override is passed.
        """

        if splitby is None:
            splitby = np.array(['-'] * self.shape[0])
        else:
            splitby = self.get_attribute(splitby, constraint='row').values.flatten()

            if len(splitby) != self.shape[0]:
                raise ValueError('Only 1-D values for `splitby` are permitted.')

        if orderby is None:
            return self.barcodes()[splitby.argsort()]
        else:
            df = self.get_attribute(orderby, constraint='row')

            if df.shape[1] > 10 ** 3:
                message = f"Clustering using {df.shape[1]} features. This might take a long time."
                if override:
                    warnings.warn(message)
                else:
                    raise Exception(f"{message} Pass 'override' to continue clustering.")

            df.loc[:, LABEL] = splitby

            # Cluster at the level of labels
            if len(set(splitby)) > 1:
                df_avg = df.groupby(LABEL).mean()

                Z = hierarchy.linkage(df_avg, metric='cityblock', method='average')
                order = hierarchy.leaves_list(Z)[::-1]
                label_order = df_avg.index[order]

                df.loc[:, LABEL] = pd.Categorical(splitby, label_order)

            df = df.sort_values(by=LABEL)

            # Cluster barcodes within each label
            leaf_order = []
            cells_done = 0
            for lab, df_lab in df.groupby(LABEL):
                df_lab = df_lab.drop(LABEL, axis=1).copy()
                if (df_lab.shape[0] == 1):
                    order = np.array([0])
                else:
                    Z = hierarchy.linkage(df_lab, metric='cityblock', method='average')
                    order = hierarchy.leaves_list(Z)[::-1]
                leaf_order.extend(order + cells_done)
                cells_done += len(order)

            return df.index[leaf_order].values

    def ids(self):
        """
        Get the list of ids.

        Returns
        -------
        np.ndarray
            The list of ids in the column attribute 'id'.
        """
        return self.col_attrs[ID]

    def clustered_ids(self, orderby, features=None, override=False):
        """
        Hierarchical clustering of ids.

        Parameters
        ----------
        orderby : str / np.ndarray
            An attribute with at least one dimension equal
            to the number of ids in the assay. Uses
            :meth:`_Assay.get_attribute` to retrieve the value
            constrained by `col`
        features : list-like
            The subset of ids to use and return, in case orderby
            is found in the layers, otherwise it is the name
            of the ids to be given to the orderby attribute.
        override : bool
            Continue clustering even if the number of `id`
            is greater than 1,000.

        Returns
        -------
        numpy.ndarray
            ordered ids

        Raises
        ------
        ValueError
            When the labels are not set.

        Exception
            When number of `id` in the assay is greater
            than 1,000. This can take a long time
            to process, hence denied.

        Warning
            If the number of `id` is greater than 1,000 and
            override is passed.
        """

        orderby = self.get_attribute(orderby, constraint='col', features=features)
        features = orderby.columns
        orderby = orderby.values

        if orderby.shape[1] > 10 ** 3:
            message = f"Clustering using {orderby.shape[1]} ids. This might take a long time."
            if override:
                warnings.warn(message)
            else:
                raise Exception(f"{message} Pass 'override' to continue clustering.")

        df = pd.DataFrame(orderby, columns=features)

        # No need to reorder if fewer than 2 features
        if df.T.shape[0] < 2:
            return df.columns

        Z = hierarchy.linkage(df.T, metric='cityblock', method='average')
        order = hierarchy.leaves_list(Z)[::-1]

        return df.columns[order]

    def get_palette(self):
        """
        Get the label color palette.

        Returns
        -------
        dictionary
            The keys are the names of the labels
            and the values are the hex color codes.
        """
        if PALETTE not in self.metadata:
            self.set_palette()

        return {v[0]: v[1] for v in self.metadata[PALETTE]}

    def set_palette(self, palette=None):
        """
        Set the label color palette.

        Modifies the `palette` metadata of the assay.

        Parameters
        ----------
        palette : dictionary
            The color values for each label.
            A subset of labels is not accepted.

        Raises
        ------
        TypeError
            When the provided palette
            is not a dictionary.
        ValueError
            When one of the labels is not
            present in the given palette.
        """

        unique_labels = np.unique(self.get_labels())
        pal = []

        if palette is None:
            for i in range(len(unique_labels)):
                pal.append([unique_labels[i], COLORS[i % len(COLORS)]])
        else:
            if not isinstance(palette, dict):
                raise TypeError('The palette should be of type dictionary.')

            for lab in unique_labels:
                if lab not in palette:
                    raise ValueError(f'The color value {lab} is missing from the given palette - {palette}.')
                pal.append([lab, palette[lab]])

        self.add_metadata(PALETTE, np.array(pal))

    def get_labels(self, barcodes=None):
        """
        Get the labels corresponding to the barcodes.

        Parameters
        ----------
        barcodes : list-like / None
            The set of barcodes for which the
            labels are to be returned. The order
            of the barcodes is maintained. The
            LABEL row attribute is returned when
            `barcodes` is `None`

        Returns
        -------
        np.array
            The list of labels.
        """
        if LABEL not in self.row_attrs:
            return np.array(['-'] * self.shape[0])

        all_labs = self.row_attrs[LABEL].copy()
        if barcodes is None:
            return all_labs

        all_bars = self.barcodes()
        labels = []
        for bar in barcodes:
            labels.append(all_labs[np.where(all_bars == bar)[0][0]])

        labels = np.array(labels)

        return labels

    def set_labels(self, clusters, others_label='others'):
        """
        Set the labels for the barcodes.

        Modifies the `id` row attribute of the assay.

        Parameters
        ----------
        clusters : list or dictionary
            If a list, then the order of the barcodes
            should be considered because the labels
            are set in the same order as the
            order of barcodes.
        others_label : str
            In case not all barcodes are labeled,
            then this is the label of the remaining
            barcodes.
        """

        if isinstance(clusters, np.ndarray) or isinstance(clusters, list):
            labels = np.array(clusters)
        else:
            barcodes = self.barcodes()
            labels = np.array([others_label] * len(barcodes), dtype=object)
            for cluster_name in clusters:
                bars = clusters[cluster_name]
                if not np.isin(bars, self.barcodes()).all():
                    raise ValueError(f'One of the barcodes in {cluster_name} was not found in the assay.')
                indexes = get_indexes(barcodes, bars)
                labels[indexes] = cluster_name

        self.add_row_attr(LABEL, labels.astype(str))
        self.set_palette()

    def set_selected_labels(self):
        """
        Custom clustering from scatter plots.

        Setting the labels of the barcodes based on
        the clusters selected on the scatter plot.

        Modifies the `id` row attribute of the assay.
        """

        i = 1
        labs = {}
        pal = {}
        for k in self.selected_bars:
            if self.selected_bars[k].size > 0:
                labs[str(i)] = self.selected_bars[k]
                pal[str(i)] = k
                i += 1

        self.set_labels(labs)
        self.set_palette(pal)

    def rename_labels(self, label_map):
        """
        Change the name of the labels.

        Parameters
        ----------
        label_map : dictionary
            The key is the existing name of the label, and
            the value is the new name of the label.
        """

        palette = self.get_palette()
        labels = self.get_labels().copy()

        for k in label_map:
            labels = np.where(labels == k, label_map[k], labels)
            palette[label_map[k]] = palette[k]
            del palette[k]

        self.add_row_attr(LABEL, labels)
        self.set_palette(palette)

    # --------- Tertiary analysis methods

    def cluster(self, attribute, method='dbscan', **kwargs):
        """
        Identify clusters of cells.

        Modifies the `label` row attribute of the assay.

        Parameters
        ----------
        attribute : str / np.array
            The attribute to use as coordinates for clustering.
            Uses :meth:`_Assay.get_attribute` to retrieve the values
            constrained by `row`.
        method : str
            dbscan, hdbscan, kmeans, or graph-community (Louvain
            on a shared nearest neighbor graph).
        kwargs : dictionary
            Passed to the appropriate clustering method.

            Available parameters for the methods are:
                1. dbscan : eps (float)
                    A measure of how close the points are.
                2. hdbscan : min_cluster_size (int)
                    The number of cells in a cluster.
                3. kmeans : n_clusters (int)
                    The number of clusters to be generated.
                4. graph-community : k (int)
                    The number of nearest neighbors to consider
                    in the shared nearest neighbor graph generation.

        Raises
        ------
        ValueError
            When the row attribute is not available in the assay.
        Exception
            When an unsupported clustering method is provided.
        """

        supported_methods = {
            'dbscan': DBSCAN,
            'hdbscan': HDBSCAN,
            'kmeans': KMeans,
            'graph-community': self._community_clustering
        }

        if method in supported_methods:
            data = self.get_attribute(attribute, constraint='row').values
            if method == 'graph-community':
                self._community_clustering(data, **kwargs)
            else:
                clusterer = supported_methods[method](**kwargs).fit
                output = clusterer(data)
                labels, idx, cnt = np.unique(output.labels_, return_inverse=True, return_counts=True)

                # Order by counts and maintain -1 for outliers
                order = cnt.argsort()[::-1]
                clones = labels[order] != -1
                labels[order[clones]] = np.arange(clones.sum()) + 1
                labels[order[~clones]] = -1
                labels = labels[idx]

                self.set_labels(labels)
        else:
            raise Exception(f'{method} not supported. Try {supported_methods.keys()}.')

    def _community_clustering(self, data, k=20, show_plot=False, **kwargs):
        """
        Louvain clustering

        Modifies the `id` row attribute of the assay.

        This is a graph-based clustering approach where
        the graph structure uses the Jaccard similarity coefficient as
        the edge weights, which is followed by a community
        cluster detection method (Louvain).

        Currently, the highest k value is used to set labels.
        The user expected is expected to try various k values.

        The plot for k vs number of clusters is provided to
        the user to interpret the accuracy of the k selected.

        Parameters
        ----------
        data : numpy.ndarray
            The array to use to create an SNN graph.
        k : int
            The number of nearest neighbors to consider
            when creating the SNN graph.
        show_plot : bool
            If true, then a plot for various k-values is
            shown to ascertain whether the provided k-value
            is sufficient.
        kwargs : dict
            Passed to the NearestNeighbors algorithm.

        Raises
        ------
        ValueError
            When the igraph package is not installed
        """

        try:
            import igraph
        except ImportError as err:
            raise ValueError("The igraph library is required for community clustering.") from err

        # Find the nearest neighbors
        print("Creating the Shared Nearest Neighbors graph.")
        knn = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', **kwargs).fit(data)
        dist, nbrs = knn.kneighbors(data)

        # Set the edges
        v1 = 0
        edges = []
        for v1_nbrs in nbrs:
            for v2 in v1_nbrs:
                edges.append((v1, v2))
            v1 += 1

        edges = np.array(edges)

        # Set the weight using Jaccard similarity
        # Optimize this step further, if possible
        print('-' * 50)
        i = 0
        weights = []
        similarities = {}
        nbrs = [set(t) for t in nbrs]
        for e in edges:
            if (i % (edges.shape[0] // 50) == 0):
                print('#', flush=True, end='')

            i += 1
            if (e[0], e[1]) not in similarities:
                intersection = len(nbrs[e[0]].intersection(nbrs[e[1]]))
                common = 2 * k - intersection
                similarity = intersection / common
                similarities[(e[1], e[0])] = similarity
            else:
                similarity = similarities[(e[0], e[1])]

            weights.append(similarity)

        # Construct the graphs and identify clusters for various values of k less than the given value
        print("\nIdentifying clusters using Louvain community detection.")
        edges = np.array(edges).reshape(len(nbrs), k, 2)
        weights = np.array(weights).reshape(len(nbrs), k, 1)
        sizes = np.linspace(10, k, 5) if show_plot else [k]
        clusters_found = []
        for size in sizes:
            size = int(size)
            sub_edges = np.array(edges)[:, :size, :]
            sub_edges = sub_edges.reshape(
                sub_edges.shape[0] * sub_edges.shape[1], sub_edges.shape[2])
            sub_weights = np.array(weights)[:, :size]
            sub_weights = sub_weights.reshape(
                sub_weights.shape[0] * sub_weights.shape[1])

            g = igraph.Graph()
            g.add_vertices(len(self.barcodes()))
            g.add_edges(sub_edges)
            g.es['weight'] = sub_weights

            vc = g.community_multilevel(weights='weight')
            vc_labels = np.array(vc.as_cover().membership).flatten()

            clusters_found.append((size, len(set(vc_labels))))

        print(f'\nNumber of clusters found: {clusters_found[-1][1]}.')
        print(f'Modularity: {vc.modularity:.3f}')

        if show_plot:
            x = np.array(clusters_found)[:, 0]
            y = np.array(clusters_found)[:, 1]
            plt.figure(figsize=(10, 10))
            plt.scatter(x, y)
            plt.ylabel('Number of clusters found')
            plt.xlabel('Number of nearest neighbors used')

        labels, idx, cnt = np.unique(vc_labels, return_inverse=True, return_counts=True)
        labels[cnt.argsort()[::-1]] = np.arange(len(labels)) + 1
        labels = labels[idx]
        self.set_labels(labels)

    def group_clusters(self, attribute):
        """
        Relabel clusters based on expression.

        Takes an assay object and reorders the
        cluster labels so that the clusters are
        grouped hierarchically. Ordering uses
        the median cluster value in the layer
        denoted by the input parameter "layer."

        For example, after implementation, all protein
        clusters with high CD3 might be numbered next
        to each other.

        Parameters
        ----------
        layer : str / np.array
            The name of the attribute from the assay object used
            to hierarchically group the clusters. Uses
            :meth:`_Assay.get_attribute` to retrieve the values
            constrained by `row`

        """

        labels = self.get_labels()

        u, lbl = np.unique(labels, return_inverse=True)

        nc = len(np.unique(lbl))

        if nc == 1:
            return

        data = self.get_attribute(attribute, constraint='row').values
        z = np.zeros((nc, len(self.ids())))
        for jj in range(nc):
            z[jj, :] = np.median(data[lbl == jj, :], axis=0)

        D = pdist(z, 'cosine')
        tree = hierarchy.linkage(D)
        s = hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(tree, D))

        s = np.argsort(s) + 1

        self.set_labels(s[lbl])

    def scale_data(self, layer, output_label=SCALED_LABEL):
        """
        Z-score normalization

        Ensures that all `id` have a mean
        of 0 and a standard deviation of 1.

        Adds `scaled_counts` to the layers.

        Parameters
        ----------
        layer : str / np.array
            The data to scale. Uses :meth:`_Assay.get_attribute`
            to retrieve the value constrained by `row+col`.
        output_label : str, default SCALED_LABEL
            The name of the output layer.
        """

        data = self.get_attribute(layer, constraint='row+col').values
        data = data - data.mean(axis=0)
        data = data / data.std(axis=0)
        self.add_layer(output_label, data)

    def run_pca(self, attribute, components, output_label=PCA_LABEL, show_plot=False, **kwargs):
        """
        Principal component analysis

        Adds `output_label` to the row attributes.

        Parameters
        ----------
        attribute : str
            The attribute to be used for PCA. Uses
            :meth:`_Assay.get_attribute` to retrieve
            the values constrained by `row`.
        components : int
            The number of PCA components to reduce to.
        output_label : str, default PCA_LABEL
            The name of the row attribute to store
            the data in.
        show_plot : bool
            Show the plot for the explained variance
            by the PCAs.
        kwargs : dict
            Passed to the PCA.

        Raises
        ------
        ValueError
            When both layer and attribute are provided.
            Only one is permitted at a time.
        """

        data = self.get_attribute(attribute, constraint='row').values
        components = np.minimum(components, data.shape[1])

        pca = PCA(n_components=components, **kwargs)
        pca.fit(data.T)
        pca_data = pca.components_.T
        self.add_row_attr(output_label, pca_data)

        if show_plot:
            plt.figure(figsize=(10, 10))
            plt.scatter(np.arange(components), pca.explained_variance_ratio_)
            plt.xlabel('Principal component')
            plt.ylabel('Explained variance ratio')
            plt.title(self.title)

    def run_umap(self, attribute, output_label=UMAP_LABEL, **kwargs):
        """
        Perform UMAP on the given data.

        Adds `output_label` to the row attributes.

        Parameters
        ----------
        attribute : str
            The attribute to be used for UMAP. Uses
            :met:`_Assay.get_attribute` to retrieve the values
            constrained by `row`.
        output_label : str, default UMAP_LABEL
            The name of the row attribute where the UMAP output is.
        kwargs : dictionary
            Passed to UMAP.

        Raises
        ------
        ValueError
            When both layer and attribute are provided.
            Only one is permitted at a time.
       """

        data = self.get_attribute(attribute, constraint='row').values

        fit = umap.UMAP(**kwargs)
        u = fit.fit_transform(data)

        # Sets UMAP x/y range ~[-10,10]
        u = stats.zscore(u) * 5

        self.add_row_attr(output_label, u)

    def run_lda(self, layer, attribute, output_label, cycles=1):
        """
        Perform LDA on a row attribute using a layer to type clusters.

        The row attribute is dimensionally reduced using UMAP.

        Adds `output_label` to the row attributes.

        Parameters
        ----------
        layer : str / np.array
            The layer used for cluster typing. Uses
            :meth:`_Assay.get_attribute` to retrieve
            the value constrained by `row+col`.
        attribute : str / np.array
            The row attribute to be used for UMAP. Uses
            :meth:`_Assay.get_attribute` to retrieve
            the value constrained by `row`.
        output_label : str
            The name of the row attribute where the UMAP output is.
        cycles : int
            The number of rounds of LDA to perform.
        kwargs : dictionary
            Passed to UMAP.
        """

        cnt = 0

        data = self.get_attribute(layer, constraint='row+col').values
        attribute = self.get_attribute(layer, constraint='row').values
        self.add_row_attr(output_label, attribute)

        for jj in range(cycles):

            u = self.row_attrs[output_label]
            nc = np.minimum(30, int(data.shape[0] / 20))

            kmeans = KMeans(n_clusters=nc, random_state=1).fit(u)
            lbl = kmeans.labels_

            clust_profile = np.zeros((nc, data.shape[1], 2))

            for ii in range(nc):
                clust_profile[ii, :, 0] = np.median(data[lbl == ii, :], axis=0)
                clust_profile[ii, :, 1] = np.std(data[lbl == ii, :], axis=0)

            clust_project = np.zeros((data.shape[0], nc))
            for ii in range(nc):
                coord = np.absolute(
                    (data - clust_profile[ii, :, 0][None, :]) / (clust_profile[ii, :, 1][None, :] + 0.0001))
                clust_project[:, ii] = 1 / (np.sum(np.minimum(coord, 10), axis=1) + .1)

            X = clust_project

            self.add_row_attr(output_label, X)

            self.run_umap(attribute=output_label,
                          output_label=output_label,
                          n_neighbors=50,
                          metric='cosine',
                          min_dist=0,
                          spread=1,
                          repulsion_strength=3,
                          negative_sample_rate=10,
                          random_state=42)

            cnt = cnt + 1
            print(f'Round {cnt} complete.')

    # --------- Plotting

    def scatterplot(self, attribute, colorby=None, features=None, title=''):
        """
        Scatter plot for all barcodes.

        Requires a 2-D row attribute.

        If `colorby` is None, then this plot can be used
        to create custom clusters using the plotly lasso tool.

        Not selecting any cell deselects all clusters, and double
        clicking refocuses all points.

        Parameters
        ----------
        attribute : str / np.array
            The row attribute to use as coordinates,
            This row attribute must be 2D. Uses :meth:`_Assay.get_attribute`
            to retrieve the values constrained by `row`.
            The shape must be `(#cells, 2)`.
        colorby : str / np.array
            The values used to color the scatterplot.
            Uses :meth:`_Assay.get_attribute` to retrieve
            the values constrained by `row`.
            The shape must be `(#cells, *)` if 2D, otherwise
            the shape must be `(#cells)`.
                1. None
                    The scatterplot is not colored and
                    highlighting using the lasso tool
                    can be used to select subpopulations.
                2. Any 1D attribute
                    In case 'label' is provided then the
                    stored paltte is used. If the values
                    are strings, then a discrete color map
                    is assumed. For numerical values a
                    continuous color scale is used.
                3. Any 2D attribute
                    One dimension must match the number of
                    cells in the assay.
                4. 'density'
                    Colors the plot based on the density of
                    the nearby points.
        features : list-like
            In the case that `colorby` is a layer then
            features sets the subset of features to
            color the plot by.

            In the case that `colorby` is a 2D array
            features sets the title of each scatterplot.
            In this case, the length of `features`
            must match the number of columns in the
            `colorby` 2D array.
        title : str
            The title given to the plot after
            the name of the assay.

        Returns
        -------
        fig : plotly.graph_objs.FigureWidget

        Raises
        ------
        ValueError
            Raised when the attribute is not present in the assay
            or when length of `features` does not match the second
            dimension of `colorby`.
        """

        if features is not None and colorby is None:
            raise ValueError("'colorby' must be provided when features are given.")

        # Get data
        if isinstance(attribute, str):
            xlabel, ylabel = f'{attribute}-1', f'{attribute}-2'
        else:
            xlabel, ylabel = 'x', 'y'

        data = self.get_attribute(attribute, constraint='row+col', features=[xlabel, ylabel])
        data.loc[:, 'hovertext'] = 'label: ' + self.get_labels().astype(object) + '<br>' + self.barcodes().astype(object)

        # Determine source of colorby
        colorby_type = colorby if isinstance(colorby, str) else '----'
        if colorby_type == 'density':
            x = data.iloc[:, 0]
            y = data.iloc[:, 1]
            xy = np.vstack([x, y])
            colorby = gaussian_kde(xy)(xy)
            idx = colorby.argsort()
            data, colorby = data.iloc[idx, :], colorby[idx][:, None]
        elif colorby is not None:
            feature_values = self.get_attribute(colorby, constraint='row', features=features)
            colorby = feature_values.values
            features = feature_values.columns.values

        # Plot
        layout = dict(xaxis=dict(zeroline=False, title_text=xlabel),
                      yaxis=dict(zeroline=False, title_text=ylabel),
                      title=f'{self.title}{title}',
                      legend_tracegroupgap=0,
                      dragmode='pan',
                      hovermode='closest',
                      template='simple_white',
                      coloraxis_colorbar_title=colorby_type)

        if colorby is None:
            p = px.scatter(data, x=xlabel, y=ylabel, hover_data=['hovertext'])
            p.data[0].marker.color = np.array([COLORS[0]] * self.shape[0])
            fw = go.FigureWidget(data=p.data, layout=layout)
            fw.update_layout(dragmode='lasso', width=600, height=550)
            fw.update_traces(marker_size=4, hovertemplate='%{customdata}<extra></extra>')
            self.selected_bars = {COLORS[0]: self.barcodes()}

            fw.data[0].on_selection(self._brush)

        elif colorby.shape[1] == 1:
            val_1, val_99 = clipped_values(colorby.flatten())

            if colorby_type == LABEL:
                p = px.scatter(data, x=xlabel, y=ylabel, hover_data=['hovertext'],
                               color=self.get_labels(), color_discrete_map=self.get_palette())
            else:
                if not isinstance(val_1, str):
                    data['hovertext'] = np.round(colorby, 2).astype(str)
                    layout.update(coloraxis_cmax=val_99, coloraxis_cmin=val_1)

                p = px.scatter(data, x=xlabel, y=ylabel, hover_data=['hovertext'],
                               color=colorby.flatten(), color_discrete_sequence=COLORS)

            fw = go.FigureWidget(data=p.data, layout=layout)
            fw.update_traces(marker_size=4, hovertemplate='<b>%{customdata}</b><extra></extra>')

            if isinstance(val_1, str):
                fw.update_layout(dragmode='lasso', width=600, height=550)
                for d in fw.data:
                    d.on_selection(self._brush)
                    self.selected_bars[d.marker.color] = np.array([i.split('<br>')[-1] for i in d.customdata.flatten()])
                    d.marker.color = np.array([d.marker.color] * len(d.x))
            else:
                fw.update_layout(dragmode='zoom', width=600, height=550)

        else:
            val_1, val_99 = clipped_values(colorby.flatten())

            datas = pd.DataFrame()
            for feature in features:
                d = data.copy()
                d.loc[:, 'value'] = feature_values.loc[:, feature].values
                d.loc[:, 'name'] = feature
                datas = pd.concat([datas, d])

            nplots = len(features)
            nrows = round(nplots ** 0.5)
            ncols = nplots // nrows + min(1, nplots % nrows)

            p = px.scatter(datas, x=xlabel, y=ylabel, color='value', hover_data=['hovertext'],
                           facet_row_spacing=max(0.01, 0.2 / nrows), facet_col_spacing=max(0.01, 0.2 / ncols),
                           facet_col='name', facet_col_wrap=ncols, color_discrete_sequence=COLORS,
                           width=max(600, 350 * ncols), height=max(350 * nrows, 550 / ncols))
            p.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

            fw = go.Figure(p)

            if not isinstance(val_1, str):
                layout.update(coloraxis=dict(cauto=False,
                                             cmax=val_99,
                                             cmin=val_1,
                                             colorbar=dict(thickness=25,
                                                           len=1 / nrows,
                                                           yanchor="top",
                                                           y=1.035,
                                                           x=1.05,
                                                           ticks="outside")))
            fw.update_layout(layout)
            fw.update_traces(marker_size=4, hovertemplate='<b>%{customdata}</b><extra></extra>')

        return fw

    def heatmap(self, attribute, splitby=LABEL, features=None, bars_order=None, convolve=0, title=''):
        """
        Heatmap of all barcodes and ids.

        Parameters
        ----------
        attribute : str / np.ndarray
            An attribute with the shape equal to the shape
            of the assay. Uses :meth:`_Assay.get_attribute` to
            retrieve the values constrained by `row+col`.
        splitby : str / np.ndarray, default LABEL
            Whether to order the barcodes based on the given
            labels or not. Only applicable when `bars_order`
            is `None`. Uses :meth:`_Assay.get_attribute` to
            retrieve the values constrained by `row`.
            The shape must be equal to `(#cells)`.
        features : list
            The ids that are to be shown. This also sets the
            order in which the ids are shown.
        bars_order : list
            The barcodes that are to be plotted.
            The order in the plot is the same
            as the order in the list. Passing this
            sets `splitby` to `None`.
        convolve : float [0, 100]:
            The percentage of barcodes from the label with
            the fewest barcodes that is used to average
            out the signal. If `0`, then no convolution
            is performed, and if `100`, the mean per
            label is returned. Only applicable when
            `splitby` is not `None`.
        title : str
            The title to be added to the plot.

        Returns
        -------
        fig : plotly.graph_objects.FigureWidget

        Raises
        -------
        ValueError
            Raised in the following cases.
            When convolve is below 0 or above 100.
            When the number of ids is too large to hierarchically
            cluster the cells and features is not provided.
       """

        # Check parameters
        if convolve < 0 or convolve > 100:
            raise ValueError('Convolve must be within [0, 100].')

        if features is None:
            features = self.clustered_ids(orderby=attribute)

        # Get data
        if bars_order is None:
            bars_order = self.clustered_barcodes(orderby=attribute, splitby=splitby)

        if splitby is None:
            splitby = LABEL
            labels = self.get_labels()
        else:
            labels = self.get_attribute(splitby, constraint='row')
            splitby = labels.index.name
            labels = labels.values.flatten()

            if len(labels) != self.shape[0]:
                raise ValueError('Only 1-D values for `splitby` are permitted.')

        labels = labels[np.where(self.barcodes() == bars_order[:, None])[1]]

        if not np.isin(features, self.ids()).all():
            # @HZ
            print("Warning: not all the given features were found in the assay ids.")
            #raise ValueError("Not all the given features were found in the assay ids.")
            
            data = self.get_attribute(attribute, constraint='row', features=np.intersect1d(features, self.ids()).tolist())

            # append the missing features as undetected (value = 0)
            for i in list(set(features) - set(self.ids())):
                print(f'{i} not in ids')
                data[i] = np.zeros((data.shape[0], 1)) # @HZ: potentially problematic because it only applies to dna

            data = data[features]
        else:
            data = self.get_attribute(attribute, constraint='row', features=features)

        data = data.loc[bars_order, :]

        # Find ticks and convolve data if needed
        un_labs, idx, inv_labs, cnt = np.unique(labels, return_index=True, return_inverse=True, return_counts=True)
        un_ordered_labs, cnt_ordered_labs = un_labs[idx.argsort()], cnt[idx.argsort()]
        un_continuous = np.ones(len(labels), dtype=bool)
        un_continuous[1:] = labels[1:] != labels[:-1]
        ordered = len(labels[un_continuous]) == len(un_labs)

        N = int(max(1, round(min(cnt_ordered_labs) * convolve / 100)))
        inv_labs = inv_labs + 1
        tickvals = []
        ticktexts = []
        lab_color = []
        col_palette = {}
        for i in range(len(un_ordered_labs)):
            col_palette[un_ordered_labs[i]] = self.get_palette()[un_ordered_labs[i]] if splitby == LABEL else COLORS[i]

        for i in range(len(cnt_ordered_labs)):
            lab_color.append((i / len(un_ordered_labs), col_palette[un_labs[i]]))
            lab_color.append(((i + 1) / len(un_ordered_labs), col_palette[un_labs[i]]))

            if ordered:
                start = cnt_ordered_labs[:i].sum()
                end = cnt_ordered_labs[:i + 1].sum()
                vals = data.iloc[start:end, :]
                tickvals.append(int((start + end) / 2))
                ticktexts.append(f'<b>{un_ordered_labs[i]}</b> {vals.shape[0] / data.shape[0]:.1%}')

                if convolve == 100:
                    vals = vals.apply(lambda x: [x.mean()] * vals.shape[0])
                elif convolve > 0:
                    vals = vals.apply(lambda x: np.convolve(x, np.ones(N) / N, mode='same'))

                data.iloc[start:end, :] = vals

        # Draw labels
        fig = make_subplots(rows=1, cols=2,
                            shared_yaxes=True,
                            horizontal_spacing=0.01,
                            column_widths=[1 / 25, 24 / 25])

        labs = go.Heatmap(z=inv_labs,
                          y=np.arange(data.shape[0]),
                          x=[0] * data.shape[0],
                          customdata=labels,
                          colorscale=lab_color,
                          hovertemplate='label: %{customdata}<extra></extra>',
                          showlegend=False,
                          showscale=False)
        fig.add_trace(labs, row=1, col=1)

        # Draw main heatmap
        labels = np.tile(labels[:, None], (1, data.shape[1]))
        vals = go.Heatmap(z=data,
                          y=np.arange(data.shape[0]),
                          x=data.columns,
                          customdata=labels,
                          coloraxis='coloraxis',
                          hovertemplate='%{z:.2f}<br>%{x}<extra>%{customdata}</extra>',
                          showlegend=False,
                          showscale=False)
        fig.add_trace(vals, row=1, col=2)

        # Add dividers
        if ordered:
            for i in range(len(cnt_ordered_labs) - 1):
                start = cnt_ordered_labs[:i + 1].sum()
                fig.add_hline(start - 0.5, line_color='rgba(255,255,255,1)')
        else:
            tickvals = []
            ticktexts = []

        # Update layout
        val_1, val_99 = clipped_values(data.values.flatten())
        fig.update_layout(coloraxis_colorscale='magma',
                          legend_tracegroupgap=0,
                          width=800,
                          height=800,
                          title_text=f'{self.title}{title}',
                          template='plotly_white',
                          coloraxis=dict(cmax=val_99,
                                         cmin=val_1,
                                         cauto=False,
                                         colorbar=dict(thickness=15,
                                                       len=0.3,
                                                       yanchor="bottom",
                                                       y=0,
                                                       ticks="outside",
                                                       title=data.index.name)),
                          yaxis=dict(type='category',
                                     tickvals=tickvals,
                                     ticktext=ticktexts,
                                     tickformat='{^text}',
                                     range=[0, data.shape[0]]),
                          yaxis2=dict(type='category',
                                      range=[0, data.shape[0]]),
                          xaxis=dict(ticktext=data.columns,  # Added for heatmap selection
                                     showticklabels=False),
                          xaxis2=dict(tickangle=-90,
                                      tickvals=np.arange(data.shape[1]),
                                      ticktext=data.columns,
                                      tickmode='array'))

        ann_text = f'{data.shape[0]} cells, {data.shape[1]} features'
        if convolve != 0 and ordered:
            ann_text = ann_text + '<br>Values smoothed using a moving average.'

        ann = go.layout.Annotation(text=ann_text,
                                   xanchor='left',
                                   yanchor='bottom',
                                   align='left',
                                   showarrow=False,
                                   xref='x2',
                                   yref='y2',
                                   x=-0.5,
                                   y=data.shape[0],
                                   font_size=10)
        fig.add_annotation(ann)

        self.selected_ids = np.array([])
        self.__heatmap = go.FigureWidget(fig)
        self.__heatmap.data[1].on_click(self._heatmap_selection)

        return self.__heatmap

    def violinplot(self, attribute, splitby=None, features=None, title=''):
        """
        Violin plot for all barcodes.

        Parameters
        ----------
        attribute : str / np.array
            The attribute used to plot the values on the violinplot.
            Uses :meth:`_Assay.get_attribute` to retrieve the values.
            constrained by 'row'.
        splitby : str / np.array
            The values used to split and color the violinplot.
            Uses :meth:`_Assay.get_attribute` to retrieve
            the values. The shape must be `(#cells)`.
                1. None
                    The violinplot is not colored.
                2. Any 1D attribute
                    In case 'label' is provided then the
                    stored paltte is used. A discrete color map
                    is used for all other cases.
        features : list-like
            In the case that `attribute` is a layer then
            features sets the subset of features to plot.
            Otherwise it sets the name of each feature shown.
        title : str
            The title given to the plot after
            the name of the assay.

        Returns
        -------
        fig : plotly.graph_objects.Figure
        """

        data = self.get_attribute(attribute, constraint='row', features=features)
        if features is None:
            features = data.columns.values

        layout = dict(width=min(1200, data.shape[1] * 200),
                      height=550,
                      xaxis=dict(zeroline=False),
                      yaxis=dict(zeroline=False),
                      title=f'{self.title}{title}',
                      dragmode='zoom',
                      hovermode='closest',
                      template='simple_white')

        if splitby is None:
            data = data.melt(var_name=ID)
            fig = go.Figure(layout=layout)

            for i, df_i in data.groupby(ID):
                fig.add_trace(go.Violin(x=df_i[ID],
                                        y=df_i['value'],
                                        scalemode='width',
                                        scalegroup=i,
                                        box_visible=True,
                                        box_width=0.05,
                                        spanmode='hard',
                                        points=False,
                                        line_color=COLORS[0],
                                        showlegend=False))
        else:
            splitby = self.get_attribute(splitby, constraint='row')
            if splitby.shape[1] != 1:
                raise ValueError('Only one dimensional `splitby` values are permitted.')

            data.loc[:, LABEL] = splitby.values
            data = data.melt(id_vars=LABEL, var_name=ID)

            un_labs = np.unique(splitby)
            line_colors = {}
            for i in range(len(un_labs)):
                line_colors[un_labs[i]] = self.get_palette()[un_labs[i]] if splitby.index.name == LABEL else COLORS[i]

            fig = go.Figure(layout=layout)
            for grp, df in data.groupby([LABEL, ID]):
                lab, i = grp
                showlegend = features[0] == str(i)
                fig.add_trace(go.Violin(x=df[ID],
                                        y=df['value'],
                                        scalemode='width',
                                        scalegroup=str(i) + str(lab),
                                        box_visible=True,
                                        box_width=0.02,
                                        meanline_visible=True,
                                        spanmode='hard',
                                        points=False,
                                        legendgroup=str(lab),
                                        name=str(lab),
                                        line_color=line_colors[lab],
                                        showlegend=showlegend))

            fig.update_layout(violinmode='group',
                              width=min(1200, data.shape[1] * len(set(self.get_labels())) * 100))

        return fig

    def ridgeplot(self, attribute, splitby=None, features=None, title=''):
        """
        Ridge plot for all barcodes.

        Parameters
        ----------
        attribute : str / np.array
            The attribute used to plot the values on the ridgeplot.
            Uses :meth:`_Assay.get_attribute` to retrieve the values.
            constrained by 'row'.
        splitby : str / np.array
            The values used to split and color the ridgeplot.
            Uses :meth:`_Assay.get_attribute` to retrieve
            the values. The shape must be `(#cells)`.
                1. None
                    The ridgeplot is not colored.
                2. Any 1D attribute
                    In case 'label' is provided then the
                    stored paltte is used. A discrete color map
                    is used for all other cases.
        features : list-like
            In the case that `attribute` is a layer then
            features sets the subset of features to plot.
            Otherwise it sets the name of each feature shown.
        title : str
            The title given to the plot after
            the name of the assay.

        Returns
        -------
        fig: plotly.graph_objects.Figure
        """

        data = self.get_attribute(attribute, constraint='row', features=features)
        features = data.columns.values
        name = f'{data.index.name}'

        layout = dict(xaxis=dict(zeroline=False, showgrid=False),
                      title=f'{self.title} {name} {title}',
                      legend_tracegroupgap=0,
                      dragmode='lasso',
                      hovermode='closest',
                      template='simple_white')

        if splitby is None:
            data = data.melt(var_name=ID)
            fig = go.Figure(layout=layout)

            for i, df_i in data.groupby(ID):
                fig.add_trace(go.Violin(y=df_i[ID],
                                        x=df_i['value'],
                                        scalemode='width',
                                        scalegroup=i,
                                        box_visible=True,
                                        box_width=0.05,
                                        spanmode='soft',
                                        orientation='h',
                                        side='positive',
                                        width=3,
                                        points=False,
                                        line_color=COLORS[0],
                                        line_width=1,
                                        showlegend=False))

            fig.update_layout(xaxis_showgrid=False,
                              xaxis_zeroline=False,
                              xaxis_title=name,
                              yaxis_categoryarray=features,
                              width=800,
                              height=max(600, len(features) * 50),)
        else:
            nplots = data.shape[1]
            nrows = round(nplots ** 0.5)
            ncols = nplots // nrows + min(1, nplots % nrows)

            splitby = self.get_attribute(splitby, constraint='row')
            if splitby.shape[1] != 1:
                raise ValueError('Only one dimensional `splitby` values are permitted.')

            data.loc[:, LABEL] = splitby.values
            data = data.melt(id_vars=LABEL, var_name=ID)
            un_labs = np.unique(data[LABEL])

            line_colors = {}
            for i in range(len(un_labs)):
                line_colors[un_labs[i]] = self.get_palette()[un_labs[i]] if splitby.index.name == LABEL else COLORS[i]

            fig = make_subplots(rows=nrows,
                                cols=ncols,
                                subplot_titles=features,
                                horizontal_spacing=0.1 / ncols,
                                vertical_spacing=0.2 / nrows)

            for grp, df in data.groupby([LABEL, ID]):
                lab, i = grp
                showlegend = features[0] == i
                row_num = np.where(features == i)[0][0] // ncols + 1
                col_num = np.where(features == i)[0][0] % ncols + 1

                fig.add_trace(go.Violin(y=df[LABEL],
                                        x=df['value'],
                                        scalemode='width',
                                        spanmode='soft',
                                        legendgroup=str(lab),
                                        name=str(lab),
                                        line_color=line_colors[lab],
                                        orientation='h',
                                        side='positive',
                                        width=3,
                                        points=False,
                                        scalegroup=i,
                                        showlegend=showlegend,
                                        line_width=1),
                              row=row_num, col=col_num)

            fig.update_layout(layout,
                              width=max(500, 300 * ncols),
                              height=max(500, max(300 * nrows, 30 * len(un_labs) * nrows)))
            fig.update_yaxes(tickvals=un_labs,
                             ticktext=[''] * len(un_labs),
                             showgrid=True,
                             type='category')

        return fig

    def stripplot(self, attribute, colorby=None, features=None, title=''):
        """
        Strip plot for all barcodes.

        Generates a strip plot for the given ids (x-axis), and
        shows the layer (y-axis) and color_layer (color) for all barcodes.
        Additional lateral jitter is added to help visualize the distribution.

        Parameters
        ----------
        attribute : str / np.array
            The attribute used to plot the values on the ridgeplot.
            Uses :meth:`_Assay.get_attribute` to retrieve the values.
            constrained by 'row'.
        colorby : str / np.array
            The values used to color the stripplot.
            Uses :meth:`_Assay.get_attribute` to retrieve
            the values constrained by 'row'. It must be the
            same shape as attribute.
        features : list-like
            In the case that `attribute` is a layer then
            features sets the subset of features to plot.
            Otherwise it sets the name of each feature shown.
        title : str
            The title given to the plot after
            the name of the assay.

        Returns
        -------
        fig: plotly.graph_objects.Figure
        """

        attribute = self.get_attribute(attribute, constraint='row', features=features)

        if colorby is None:
            colorby = pd.DataFrame(np.full_like(attribute, COLORS[0], dtype='<U10'), columns=attribute.columns)
            colorby.index.name = '----'
        else:
            colorby = self.get_attribute(colorby, constraint='row', features=features)

        if colorby.shape != attribute.shape:
            raise ValueError(f'The shape of `colorby` {colorby.shape} must be the same as the shape of `attribute` {attribute.shape}')

        nb = attribute.shape[0]
        nv = attribute.shape[1]
        nn = nv * nb

        x = np.array([[i for i in range(nv)] for j in range(nb)])
        x = x.reshape(nn, 1) + np.random.rand(nn, 1) * 0.8 - 0.4
        y = attribute.values.reshape(nn,)
        c = colorby.values.reshape(nn,)
        x = x[:, 0]

        # Plotly version. Slow to load but interactive.
        scat = go.Scattergl(x=x,
                            y=y,
                            mode='markers',
                            marker=dict(size=1.5,
                                        color=c,
                                        colorscale='rainbow'))

        if colorby.index.name != '----':
            scat.marker.colorbar = dict(title=colorby.index.name, ticks='inside')

        xax = dict(linecolor='black',
                   linewidth=1,
                   mirror=True,
                   ticks='inside',
                   tickfont={'size': 9})

        layout = dict(xaxis=xax,
                      yaxis=xax,
                      plot_bgcolor='rgba(0,0,0,0)',
                      width=900,
                      height=600)

        fig = go.Figure(data=scat, layout=layout)

        fig.update_layout(xaxis_title='',
                          yaxis_title=attribute.index.name,
                          title=self.title)

        fig.update_xaxes(ticks='outside',
                         range=[-1, nv],
                         tickvals=np.arange(nv),
                         ticktext=attribute.columns,
                         tickangle=-90)

        return fig

    @require_seaborn
    def read_distribution(self, title='', **kwargs):
        """
        Rank-ordered barcodes vs total reads.

        Parameters
        ----------
        title : str
            The suffix to the name of the assay to be
            added in the plot title.
        kwargs : dict
            Passed to seaborn.

        Returns
        -------
        ax : matplotlib.pyploy.axis
        """
        read_counts = self.layers[READS].sum(axis=1)
        read_counts = np.sort(read_counts)[::-1]

        sns.set(style='whitegrid', font_scale=1.5)
        plt.figure(figsize=(10, 10))

        rank = np.arange(1, len(self.barcodes()) + 1)
        ax = sns.lineplot(x=rank, y=read_counts, ci=None, **kwargs)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Rank-ordered barcodes')
        ax.set_ylabel('Number of reads')
        ax.set_title(self.title, fontsize=19)

        return ax

    def feature_scatter(self, layer, ids, **kwargs):
        """
        Plot for given data across 2 ids.

        This is the typical biaxial plot seen in
        protein expression analysis.

        Parameters
        ----------
        layer : str / np.array
            The layer to use for the values
            that are plotted on the biaxial plot.
            Uses :meth:`_Assay.get_attribute` to retrieve
            the value constrained by `row+col`.
        ids : 2-tuple
            The x and y coordinates.
        kwargs : dict
            Passed to :meth:`missionbio.mosaic.assay._Assay.scatterplot`

        Returns
        -------
        fig : plotly.graph_objs.figure
        """

        id_ind = [np.where(self.ids() == i)[0][0] for i in ids]
        data = self.get_attribute(layer, constraint='row+col').values
        data = data[:, id_ind]

        fig = self.scatterplot(attribute=data, **kwargs)
        for i in range(len(fig.data)):
            ax = fig.data[i].xaxis[1:]
            if fig.layout[f'xaxis{ax}'].title.text:
                fig.layout[f'xaxis{ax}'].title.text = ids[0]

            ax = fig.data[i].yaxis[1:]
            if fig.layout[f'yaxis{ax}'].title.text:
                fig.layout[f'yaxis{ax}'].title.text = ids[1]

        return fig

    def fishplot(self, sample_order=None, label_order=None):
        """
        Draws a fish plot

        Parameters
        ----------
        sample_order : list-like
            The order in which the samples
            are to be ordered. If None, then
            an arbitrary order is drawn. A subset
            of the samples can also be provided

        label_order : list-like
            The order in which the labels
            are to be ordered. If None, then
            an arbitrary order is drawn. A subset
            of the labels can also be provided

        Returns
        -------
        fig : plotly.graph_objs.figure

        Raises
        --------
        RuntimeError
            If the assay object has only one sample

        ValueError
            If one of the given samples or labels is
            not found in the assay
        """

        # Pre-flight checks
        samples = self.row_attrs[SAMPLE].copy()
        labels = self.get_labels()
        palette = self.get_palette()

        if sample_order is None:
            sample_ids = list(set(samples))
        else:
            sample_ids = sample_order
            for s in sample_order:
                if s not in samples:
                    raise ValueError(f'Sample "{s}" was not found in the assay.')

        if len(set(sample_ids)) == 1:
            raise RuntimeError('Only one sample found in the assay. Multiple samples required to draw a fish plot.')

        if label_order is None:
            label_ids = list(set(labels))
        else:
            label_ids = label_order[::-1]
            for lab in label_order:
                if lab not in labels:
                    raise ValueError(f'Label "{lab}" was not found in the assay.')

        # Methods to draw the curves
        def _get_first_svg(prop_start, prop1, prop2):
            start = 0.8
            end = 1
            mid = (end + start) / 2

            svg = f'M {start},{prop_start} '
            svg += f'C {mid},{prop_start} {mid},{prop1} {end},{prop1} '
            svg += f'L {end},{prop2} '
            svg += f'C {mid},{prop2} {mid},{prop_start} {start},{prop_start} '

            return svg

        def _get_inner_svg(sample, start, average, end):
            svg = f'M {sample[0]},{start[0]} '
            svg += f'C {sample[1]},{start[0]} {sample[1]},{end[0]} {sample[2]},{end[0]} '
            svg += f'L {sample[2]},{end[1]} '
            svg += f'C {sample[1]},{end[1]} {sample[1]},{start[1]} {sample[0]},{start[1]} '
            svg += 'Z'

            return svg

        def _get_shape_object(svg, color):
            shape = {
                'type': 'path',
                'path': svg,
                'fillcolor': color,
                'line_color': color,
                'line_width': 0.5
            }

            return shape

        def _get_shapes(props):
            shapes = []

            props = np.array(props)
            props = props.cumsum(axis=1)
            props = np.append(np.zeros(props.shape[0])[:, None], props, axis=1)

            for s in range(props.shape[0] - 1):
                for c in range(len(props[s]) - 1):
                    if s == 0:
                        mid = (props[s][c] + props[s][c + 1]) / 2
                        svg = _get_first_svg(mid, props[s][c], props[s][c + 1])

                        shape = _get_shape_object(svg, palette[label_ids[c]])
                        shapes.append(shape)

                    sample = [s + 1, s + 1.5, s + 2]  # First sample, average, and second sample
                    start = [props[s][c], props[s][c + 1]]
                    average = [(props[s][c] + props[s + 1][c]) / 2, (props[s][c + 1] + props[s + 1][c + 1]) / 2]
                    end = [props[s + 1][c], props[s + 1][c + 1]]

                    svg = _get_inner_svg(sample, start, average, end)
                    shape = _get_shape_object(svg, palette[label_ids[c]])
                    shapes.append(shape)

            return shapes

        proportions = np.zeros((len(sample_ids), len(label_ids)))
        proportions = pd.DataFrame(proportions, columns=label_ids, index=sample_ids)
        for s in sample_ids:
            un, cnts = np.unique(labels[samples == s], return_counts=True)
            for i in range(len(un)):
                proportions.loc[s, un[i]] = cnts[i]

        proportions = proportions[label_ids]
        proportions = 100 * proportions / proportions.sum(axis=1)[:, None]
        fig = go.Figure()

        # Update axes properties
        fig.update_xaxes(
            range=[0.6, len(sample_ids)],
            zeroline=False,
            showgrid=False,
            fixedrange=True,
            tickvals=np.arange(1, len(sample_ids) + 1),
            ticktext=sample_ids
        )

        fig.update_yaxes(
            range=[0, 100],
            zeroline=False,
            showgrid=False,
            fixedrange=True,
            tickvals=[],
            ticktext=[],
            title_text='Cluster Prevalance',
            linecolor='darkslategrey'
        )

        # Add shapes
        fig.update_layout(shapes=_get_shapes(proportions),
                          width=1000,
                          height=500,
                          plot_bgcolor='white',
                          showlegend=True,
                          dragmode='select',
                          legend=dict(itemclick=False,
                                      itemdoubleclick=False))

        # Vertical lines at each sample
        for i in range(1, len(sample_ids) + 1):
            fig.add_vline(x=i, line_width=1, line_color='darkslategrey')

        # Legend names
        for lab in label_ids[::-1]:
            data = go.Scatter(x=[-1], y=[50], name=lab, mode='markers', marker_color=palette[lab])
            fig.add_trace(data)

        return fig

    def barplot(self, sample_order=None, label_order=None, percentage=False):
        """
        Draws a bar plot of label counts for each sample

        Parameters
        ----------
        sample_order : list-like
            The order in which the samples
            are to be ordered. If None, then
            an arbitrary order is drawn. A subset
            of the samples can also be provided

        label_order : list-like
            The order in which the labels
            are to be ordered. If None, then
            an arbitrary order is drawn. A subset
            of the labels can also be provided

        percentage : bool
            Whether the proportion should be shown
            as a percentage of total cells or the
            count of the cells

        Returns
        -------
        fig : plotly.graph_objs.figure

        Raises
        --------
        RuntimeError
            If the assay object has only one sample

        ValueError
            If one of the given samples or labels is
            not found in the assay
        """

        # Pre-flight checks
        samples = self.row_attrs[SAMPLE].copy()
        labels = self.get_labels()
        palette = self.get_palette()

        if sample_order is None:
            sample_ids = list(set(samples))
        else:
            sample_ids = sample_order[::-1]
            for s in sample_order:
                if s not in samples:
                    raise ValueError(f'Sample "{s}" was not found in the assay.')

        if label_order is None:
            label_ids = list(set(labels))
        else:
            label_ids = label_order
            for lab in label_order:
                if lab not in labels:
                    raise ValueError(f'Label "{lab}" was not found in the assay.')

        proportions = np.zeros((len(sample_ids), len(label_ids)))
        proportions = pd.DataFrame(proportions, columns=label_ids, index=sample_ids)
        for s in sample_ids:
            un, cnts = np.unique(labels[samples == s], return_counts=True)
            for i in range(len(un)):
                proportions.loc[s, un[i]] = cnts[i]

        proportions = proportions[label_ids]

        if percentage:
            proportions = 100 * proportions / proportions.sum(axis=1)[:, None]

        proportions.loc[:, 'Sample'] = proportions.index

        proportions = proportions.melt('Sample', value_name='Counts', var_name='Label')
        fig = px.bar(proportions, x='Counts', y='Sample',
                     orientation='h',
                     template='gridon',
                     color='Label',
                     color_discrete_map=palette)

        fig.update_layout(xaxis_title_text='',
                          yaxis_title_text='',
                          xaxis_fixedrange=True,
                          yaxis_fixedrange=True,
                          yaxis_automargin=True)

        for data in fig.data:
            if percentage:
                data.hovertemplate = f'<b>%{{x:.2f}}%</b><extra>{data.legendgroup}</extra>'
            else:
                data.hovertemplate = f'<b>%{{x}}</b><extra>{data.legendgroup}</extra>'

        return fig

    def _brush(self, trace, points, state):
        """
        Scatter plot custom clustering.

        The selection of the points on a scatter plot
        created using plotly.

        This method is passed to the on_selection
        method of the FigureWidget object in plotly.
        It is not meant to be used as-is.
        """
        inds = np.array(points.point_inds)

        # Reset values from previous selection
        if points.trace_index == 0:
            # Remove colors which are empty
            dels = []
            for k in self.selected_bars:
                if self.selected_bars[k].size == 0:
                    dels.append(k)

            for d in dels:
                del self.selected_bars[d]

            # Add new color if not there
            # Store new color as a variable to use on other traces
            for c in COLORS:
                if c not in self.selected_bars.keys():
                    self.selected_bars[c] = np.array([])
                    self.__brush_color = c
                    break

        if inds.size:
            bars = np.array([i.split('<br>')[-1] for i in trace.customdata.flatten()])[inds]
            # Remove bars from the color in which they are present
            # Add bars to the new color
            for k in self.selected_bars:
                if k != self.__brush_color:
                    shift = np.isin(self.selected_bars[k], bars)
                    self.selected_bars[self.__brush_color] = np.append(self.selected_bars[self.__brush_color], self.selected_bars[k][shift])
                    self.selected_bars[k] = self.selected_bars[k][~shift]

            # Change the color of the bars which are moved
            new_colors = trace.marker.color.copy()
            new_colors[inds] = self.__brush_color
            trace.marker.color = new_colors

    def _heatmap_selection(self, trace, points, selector):
        """
        Heatmap id selection.

        The selection of the ids on the heatmap
        created using plotly.

        This method is passed to the on_click
        method of the FigureWidget object in
        plotly. It is not meant to be used as-is.
        """

        def switch_color(text):
            style_open = "<span style='color:red'>"
            style_close = "</span>"
            if style_open in text:
                t = text.replace(style_open, '').replace(style_close, '')
            else:
                t = style_open + str(text) + style_close

            return t

        if len(points.xs) != 0:
            click_point = points.xs[0]
            click_pos = np.where(trace.x == click_point)[0][0]
            ticks = self.__heatmap.layout.xaxis.ticktext.copy()
            hovertext = trace.x.copy()

            # Add tick to selected ids
            old_id = ticks[click_pos]
            new_id = switch_color(old_id)
            ticks[click_pos] = new_id
            if len(self.selected_ids) > 0 and new_id in self.selected_ids:
                self.selected_ids = np.delete(self.selected_ids, np.where(self.selected_ids == new_id))
            else:
                self.selected_ids = np.append(self.selected_ids, old_id)

            # Update color of tick text
            if set(self.__heatmap.layout.xaxis2.ticktext) == set(self.__heatmap.layout.xaxis.ticktext):
                self.__heatmap.update_layout(xaxis2_ticktext=ticks)
            self.__heatmap.update_layout(xaxis_ticktext=ticks)

            # Update color of hover text
            old_hovertext = hovertext[click_pos]
            new_hovertext = switch_color(old_hovertext)
            hovertext[click_pos] = new_hovertext

            trace.x = hovertext

    # --------- Statistics

    def feature_signature(self, layer, exclude_missing=False):
        """
        T-Test on given data.

        Generate feature signatures for each cluster/feature pair across all barcodes
        using the supplied assay and layer. For each cluster/feature, returns median feature,
        standard deviation of feature, p-value, and the t-statistic of the feature in the cluster
        compared to all other clusters.

        Parameters
        ----------
        layer : string
            Layer name to be evaluated.
        exclude_missing : bool
            Whether missing dna data is considered.

        Returns
        -------
        med : dataframe
            The median value of the layer for each cluster/feature pair.
        stdev : dataframe
            The standard deviation of the layer for each cluster/feature pair.
        pval : dataframe
            The feature p-value of for each cluster vs other clusters.
        tstat : dataframe
            The value of the t-statistic corresponding to the pval.
        """

        dat = deepcopy(self.layers[layer])
        features = self.ids()
        lbl = self.get_labels()
        u, uind = np.unique(lbl, return_inverse=True)
        nc = len(u)
        nf = len(features)

        med = pd.DataFrame(data=np.zeros((nf, nc)), index=features, columns=u)
        stdev = pd.DataFrame(data=np.zeros((nf, nc)), index=features, columns=u)
        pval = pd.DataFrame(data=np.zeros((nf, nc)), index=features, columns=u)
        tstat = pd.DataFrame(data=np.zeros((nf, nc)), index=features, columns=u)

        for i in range(nc):
            for j in range(nf):
                ind = (uind == i)
                nind = np.invert(ind)
                if (exclude_missing is True) & (self.name == 'dna'):
                    ind = ind & (self.layers['NGT'][:, j] < 3)
                    nind = nind & (self.layers['NGT'][:, j] < 3)

                if np.sum(ind) == 0:
                    med.iloc[j, i] = np.nan
                    pval.iloc[j, i] = 1
                else:
                    med.iloc[j, i] = np.median(dat[ind, j], axis=0)
                    stdev.iloc[j, i] = np.std(dat[ind, j], axis=0)
                    if np.sum(ind) > 1:
                        tstat.iloc[j, i], pval.iloc[j, i] = stats.ttest_ind(dat[ind, j], dat[nind, j], axis=0)
                    else:
                        pval.iloc[j, i] = 1

        return med, stdev, pval, tstat

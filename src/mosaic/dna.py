import warnings, logging
from copy import deepcopy

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import requests  # For pulling annotations from APIs
from h5.constants import AF, DP, GQ, ID, NGT
from plotly.subplots import make_subplots
from scipy import stats  # For z-score function
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform  # For hierarchical clustering

from mosaic.assay import _Assay
from mosaic.constants import AF_MISSING, COLORS, NGT_FILTERED, UMAP_LABEL
from mosaic.plotting import plt, require_seaborn, sns, to_hex
from mosaic.utils import extend_docs


@extend_docs
class Dna(_Assay):
    """
    Container for DNA data.

    Inherits most methods from :class:`mosaic.assay._Assay`.
    See that for the documentation on other methods and visualizations.

    .. rubric:: Algorithms
    .. autosummary::

       get_annotations
       filter_variants
       filter_variants_consecutive
       filter_incomplete_barcodes
       find_relevant_variants
       find_clones
       count
       cluster_cleanup

    .. rubric:: DNA specific visualizations
    .. autosummary::

       plot_variant_correlation
       update_coloraxis

    .. rubric:: Extended methods
    .. autosummary::

        run_umap
        heatmap
        scatterplot
    """

    assay_color = COLORS[0]

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        args: list
            To be passed to mosaic.assay object.

        kwargs: dict
            To be passed to mosaic.assay object.
        """

        super().__init__(*args, **kwargs)

        vaf = self.layers[AF].copy()
        vaf[self.layers[DP] == 0] = -50
        self.add_layer(AF_MISSING, vaf)

    def sort_ids(self):
        """
        Sort variants alphanumerically

        Sorts ids so that 10 comes after 3 rather than before
        """
        chrom = self.col_attrs['CHROM']
        pos = self.col_attrs['POS']
        ind = np.argsort([c.rjust(20) + '%12.0f' % p for c, p in zip(chrom, pos)])
        self.select_columns(ind)

    def find_clones(self, similarity=0.8, **kwargs):
        """
        Identify clones based on VAF.

        Labels the barcodes based on a UMAP-dbscan clustering,
        and then merges similar clusters. Adds a row_attrs `id`.

        Parameters
        ----------
        similarity : float [0, 1]
            The proportion of variants that must be similar
            so as to combine multiple clusters into one
            or to identify mixed cell line populations.

        Warns
        -----
        A warning is raised when the pre-existing "umap" row
        attribute is used to cluster instead of creating a
        new projection.
        """

        if UMAP_LABEL in self.row_attrs:
            warnings.warn(f'Using the "{UMAP_LABEL}" that is already present in the row attributes.')
        else:
            self.run_umap(attribute=AF, **kwargs)

        self.cluster(attribute=UMAP_LABEL, method='dbscan')

        self.cluster_cleanup(layer=AF, similarity=similarity)

    def cluster_cleanup(self, layer, similarity=0.8):
        """
        Merge similar clusters while ignoring missing data.

        Parameters
        ----------
        layer : str
            The layer with the VAF data.
        similarity : float [0, 1]
            The proportion of variants that must be similar
            so as to combine multiple clusters into one cluster.
        """

        vaf = pd.DataFrame(self.layers[layer], index=self.barcodes(), columns=self.ids())
        vaf = vaf / 100

        # Identifying cluster characteristics
        # i.e., mean VAF is closest to 0, 0.25, 0.5, 0.75, or 1
        vaf.loc[:, 'label'] = self.get_labels()

        cluster_char = pd.DataFrame(columns=vaf.columns)
        for lab, vaf_lab in vaf.groupby('label'):
            cluster_char.loc[lab, :] = np.round(vaf_lab.mean() * 4) / 4

        cluster_char = cluster_char.drop('label', axis=1)

        clusters = np.array(cluster_char.index)
        print(f'Unique clusters found - {len(clusters)}')

        # Removing clusters caused due to missing values
        cluster_char_pos = cluster_char.loc[:, (cluster_char >= 0).all()]
        renamed_clusters = {}

        for i in range(len(clusters)):
            if clusters[i] not in renamed_clusters:
                renamed_clusters[clusters[i]] = clusters[i]

            for j in range(i + 1, len(clusters)):
                char1 = cluster_char_pos.iloc[i, :]
                char2 = cluster_char_pos.iloc[j, :]
                simi = (char1 == char2).mean()
                if simi >= similarity:
                    renamed_clusters[clusters[j]] = renamed_clusters[clusters[i]]

        clusters = np.unique(list(renamed_clusters.values()))

        print(f'Clusters after removing missing data - {len(clusters)}')

        # Renaming labels
        newlabs = np.array([renamed_clusters[lab] for lab in self.get_labels()])
        labels, idx, cnt = np.unique(newlabs, return_inverse=True, return_counts=True)
        labels[cnt.argsort()[::-1]] = np.arange(len(labels)) + 1
        labels = labels[idx]

        self.set_labels(labels)

    def genotype_variants(self, het_vaf=20, hom_vaf=80, min_dp=None, min_alt_read = None, min_gq = -1, assign_low_conf_genotype=False):

        '''
        @HZ: Mission Bio's method by default seems to already construct the NGT matrix based on their default filtering thresholds (for values see here: https://github.io/mosaic/pages/methods/mosaic.dna.Dna.filter_variants.html#mosaic.dna.Dna.filter_variants)
        
        This function aims to build the 'NGT' [genotype (0: WT; 1: HET; 2: HOM; 3: MISSING)] matrix based on custom thresholds:

        Parameters
        ----------
        het_vaf : float [0, 100]
            minimum vaf for a variant in one cell to be called heterozygously mutated
            default: 20
        hom_vaf : float [0, 100]
            minimum vaf for a variant in one cell to be called homozygously mutated
        min_dp : int [0, inf]
            minimum depth for a variant to be considered covered in one cell
        min_alt_read : int [0, inf]
            minimum alternative read count for a variant to be considered real in one cell (otherwise WT)
        min_gq : int [0, 99]
            minimum genotype quality (HaplotypeCaller) for a variant to be considered real in one cell (otherwise WT)
        assign_low_conf_genotype : bool
            if True, assign a separate number (`4`) to low-confidence genotypes (i.e., 0: WT; 1: HET; 2: HOM; 3: MISSING; 4: low-confidence)
        '''
        try:
            ngt = self.layers[NGT]
        except KeyError:
            logging.warning('original NGT layer not found.')

        # self.del_layer('NGT')

        dp = self.layers[DP]
        vaf = self.layers[AF]
        try:
            gq = self.layers[GQ]
        except KeyError:
            logging.warning('GQ layer not found. Using all zeros.')
            gq = np.zeros_like(dp)

        # by default:
        # min_dp = mean - 1.5 * std;
        # min_alt_read = 0.4 * min_dp 
        if min_dp is None:
            min_dp = dp.mean() - 1.5 * dp.mean(axis=0).std()
            logging.warning(f'min_dp not given; using default setting for min_dp, value= {min_dp}.')
        else:
            logging.info(f'min_dp given, value= {min_dp}.')

        if min_alt_read is None:
            min_alt_read = 0.4 * min_dp
            logging.warning(f'min_alt_read not given; using default setting for min_alt_read, value= {min_alt_read}.')
        else:
            logging.info(f'min_alt_read given, value= {min_alt_read}.')
        if not 'alt_read_count' in self.layers:   
            # calculate alternative read count
            alt = (np.rint(np.multiply(vaf, dp)/100)).astype(int)
            self.add_layer('alt_read_count', alt)
            logging.warning(f'`alt_read_count` not found in layers; calculated based on DP and VAF and added to layers.')
        else:
            alt = self.layers['alt_read_count']

        # @HZ 07/18/2022: we might want to differentiate between low-confidence and real homdel
        #gt = (vaf > het_vaf) + (vaf > hom_vaf)
        ngt_unfiltered = np.full_like(vaf, 0) + (alt > 0) * (gq >= min_gq) * ((vaf > het_vaf)*1 + (vaf > hom_vaf)*1)
        ngt_unfiltered = np.where(dp < 1, 3, ngt_unfiltered) # convert SNVs with strictly 0 read to homdel ('3')

        if assign_low_conf_genotype:
            ngt_filtered = np.where( (alt > 0) & (((dp > 0) & (dp < min_dp)) | (alt < min_alt_read)) , 4, ngt_unfiltered) # convert SNVs with low depth/low alt-read to low-confidence mutant calls ('4')
        else:
            ngt_filtered = np.where( (alt > 0) & (((dp > 0) & (dp < min_dp)) | (alt < min_alt_read)) , 0, ngt_unfiltered) # convert SNVs with low depth/low alt-read to WT ('0')

        self.add_layer('NGT_unfiltered', ngt_unfiltered)
        logging.info('added layer `NGT_unfiltered` ')
        self.add_layer('NGT', ngt_filtered)
        logging.info('added layer `NGT` ') # switched from NGT_filtered to NGT for use in filter_variants()

        mut = ((ngt_filtered %3 != 0)).astype(int) # for the 'mut' layer, we only want to keep the variants that are not WT or missing. So low-confidence calls are included as 'mut' as well.
        self.add_layer('mut_unfiltered', mut)
        logging.info('added layer `mut_unfiltered` ')

        mut_filtered = ((ngt_filtered == 1) | (ngt_filtered == 2)).astype(int)
        self.add_layer('mut_filtered', mut_filtered) # if assign_low_conf_genotype is False, this layer is the same as 'mut_unfiltered'


    # @HZ 09/19/2022: deprecated with the new genotype_variants function
    def filter_variants(self, min_dp=10, min_gq=0, min_vaf=20, max_vaf=100, min_prct_cells=25, min_mut_prct_cells=0.5, min_mut_num_cells=None, min_std=0, method='mb', min_alt_read = 5):
        """
        Find informative variants.

        This method also adds the `NGT_FILTERED` layer to the assay
        which is a copy of the NGT layer but with the NGT for the
        cell-variants not passing the filters set to 3 i.e. missing.

        Parameters
        ----------
        min_dp : int
            The minimum depth (DP) for the call to be considered.
            Variants with less than this DP in a given
            barcode are treated as no calls.
        min_gq : int
            The minimum genotype quality (GQ) for the call to be
            considered. Variants with less than this GQ
            in a given barcode are treated as no calls.
        min_vaf : float [0, 100]
            If the VAF of a given variant for a given barcode
            is less than the given value, and the call is
            '1' i.e. HET, then it is converted to a no call.
        max_vaf : float [0, 100]
            If the VAF of a given variant for a given barcode
            is greater than the given value, and the call is
            '1' i.e. HET, then it is converted to a no call.
        min_prct_cells : float [0, 100]
            The minimum percent of total cells in which the variant
            should be called as '0', '1', or '2' after the
            filters are applied.
        min_mut_prct_cells : float [0, 100]
            The minimum percent of the total cells in which the
            variant should be mutated, i.e., either '1' or '2'
            after the filters are applied.
        min_mut_num_cells : integer [0, self.shape[1]]
            The minimum NUMBER of cells in which the variant should be mutated. Default to none, only applicable when min_mut_prct_cells is None
        min_std : float [0, 100]
            The standard deviation of the VAF across the cells
            of the variants should be greater than
            this value.

        @HZ
        method : 'mb' or 'hz'
            if 'mb': the above logic is used.
            if 'hz': min_prct_cells filter is calculated based on min_dp only 
        
        min_alt_read: int [0, inf]
            minimum count of alternative read for the mutation to be considered

        Returns
        -------
        numpy.ndarray
        """
        
        gt = self.layers[NGT]
        dp = self.layers[DP]
        gq = self.layers[GQ]
        vaf = self.layers[AF]

        # @ HZ: filter on alternative reads absolute value
        if min_alt_read > 0 and 'alt_read_count' in self.layers:
            alt = self.layers['alt_read_count']
            alt_keep = alt >= min_alt_read
        elif min_alt_read > 0 and 'alt_read_count' not in self.layers:
            print('alt_read_count not calculated, calculate now')
            alt = (np.multiply(vaf, dp)/100).astype(int)
            self.add_layer('alt_read_count', alt)
            alt_keep = alt >= min_alt_read
        else:
            alt_keep = 1

        dp_keep = dp >= min_dp
        gq_keep = gq >= min_gq
        min_vaf_keep = ~np.logical_and(vaf < min_vaf, gt == 1)
        max_vaf_keep = ~np.logical_and(vaf > max_vaf, gt == 1)
        gt = (gt - 3) * dp_keep * gq_keep * min_vaf_keep * max_vaf_keep * alt_keep + 3  # workaround to apply filter in one line
        # ^^^    
        # @HZ: this is dangerous since this will only trim down variants already filtered by the default thresholds

        if 'NGT_FILTERED' not in self.layers.keys():
            print('[filter_variants] NGT_filtered layer not present. Adding NGT_FILTERED layer.')
        else:
            print('[filter_variants] NGT_filtered layer already present. Overwriting NGT_FILTERED layer.')
        self.add_layer(NGT_FILTERED, gt)

        num_cells = len(self.barcodes())
        
        ##############################################################
        # @HZ: different way of filtering based on read depth per cell
        if method == 'mb':
            min_cells_filter = np.isin(gt, [0, 1, 2]).sum(axis=0) > num_cells * min_prct_cells / 100
        elif method == 'hz':
            min_cells_filter = dp_keep.sum(axis=0) > num_cells * min_prct_cells / 100
        else:
            print("method should be either 'mb' or 'hz' ")
            raise NotImplementedError
        ########################################

        if min_mut_num_cells is not None:
            if min_mut_prct_cells is not None:
                print("only one of [min_mut_prct_cells] and [min_mut_num_cells] should be input ")
                raise NotImplementedError
            elif not (0 <= min_mut_num_cells < num_cells):
                print("[min_mut_num_cells] should be greater than or equal to zero and smaller than the total number of cells in the sample")
                raise ValueError

            else:
                min_cells_mut_filter = np.isin(gt, [1, 2]).sum(axis=0) > min_mut_num_cells
        else:
            min_cells_mut_filter = np.isin(gt, [1, 2]).sum(axis=0) > round(num_cells * min_mut_prct_cells / 100)

        good_variants = min_cells_mut_filter * min_cells_filter

        final_filter = (vaf.std(axis=0) >= min_std) * good_variants

        # @HZ: add reason for exclusion as a layer "filter_info" to each variant-cell pair
        
        # dp_fil = np.char.array(np.where(~dp_keep, 'dp', ''))
        # gq_fil = np.char.array(np.where(~gq_keep, 'gq', ''))
        # min_vaf_fil = np.char.array(np.where(~min_vaf_keep, 'min_vaf', ''))
        # max_vaf_fil = np.char.array(np.where(~max_vaf_keep, 'max_vaf', ''))

        # filter_info = dp_fil + ' ' + gq_fil + ' ' + min_vaf_fil + ' ' + max_vaf_fil
        # self.add_layer('filter_info', filter_info)

        # min_cells_fil = np.char.array(np.where(~min_cells_filter, 'min_cells_covered', ''))
        # min_cells_mut_fil = np.char.array(np.where(~min_cells_mut_filter, 'min_cells_mut', ''))
        
        # var_filter_info = min_cells_fil + ' ' + min_cells_mut_fil

        # var_filter_info_dict = {}
        # for variant, info in zip(self.col_attrs[ID], var_filter_info):
        #     var_filter_info_dict[variant] = info

        return self.col_attrs[ID][final_filter].astype(str)

    def filter_variants_consecutive(self, proximity=[25, 50, 100, 200]):
        """
        Remove nearby variants.

        Remove the variants that are close to each other on the same amplicon.
        This is likely primer misalignment.

        Parameters
        ----------
        proximity : list-like (int)
            If `i + 1` variants are within `proximity[i]`,
            then the variants are removed.

        Returns
        -------
        keep : np.ndarray
            Variants that are to be kept, i.e., variants
            that are close to each other are discarded.

        Notes
        -----
        Assumes the variants are in order.

        Examples
        --------
        - If >1 variants are within 25 bases of each other, remove.
        - If >2 variants are within 50 bases of each other, remove.
        - If >3 variants are within 100 bases of each other, remove.
        - If >4 variants are within 250 bases of each other, remove.
        """

        keep = np.ones(self.shape[1], dtype=int)
        delete = []

        chrom = self.col_attrs['CHROM']
        pos = self.col_attrs['POS']

        print('Variants too close to each other:')

        loc = 0
        while loc < len(chrom):
            found = 0
            fa = np.array(np.where((chrom == chrom[loc]) & (np.arange(len(chrom)) > loc)))[0, ::-1]
            for jj in fa:
                for ii in range(len(proximity)):
                    if (pos[loc] + proximity[ii] > pos[jj]) & (jj - loc > ii):
                        found = 1
                if found == 1:
                    for i in np.arange(loc, jj + 1):
                        print([chrom[i], pos[i]])
                    delete = delete + np.arange(loc, jj + 1).tolist()
                    loc = jj + 1
                    break
            if found == 0:
                loc = loc + 1

        if len(delete) == 0:
            print('None')

        keep[delete] = 0

        keep = np.where(keep)[0]

        return self.ids()[keep]

    def filter_incomplete_barcodes(self, barcode_thresh=0):
        """
        Remove cells with missing information.

        Parameters
        ----------
        barcode_thresh : float
            The fraction of variants with missing genotype below
            which a barcode can be kept.

        Returns
        -------
        keep : np.ndarray
            The list of barcodes with fewer missing variants than the threshold.
        """

        dat = self.layers[NGT]

        keep = (np.sum(dat == 3, axis=1) <= barcode_thresh * dat.shape[1])

        keep = np.where(keep)[0]

        return self.barcodes()[keep]

    def pathscale(self, layer=AF):
        """
        Scale variants based on the type of mutant.

        Adds `vaf_pathscale` to the layers.

        Parameters
        ----------
        layer : str
            The layer of the VAF data.
        """

        data = self.layers[layer]

        variants = self.ids()

        X = stats.zscore(data)

        pathtypes = ['PATH', 'L.PATH', 'NONS', 'MISS']
        pathscales = [2.5, 1.7, 1.4, 1.4]

        for pathtype, pathscale in zip(pathtypes, pathscales):
            patho = [f'({pathtype})' in variant for variant in variants]
            X[:, patho] = X[:, patho] * pathscale

        self.add_layer('vaf_pathscale', X)

    def find_relevant_variants(self, variant_corr_thresh=.15, variant_cluster_thresh=.3):
        """
        Variants that vary across clusters.

        Parameters
        ----------
        variant_corr_thresh : float [0, 1]
            The variant is labeled as relevant if the correlation coeffecient
            with any other variant is greater than this threshold.

        variant_cluster_thresh : float [0, 1]
            The variant is labeled as relevant if the difference in median VAF
            of two clusters is larger than the threshold.

        Returns
        -------
        variant_keep : np.ndarray
            The list of variants that vary across clusters.
        """

        vaf = self.layers[AF] / 100
        cc = np.corrcoef(vaf.T)
        cc[cc > .9999] = 0

        variant_keep = np.max(np.absolute(cc), axis=0) > variant_corr_thresh

        cluster_vaf, _, _, _ = self.feature_signature(layer=AF, exclude_missing=False)

        array_no_missing = deepcopy(cluster_vaf.to_numpy())
        array_no_missing[array_no_missing < 0] = float('inf')
        mx = np.max(cluster_vaf.to_numpy(), axis=1)
        mn = np.min(array_no_missing, axis=1)

        variant_keep = variant_keep & (mx - mn > variant_cluster_thresh)

        if sum(variant_keep) > 0:
            variant_keep = np.where(variant_keep)[0]
            print(f'Keeping {len(variant_keep)} relevant variants.')
        else:
            variant_keep = np.arange(len(variant_keep))
            print('Found no relevant variants. Keeping all.')

        return self.ids()[variant_keep]

    def get_annotations(self):
        """
        Annotate DNA variants.

        Returns an annotated string array based on the ids of the assay (of form chrA-B-C-D), where
        A,B,C,D represent the chromosome, position, reference, and alternate variant values of an arbitrary length.

        The new strings have a pathological type (PATH, L.PATH, MISS, NONS), and protein or gene added to the beginning
        of the string. This information is pulled from the MB annotation API.

        Returns
        -------
        variants : list
            The array of strings with pathogenicity and gene/protein appended to the front.
        """

        variants = self.ids()
        variants = np.array([var.replace(':', '-').replace('/', '-') for var in variants], dtype='object')

        url = 'https://api.io/annotations/v1/variants?ids=' + ','.join(variants.astype(str))
        r = requests.get(url=url)
        vars = r.text.split('chromosome')[1:]
        genes = deepcopy(variants)

        for ii in range(len(vars)):

            vals = vars[ii].split('"')
            p = np.array(np.where(np.isin(vals, ['Protein'])))[0]
            g = np.array(np.where(np.isin(vals, ['Gene'])))[0]
            if len(g) == 0:
                continue

            prot = vals[p[0] + 4]
            gene = vals[g[0] + 4]

            patho = vars[ii].find('Pathogenic') != -1
            lpatho = vars[ii].find('Likely') != -1
            missense = vars[ii].find('missense') != -1
            nonsense = vars[ii].find('nonsense') != -1

            variants[ii] = ('(PATH) ' if patho else '') + \
                           ('(L.PATH) ' if (lpatho & (not patho)) else '') + \
                           ('(MISS) ' if (missense & (not patho) & (not lpatho)) else '') + \
                           ('(NONS) ' if (nonsense & (not patho) & (not lpatho)) else '') + \
                           (gene if (len(prot) == 0) & (len(gene) > 0) else '') + \
                           (prot) + \
                           (' - ' if len(gene) > 0 else '') + variants[ii]

            genes[ii] = gene if len(gene) else variants[ii]

        return variants

    # --------- Tertiary analysis methods

    def run_umap(self, jitter=0.1, **kwargs):
        """
        Extends :meth:`_Assay.run_umap`

        Sets the default values for the dna UMAP.
        n_neighbors=50, metric='euclidean', min_dist=0
        Also adds jitter for the NGT layer.
        """

        defaults = dict(n_neighbors=50,
                        metric='euclidean',
                        min_dist=0)

        for key in defaults:
            if key not in kwargs:
                kwargs[key] = defaults[key]

        if 'attribute' in kwargs and isinstance(kwargs['attribute'], str) and kwargs['attribute'] == 'NGT':
            kwargs['attribute'] = self.layers['NGT'].copy() + np.random.normal(loc=0, scale=jitter, size=self.shape)

        super().run_umap(**kwargs)

    def count(self, features, layer=NGT, group_missing=True, min_clone_size=1, ignore_zygosity=False, show_plot=False):
        """
        A clustering method available only for DNA.

        Labels the cells based on the groups formed by the chosen
        features. The values are stored in the `LABEL` layer.

        The returned dataframe contains information regarding the
        nature of the subclones. False positive clones can
        be obtained due to Allele Dropout (ADO) events.
        It contains three columns, a score, the parent clones,
        and the sister ADO clones. The indices are the
        subclone names.

        In an ADO event HET goes to WT and HOM for a given
        variant in a subset of the cells. Here, the HET clone is
        called the parent clone, the HOM and WT clones are
        the ADO clones, together called the sister clones.

        The parent and sister clones will be `np.nan` if the
        score is zero. Otherwise it is the name of the clone
        from which the subclone was obtained due to an ADO event.

        The score for each subclone measures the possibility that
        it's a flase positive subclone obtained due to an ADO
        event. The score is 0 if it unlikely to be a clone due
        to ADO and 1 if it is highly likely to be an ADO clone.

        The score takes into account the following metrics.
            1. NGT values of the clones
            2. Relative proportions of the clones
            3. Absolute proportions of the clones (uses min_clone_size as a parameter)
            4. Mean GQ of the clones
            5. Mean DP of the clones

        The score is calculated using four sub scores.
            ``score = (ss + ds + gs) * ps``

            1. `ss - sister score (0 - 0.8)`
                It measures the proportion of the clone
                with resepect to its sister clone. This
                score is closer to 0.8 when the sister clones
                have similar proportions and exactly 0.8 when
                their proportions are within the min_clone_size.
            2. `ds - DP score (0 - 0.1)`
                It measures the mean DP of the clone
                with resepect to its parent clone. It is
                closer to 0.1 if the DP of the clone is
                lower than the parents' DP.
            3. `gs - GQ score (0 - 0.1)`
                It measures the mean GQ of the clone
                with resepect to its parent clone. It is
                closer to 0.1 if the GQ of the clone is
                lowert than the parents' GQ.
            4. `ps - parent score (0 - 1)`
                It measures the proportion of the clone with
                respect to the parent clone. This score is closer
                to 1 the larger the parent is compared to the
                clone, and closer to 0 the smaller the parent
                compared to the clone.

        Parameters
        ----------
        features : list-like
            The features which are to be considered while
            allocating the groups formed by the genotype.
        layer : str
            Name of the layer used to count the cell types.
            Expected values are NGT or NGT_FILTERED as obtained
            from the :meth:`Dna.filter_variants` method.
        group_missing : bool
            Whether the clusters caused due to missing values
            are merged together under one cluster named 'Missing'.
        min_clone_size : float [0, 100]
            The minimumum proportion of total cells to be present
            in the clone to count it as a separate clone.
        ignore_zygosity : bool
            Whether HET and HOM are considered the same or not
        show_plot : bool
            Whether a plot showing the ADO identification process
            should be shown or not.

        Returns
        -------
        pd.DataFrame / None
            `None` is returned if `ignore_zygosity` is True or
            `group_missing` is False otherwise a pandas dataframe
            is returned.
        """

        # Checking attributes
        if len(features) == 0:
            raise ValueError("At least on feature is needed to cluster.")

        # Renaming labels based on proportion
        def _sort_labels(labels):
            labels, idx, cnt = np.unique(labels, return_inverse=True, return_counts=True)
            clones = (labels != 'missing') & (labels != 'small')
            labels[cnt[clones].argsort()[::-1]] = np.arange(clones.sum()) + 1
            labels = labels[idx]
            return labels

        # Assigning labels
        gt = self.get_attribute(layer, constraint='row+col', features=features)

        if ignore_zygosity:
            gt[gt == 1] = 2

        un, idx, cnt = np.unique(gt, return_inverse=True, return_counts=True, axis=0)
        labels = np.unique(idx).astype(str)

        if group_missing:
            labels[(un == 3).any(axis=1)] = 'missing'

        labels = _sort_labels(labels[idx])
        ado_labels = labels

        # Small clusters
        labels, idx, cnt = np.unique(labels, return_inverse=True, return_counts=True, axis=0)
        proportion = 100 * cnt / cnt.sum()
        labels[(proportion < min_clone_size) & (labels != 'missing')] = 'small'

        labels = labels[idx]

        self.set_labels(labels)

        # Handling ADOs
        if group_missing and not ignore_zygosity:
            gt.loc[:, 'label'] = ado_labels

            cnts = gt.groupby('label').count().T.iloc[0, :]
            cnts = cnts / cnts.sum()
            if 'missing' in cnts.index:
                cnts = cnts.drop('missing')
            cnts = cnts[np.arange(1, len(cnts) + 1).astype(str)]

            signs = gt.groupby('label').median().T
            signs = signs.loc[:, cnts.index]

            gq = self.get_attribute('GQ', constraint='row+col', features=features)
            gq.loc[:, 'label'] = ado_labels
            gq = gq.groupby('label').mean().T
            gq = gq.loc[:, signs.columns]

            dp = self.get_attribute('DP', constraint='row+col', features=features)
            dp.loc[:, 'label'] = ado_labels
            dp = dp.groupby('label').mean().T
            dp = dp.loc[:, signs.columns]

            # Build database of ADO clones
            ado_data = pd.DataFrame()
            for parent in signs.columns:
                sign = signs.loc[:, parent]
                for var in sign.index:
                    if sign[var] == 1:  # Find ADO subclones
                        ado_clones = []
                        for v in [0, 2]:
                            ado_sign = sign.copy()
                            ado_sign[var] = v
                            ado_present = (signs.T == ado_sign).all(axis=1)
                            if ado_present.any():
                                clone = signs.columns[ado_present][0]
                                ado_clones.append(clone)
                            else:
                                break
                        else:  # If both ADO clones are found
                            for clone, sister in zip(ado_clones, ado_clones[::-1]):
                                n = ado_data.shape[0] + 1
                                gql = 100 * (gq.loc[var, parent] - gq.loc[var, clone]) / gq.loc[var, parent]
                                dpl = 100 * (dp.loc[var, parent] - dp.loc[var, clone]) / dp.loc[var, parent]
                                ado_data.loc[n, 'clone'] = clone
                                ado_data.loc[n, 'parent'] = parent
                                ado_data.loc[n, 'sister'] = sister
                                ado_data.loc[n, 'parent_proportion'] = cnts[parent] * 100
                                ado_data.loc[n, 'clone_proportion'] = cnts[clone] * 100
                                ado_data.loc[n, 'sister_proportion'] = cnts[sister] * 100
                                ado_data.loc[n, 'GQ_loss'] = gql
                                ado_data.loc[n, 'DP_loss'] = dpl

            if not ado_data.empty:
                ado_data = ado_data.set_index(['clone', 'parent']).sort_index()

            # Calculate score
            ado_scores = pd.DataFrame(index=signs.columns, columns=['parents', 'sisters', 'score'])
            ado_scores.index.name = 'clone'
            for clone in ado_scores.index:
                parents, sisters, score = np.nan, np.nan, 0
                if clone in ado_data.index:
                    pclone = ado_data.loc[clone, 'clone_proportion'][0]
                    pparent = max(ado_data.loc[clone, 'parent_proportion'])  # Only the largest parent looked at
                    psis = ado_data.loc[clone, 'sister_proportion'].sum()  # All sisters considered
                    sis = ado_data.loc[clone, 'sister']
                    pcousins = ado_data.loc[sis, :]
                    cousins = pcousins['sister'] != clone
                    pcousins = pcousins[cousins]['sister_proportion'].sum()

                    # Smaller clone must be all ADO - given 0.8 score
                    # Larger clone scored based on its size relative to the smaller one
                    # Minimum permissible error increases with multiple parents
                    # A = a +- error
                    # B = b +- error
                    # A + B = a + b +- 2 * error
                    corrected_psis = max(0, psis - pcousins)
                    extra = pclone - min(pclone, corrected_psis)
                    permitted_error = min_clone_size * (cousins.sum() + len(sis))
                    prop = (extra - permitted_error) / pclone
                    sister_score = np.interp(prop, xp=[0, 1], fp=[0.8, 0])

                    # Give small weightage of score to GQ and DP
                    clone_data = ado_data.loc[(clone, slice(None)), :]
                    clone_data = clone_data.sort_values(by='parent_proportion', ascending=False)
                    gql = clone_data['GQ_loss'].values[0]
                    gq_score = np.interp(gql, xp=[0, 30], fp=[0, 0.1])
                    dpl = clone_data['DP_loss'].values[0]
                    dp_score = np.interp(dpl, xp=[0, 30], fp=[0, 0.1])

                    # The parent size will affect the amount of ADO in the clone
                    # Parent has to be at least as large as the clone for a sufficient score
                    # Signmoid funciton used to score i.e. A parent of the same size has a 0.5 score
                    prop = (pparent - pclone) / min(pclone, pparent)  # (-inf, inf)
                    parent_score = 1 / (1 + np.exp(-prop))

                    score = (sister_score + gq_score + dp_score) * parent_score

                    parents = np.array(ado_data.loc[clone].index)
                    sisters = np.array(ado_data.loc[clone, 'sister'])

                ado_scores.loc[clone, 'parents'] = parents
                ado_scores.loc[clone, 'sisters'] = sisters
                ado_scores.loc[clone, 'score'] = score

            if show_plot:
                h = max(8, 4 * signs.shape[0])
                w = max(10, 0.75 * signs.shape[1])
                sns.set(style='whitegrid')
                fig, axs = plt.subplots(4, 1, figsize=(w, h))

                ax = sns.barplot(x=cnts.index, y=cnts, ax=axs[0], color=COLORS[0], order=cnts.index)
                ax.set_xticklabels([f'{i:.1%}\n{s:.2f}' for i, s in zip(cnts.values, ado_scores.loc[:, 'score'])])
                sns.despine(right=True, left=True)

                cols = sns.cubehelix_palette(3, rot=(-0.2), light=0.3, dark=0.9)
                sns.heatmap(signs, ax=axs[1], yticklabels=True, cbar=False, annot=True,
                            fmt='.0f', linewidths=1, vmax=2, vmin=0, cmap=cols)
                axs[1].set_title('NGT')

                sns.heatmap(gq, ax=axs[2], yticklabels=True, cbar=False, annot=True,
                            fmt='.0f', linewidths=1, vmax=100, vmin=0, cmap='Greens')
                axs[2].set_title('Mean GQ')

                sns.heatmap(dp, ax=axs[3], yticklabels=True, cbar=False, annot=True,
                            fmt='.0f', linewidths=1, vmax=100, vmin=0, cmap='Reds')
                axs[3].set_title('Mean DP')

                for ax in axs:
                    ax.set_xlabel('')
                    ax.set_ylabel('')
                    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

                def connect(a, b, col, parent_num):
                    nonlocal axs, cnts
                    start = np.where(cnts.index == a)[0][0]
                    end = np.where(cnts.index == b)[0][0] + 0.15 * int((1 + parent_num) / 2) * (parent_num % 2)
                    height = cnts[a] * 0.75
                    head = cnts[b]
                    axs[0].hlines(y=height, xmin=start, xmax=end, color=COLORS[20], linewidth=1)
                    axs[0].vlines(x=end, ymax=height, ymin=head, color=col, linewidth=2)

                xlim = axs[0].get_xlim()
                ado_drawn = []
                i = 0
                for c in ado_scores.dropna().index:
                    parents = ado_scores.loc[c, 'parents']
                    sisters = ado_scores.loc[c, 'sisters']
                    for j in range(len(parents)):
                        if {c, parents[j], sisters[j]} not in ado_drawn:
                            ado_drawn.append({c, parents[j], sisters[j]})
                            col = COLORS[i]
                            i += 1
                            if cnts[c] < cnts[parents[j]]:
                                connect(parents[j], c, col, j)
                            if cnts[sisters[j]] < cnts[parents[j]]:
                                connect(parents[j], sisters[j], col, j)

                plt.tight_layout()
                axs[0].set_xlim(xlim)
                axs[0].set_title(self.title)
                axs[0].set_yticks([])

            # Rename small subclones in returned dataframe
            names = np.array([ado_labels, labels]).T
            rename_df = pd.DataFrame(np.unique(names, axis=0), columns=['old', 'new'])
            rename_df.index = rename_df['old']
            if 'missing' in rename_df.index:
                rename_df = rename_df.drop('missing')

            for clone in ado_scores.dropna().index:
                parents = ado_scores.loc[clone, 'parents']
                parents = rename_df.loc[parents]['new'].values
                ado_scores.loc[clone, 'parents'] = parents

                sisters = ado_scores.loc[clone, 'sisters']
                sisters = rename_df.loc[sisters]['new'].values
                ado_scores.loc[clone, 'sisters'] = sisters

            big_clones = rename_df[rename_df['new'] != 'small']['old']
            ado_scores = ado_scores.loc[big_clones, :]
            return ado_scores

    # --------- Plotting

    def heatmap(self, *args, **kwargs):
        """
        Extends :meth:`_Assay.heatmap`

        Set specific colorscales for DNA.

        ----------
        """

        fig = super().heatmap(*args, **kwargs)
        if 'attribute' in kwargs:
            self.update_coloraxis(fig, kwargs['attribute'])
        else:
            self.update_coloraxis(fig, args[0])

        return fig

    def scatterplot(self, *args, **kwargs):
        """
        Extends :meth:`_Assay.scatterplot`

        Set specific colorscales for DNA.

        ----------
        """

        fig = super().scatterplot(*args, **kwargs)

        if 'colorby' in kwargs and isinstance(kwargs['colorby'], str):
            self.update_coloraxis(fig, kwargs['colorby'])
        elif len(args) >= 2:
            self.update_coloraxis(fig, args[1])

        return fig

    def feature_scatter(self, *args, **kwargs):
        """
        Extends :meth:`_Assay.feature_scatter`

        Add jitter for NGT, AF, and AF_MISSING layers.

        ----------
        """

        if 'layer' in kwargs:
            data = kwargs['layer']
        else:
            data = args[0]
            del args[0]

        data = self.get_attribute(data, constraint='row+col')
        name = data.index.name
        data = data.values
        if name == NGT or name == NGT_FILTERED:
            data = data + np.random.normal(loc=0, scale=0.1, size=self.shape)
        elif name == AF or name == AF_MISSING:
            data = data + np.random.normal(loc=0, scale=2, size=self.shape)

        kwargs['layer'] = data

        fig = super().feature_scatter(*args, **kwargs)

        if name == NGT or name == NGT_FILTERED or name == AF or name == AF_MISSING:
            fig.layout.title.text = fig.layout.title.text + '<br><span style="font-size: 10px;">Jitter added.</span>'

        if 'colorby' in kwargs and isinstance(kwargs['colorby'], str):
            self.update_coloraxis(fig, kwargs['colorby'])

        return fig

    def read_distribution(self, *args, **kwargs):
        """
        Unsupported for DNA.
        """

        raise TypeError("The read distribution plot is unsupported for DNA. Try plotting using CNV.")

    def plot_variant_correlation(self):
        """
        The correlation of variants across all cells.

        Generates two plots showing the correlation between the variants across barcodes.
        The first plot is ordered by variant position. The second is hierarchically clustered.

        Returns
        -------
        fig : plotly.grahp_objects.Figure
        """

        vaf = self.layers[AF]
        variants = self.ids()
        cc = np.corrcoef(vaf.T)

        # Pearson's correlation with negative values (missing) removed
        def corr(u, v):

            pos = (u >= 0) & (v >= 0)

            if sum(pos) == 0:
                return 0

            u = np.array(u)[pos].astype(float)
            v = np.array(v)[pos].astype(float)

            def mdot(u, v):
                return np.dot(u - u.mean(), v - v.mean())

            if (mdot(u, u) == 0) | (mdot(v, v) == 0):
                return 0

            return mdot(u, v) / np.sqrt(mdot(u, u) * mdot(v, v))

        cc = squareform(pdist(vaf.T, corr))

        for i in range(cc.shape[0]):
            cc[i, i] = 1

        D = pdist(np.absolute(cc), 'euclidean')
        tree = hierarchy.linkage(D)
        s = hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(tree, D))

        fig = make_subplots(rows=1,
                            cols=3,
                            column_widths=[.4, 0.2, .4],
                            subplot_titles=("Variant Correlation", "", "Variant Correlation, Ordered"))

        var = variants.tolist()
        fig = fig.add_trace(go.Heatmap(z=cc, x=var, y=var, zmin=-1, zmax=1),
                            row=1,
                            col=1)

        var = variants[s].tolist()
        fig = fig.add_trace(go.Heatmap(z=cc[s, :][:, s], x=var, y=var, zmin=-1, zmax=1),
                            row=1,
                            col=3)

        fig.update_xaxes(tickfont={'size': 8}, tickangle=-90)
        fig.update_yaxes(tickfont={'size': 8})

        return fig

    @classmethod
    @require_seaborn
    def update_coloraxis(self, fig, layer):
        """
        Sets the colorscale for DNA.

        Parameters
        ----------
        fig : plotly.Figure
            The figure whose layout has to
            be updated.
        layer : str
            The layer according to which the
            coloraxis has to be updated.
        """

        if isinstance(layer, str) and layer == AF_MISSING:
            # @HZ 07/18/2022: updated colorscale for AF_MISSING heatmap
            # cols = sns.cubehelix_palette(40, rot=(-0.2), light=0.3, dark=0.9)
            # cols = [to_hex(c) for c in cols]
            # cols = list(zip(np.linspace(1 / 3, 1, len(cols)), cols))
            # scale = [(0, "#000000"), (0.33, "#000000")]
            # scale.extend(cols)
            scale = [
                [0, 'rgb(255,255,51)'], 
                [1/3, 'rgb(204,229,255)'], 
                [2/3, 'rgb(112,112,255)'],
                [1, 'rgb(255,0,0)']
            ]

            fig.update_layout(coloraxis=dict(colorscale=scale,
                                             colorbar_tickvals=[-50, 0, 50, 100],
                                             colorbar_ticktext=['missing', 0, 50, 100],
                                             colorbar_title='VAF',
                                             cmax=100,
                                             cmin=-50))
        elif isinstance(layer, str) and layer == AF:
            cols = sns.cubehelix_palette(40, rot=(-0.2), light=0.3, dark=0.9)
            cols = [to_hex(c) for c in cols]
            cols = list(zip(np.linspace(0, 1, len(cols)), cols))
            scale = cols

            fig.update_layout(coloraxis=dict(colorscale=scale,
                                             colorbar_tickvals=[0, 50, 100],
                                             colorbar_ticktext=[0, 50, 100],
                                             colorbar_title='VAF',
                                             cmax=100,
                                             cmin=0))
        elif isinstance(layer, str) and ('NGT' in layer):
            # @HZ 07/18/2022: update NGT color scale for improved genotyping method
            # cols = sns.cubehelix_palette(3, rot=(-0.2), light=0.3, dark=0.9)
            # cols = [to_hex(c) for c in cols]
            # scale = [(0 / 4, cols[0]), (1 / 4, cols[0]),
            #          (1 / 4, cols[1]), (2 / 4, cols[1]),
            #          (2 / 4, cols[2]), (3 / 4, cols[2]),
            #          (3 / 4, "#000000"), (4 / 4, "#000000")]

            # fig.update_layout(coloraxis=dict(colorscale=scale,
            #                                  colorbar_tickvals=[3 / 8, 9 / 8, 15 / 8, 21 / 8],
            #                                  colorbar_ticktext=['WT', 'HET', 'HOM', 'Missing'],
            #                                  colorbar_title='Genotype',
            #                                  cmax=3,
            #                                  cmin=0))
            scale = [
                (0, 'rgb(204,229,255)'), (1/5, 'rgb(204,229,255)'),
                (1/5, 'rgb(245, 182, 166)'), (2/5, 'rgb(245, 182, 166)'), 
                (2/5, 'rgb(255, 0, 0)'), (3/5, 'rgb(255, 0, 0)'),
                (3/5, 'rgb(255, 255, 51)'), (4/5, 'rgb(255, 255, 51)'),
                (4/5, 'rgb(166, 178, 255)'), (1, 'rgb(166, 178, 255)')
                    ]

            fig.update_layout(coloraxis=dict(colorscale=scale,
                                            colorbar_tickvals=[2 / 5, 6 / 5, 10 / 5, 14 / 5, 18 / 5],
                                            colorbar_ticktext=['WT', 'HET', 'HOM', 'Missing','Low-conf.'],
                                            colorbar_title='Genotype',
                                            cmax=4,
                                            cmin=0))

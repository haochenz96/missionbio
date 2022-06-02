from copy import deepcopy

import matplotlib.cm as cm
import matplotlib.colors as col
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from h5.constants import AF, BARCODE, DNA_READ_COUNTS_ASSAY, PROTEIN_ASSAY, SAMPLE
from plotly.subplots import make_subplots
from scipy.cluster import hierarchy  # For hierarchical clustering
from scipy.spatial.distance import pdist  # For hierarchical clustering

from mosaic.constants import COLORS, GENE_NAME, NORMALIZED_READS, PLOIDY, READS
from mosaic.dna import Dna
from mosaic.utils import clipped_values


class Sample():
    """
    Container for multiple assays.

    Just like the :class:`mosaic.assay._Assay` object,
    `Sample` can also be filtered using Python's slice notation.

    It accepts only one key - a list of barcodes, a
    list of the position of the barcodes, or a Boolean list.

    Load the sample.

        >>> import mosaic.io as mio
        >>> sample = mio.load('/path/to/h5')

    Selecting the first 100 cells (these aren't necessarily cells
    with the highest reads, they're arbitrary  cells).

        >>> select_bars = sample.dna.barcodes()[:100]

    Slice all assays in the sample.

        >>> filtered_sample = sample[select_bars]

    Once the analysis is complete, it can be saved
    and shared using:

        >>> mio.save(sample)

    .. rubric:: Methods
    .. autosummary::

       reset
       combine_samples

    .. rubric:: Multi-assay visualizations
    .. autosummary::

       heatmap
       clone_vs_analyte
       umaps

    .. rubric:: Raw count visualizations
    .. autosummary::

       assay_scatter
       raw_heatmaps
       read_data
    """

    assay_color = COLORS[1]

    def __init__(self, name=None, dna=None, cnv=None, protein=None, cnv_raw=None, protein_raw=None):

        self.name = name
        if name is None and dna is not None:
            self.name = dna.metadata[SAMPLE]
            self.name = ", ".join(self.name.flatten())

        self.dna = dna
        self.cnv = cnv
        self.protein = protein
        self.cnv_raw = cnv_raw
        self.protein_raw = protein_raw

        self._original_dna = self.dna
        self._original_cnv = self.cnv
        self._original_protein = self.protein

    def __getitem__(self, key_bars):
        """
        Returns a slice of a multi-analyte sample using bracket syntax.
        E.g., sample_filtered = sample[good_barcodes,:]

        Parameters
        ----------
        key_bars: tuple
            Indices of the rows, columns of the slice to be returned.

        Returns
        -------
        sample: Sample object
            A slice of the original sample object.

        Raises
        ------
        TypeError
            When "key_bars" provided is not of type tuple.
        """

        sample = deepcopy(self)

        if isinstance(key_bars, tuple):
            raise TypeError('The key should be a list of barcodes or indices.')

        assays = [sample.dna, sample.protein, sample.cnv]

        for i in range(len(assays)):
            if assays[i] is not None:
                assays[i] = assays[i][key_bars, :]

        sample.dna, sample.protein, sample.cnv = assays

        return sample

    def reset(self, assay=None):
        """
        Resets to the original state.

        If the given assay is `None`, then all the assays
        are restored to the state where all the `barcode`s and `id`s
        from the original state are available.

        Parameters
        ----------
        assay : str
            'dna', 'protein', or 'cnv'

        Notes
        -----
        Some added layers might persist, depending on when
        the assay object was sliced.
        """
        if assay is None:
            self.dna = self._original_dna
            self.cnv = self._original_cnv
            self.protein = self._original_protein
        elif assay == 'dna':
            self.dna = self._original_dna
        elif assay == 'cnv':
            self.cnv = self._original_cnv
        elif assay == 'protein':
            self.protein = self._original_protein

    # --------- Plotting

    def assay_scatter(self, ax=None, title='', highlight=None):
        """
        Plots DNA reads vs Protein reads.

        This plot is made for all barcodes, not just
        called cells, and the called cells are colored.

        Requires
        --------
        cnv_raw and protein_raw to be loaded.

        Parameters
        ----------
        ax : matplotlib.pyplot.axis
            The axis to plot the plot on.
        title : str
            The title of the plot.
        highlight : list
            The barcodes to highlight on the plot
            (besides the called cells).

        Returns
        -------
        ax : matplotlib.pyplot.axis

        Raises
        ------
        ValueError
            When the raw counts are not loaded.
        """

        if self.cnv_raw is None or self.protein_raw is None:
            raise ValueError('Raw counts are not loaded.')

        df_cnv = pd.DataFrame(self.cnv_raw.layers[READS], index=self.cnv_raw.barcodes(), columns=self.cnv_raw.ids())
        df_cnv.loc[:, 'DNA reads'] = df_cnv.sum(axis=1)
        df_prot = pd.DataFrame(self.protein_raw.layers[READS], index=self.protein_raw.barcodes())
        df_prot.loc[:, 'Protein reads'] = df_prot.sum(axis=1)
        data = pd.concat([df_cnv['DNA reads'], df_prot['Protein reads']], axis=1, sort=True).fillna(0)
        data = np.log10(data + 1)
        cell_data = data.loc[self.protein.barcodes()]

        sns.set(style='whitegrid', font_scale=1.5)
        plt.figure(figsize=(10, 10))

        ax = sns.scatterplot(data=data, x='DNA reads', y='Protein reads', ax=ax, color=COLORS[-1], s=12)
        ax = sns.scatterplot(data=cell_data, x='DNA reads', y='Protein reads', ax=ax, color=self.assay_color, s=12)
        if highlight is not None:
            highlight_data = data.loc[highlight]
            ax = sns.scatterplot(data=highlight_data, x='DNA reads', y='Protein reads', ax=ax, color=COLORS[2], s=12)

        ax.set_xlabel('log$_{10}$(1 + Number of DNA reads)')
        ax.set_ylabel('log$_{10}$(1 + Number of Protein reads)')

        if title == '':
            title = f'{self.name} - '

        title += 'DNA vs Protein'
        ax.set_title(title, fontsize=19)

        return ax

    def heatmap(self, clusterby, sortby, drop=None, flatten=True):
        """
        Multi-assay heatmap.

        Plots heatmaps (cluster vs target) for each available assay next to each other.
        The plots can either be "flattened", in which case the heatmap shows the median target value for all
        barcodes in the cluster. Or "not flattened", in which case all barcodes are shown within each cluster.

        Parameters
        ----------
        clusterby : str
            The assay name ('dna', 'protein') whose labels
            will be used to cluster the barcodes.
        sortby : str
            The assay name ('dna', 'protein') whose labels will
            be used to sort the barcodes within each cluster.
        drop : str
            At most one of ('dna', 'protein', 'cnv') that is to be dropped.
        flatten : boolean
            Determines whether to display the median of barcodes for
            each cluster at each target (True) or plot each barcode,
            grouped by cluster, at each target (False).


        Returns
        -------
        fig : plotly.graph_objects.Figure
        """

        data = {'dna': self.dna,
                'protein': self.protein,
                'cnv': self.cnv}

        if drop is not None:
            if drop == clusterby or drop == sortby:
                raise ValueError('Cannot drop `clusterby` or `sortby`')
            del data[drop]

        titles = {'dna': 'DNA', 'protein': 'Protein', 'cnv': 'CNV'}
        title = titles[clusterby] + ' Cluster Signature vs Analyte'

        if not flatten:
            title = title + ' and Barcode, Subsorted by ' + titles[sortby]

        labels, omics = self._heatmap_setup(clusterby=clusterby, sortby=sortby, data=data)

        u, lbl2 = np.unique(labels, return_inverse=True)

        no = len(omics)
        nc = len(u)

        w = [omics[ii].width for ii in range(no)]
        weight = [omics[ii].colweight for ii in range(no)]

        p = np.array(w) * np.array(weight)
        cw = np.append(0.05, 0.95 * p / np.sum(p)).tolist()
        fig = make_subplots(rows=1, cols=no + 1, column_widths=cw,
                            horizontal_spacing=.01, vertical_spacing=0,
                            subplot_titles=np.append('', [omics[ii].title for ii in range(no)]))

        # Left-most "cluster" column

        cmap = []
        tickvals_lab = []
        ticktexts = []
        nu = len(u)
        pal = data[clusterby].get_palette()
        for i in range(nu):
            cmap.append((i / nu, pal[u[i]]))
            cmap.append(((i + 1) / nu, pal[u[i]]))
            tickvals_lab.append(len(lbl2) - ((lbl2 >= i).sum() + (lbl2 > i).sum()) / 2)
            ticktexts.append(f'<b>{u[i]}</b> {(lbl2 == i).sum() / len(lbl2):.1%}')

        if flatten:
            fig = fig.add_trace(go.Heatmap(z=np.unique(lbl2)[:, None].astype(str),
                                           showscale=False,
                                           colorscale=cmap),
                                row=1, col=1)
        else:
            fig = fig.add_trace(go.Heatmap(z=np.sort(lbl2)[:, None].astype(str),
                                           showscale=False,
                                           colorscale=cmap),
                                row=1, col=1)

        # For each omic
        for ii in range(no):
            if flatten:

                z = np.zeros((nc, omics[ii].width))
                for jj in range(nc):
                    z[jj, :] = np.median(omics[ii].data[lbl2 == jj, :], axis=0)
            else:
                z = omics[ii].data

            # Heatmap
            heat = go.Heatmap(z=z, x=np.arange(omics[ii].width),
                              colorbar=dict(len=1 / no, y=(no - ii - 1 / 2) / no,
                                            title=omics[ii].title),
                              zmin=omics[ii].ztickvals[0], zmax=omics[ii].ztickvals[1],
                              colorscale=px.colors.sequential.Viridis,
                              name=omics[ii].title)

            # Add heatmap to the figure
            fig = fig.add_trace(heat, row=1, col=2 + ii)

            # Set heatmap ranges/labels
            tickvals = np.arange(0, omics[ii].width, omics[ii].xtickgap)
            fig.update_xaxes(ticks='outside', range=[-0.5, omics[ii].width - 0.5],
                             tickvals=tickvals, ticktext=omics[ii].ticklabels[tickvals], col=2 + ii)

            # Add lines separating the chromosomes
            if omics[ii].title == 'CNV':
                for line in np.arange(omics[ii].width - 1):
                    if omics[ii].ticklabels[line].split(':')[0] != omics[ii].ticklabels[line + 1].split(':')[0]:
                        fig = fig.add_trace(go.Scatter(x=[line + 0.5, line + 0.5], y=[-0.5, z.shape[0] + 0.5],
                                                       mode='lines', line_color='rgba(0,0,0,.6)', showlegend=False),
                                            row=1, col=2 + ii)
            elif omics[ii].title == 'SNV':
                Dna.update_coloraxis(fig, 'NGT')
                fig.data[ii + 1].coloraxis = 'coloraxis'
                fig.layout.coloraxis.colorbar.len = 1 / no
                fig.layout.coloraxis.colorbar.y = (no - ii - 1 / 2) / no
            elif omics[ii].title == 'Protein':
                fig.data[-1].colorscale = 'magma'

        # Index of the last barcode in each cluster
        cs = np.append(0, np.cumsum([np.sum(lbl2 == i) for i in range(nc)]))

        # Cycle through each cluster
        for ii in range(nc):
            # Draw lines across all subplots to denote clusters
            if (not flatten) & (ii < nc - 1):
                for jj in range(no + 1):
                    xmax = 0.5 if jj == 0 else omics[jj - 1].width - 0.5
                    fig = fig.add_trace(go.Scatter(x=[-0.5, xmax], y=[cs[ii + 1], cs[ii + 1]],
                                                   mode='lines', line_color='rgba(1,1,1,1)', showlegend=False),
                                        row=1, col=jj + 1)

        # Set up all subplots
        fig.update_yaxes(showticklabels=False)
        fig.update_xaxes(tickangle=-90, tickmode='array',
                         showline=True, mirror=True)

        # Set up the left-most "cluster" column
        if flatten:
            rng = [nc - 0.5, -0.5]
            height = 800
            fig.update_yaxes(title='Clusters', showticklabels=False, col=1)
        else:
            rng = [len(labels), 0]
            height = 800
            fig.update_yaxes(showticklabels=True,
                             ticks='outside', col=1,
                             ticktext=ticktexts,
                             tickvals=tickvals_lab)

        fig.update_yaxes(range=rng, showline=True, mirror=True)

        fig.update_xaxes(showticklabels=False, col=1)

        # Finish the figure
        fig.update_layout(height=height, title=title)

        return fig

    def clone_vs_analyte(self, analyte='cnv', plot_width=0):
        """
        Set of summary plots to give a run overview.

        The plots have three portions:
            1) The left-most is the VAF vs subclone/variant table.
            2) The middle is the cell population count/fraction.
            3) It depends on the input:
                - If the analyte is cnv, it shows the read count distribution plot.
                - If the analyte is protein, it shows the violin plot of proteins.

        Parameters
        ----------
        analyte : str
            cnv or protein - to be used in the third plot.
        plot_width : float
            The width of the plot in points.
        """

        scale = 1
        gap = .03
        titlefontsize = 14

        dna = self.dna
        if analyte == 'protein':
            analyte = self.protein
        else:
            analyte = self.cnv

        if analyte is None:
            raise Exception("Analyte is None. Please choose another analyte.")

        if set(dna.barcodes()) != set(analyte.barcodes()):
            raise Exception('The DNA assay does not contain the same barcodes as the given assay.'
                            ' Please make sure that these two assays of the sample object contain the same cells')

        nv = dna.shape[1]
        clust, clustind = np.unique(dna.get_labels(), return_inverse=True)

        clust_names = np.array([clust[i] for i in range(len(clust))])

        nc = len(clust)

        vaf, vafstd, pval, _ = dna.feature_signature(layer=AF)
        vaf = vaf / 100

        vaf = vaf.T

        clust_keep = np.array([np.sum(clustind == i) for i in range(nc)])
        clust_keep = np.where(clust_keep >= 0)[0]

        nc = len(clust_keep)
        w = 20

        if analyte is None:
            h = nc
        else:
            if analyte.name == PROTEIN_ASSAY:
                h = nc
            else:
                h = nc * 1.5

        if plot_width > 0:
            w = plot_width

        fig = plt.figure(figsize=(w, h))

        # Flatten the sample name array down to strings
        samplename = dna.title

        cols = [dna.get_palette()[i] for i in clust_names]
        vals = np.arange(len(cols)) / (len(cols) - 1)
        colors = list(zip(vals, cols))
        custom_color_map = LinearSegmentedColormap.from_list(
            name='custom_navy',
            colors=colors,
        )

        # First column
        ax = fig.add_axes([.05, .1, .01, .75])
        ims = ax.imshow(np.arange(len(np.unique(dna.get_labels())))[:, None], aspect='auto', origin='high', cmap=custom_color_map)

        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks(np.arange(nc))
        ax.set_yticklabels(clust_names, fontsize=titlefontsize)
        ax.set_ylim([nc - 0.5, -0.5])
        ax.set_ylabel('Subclones', fontsize=titlefontsize)

        # Clonal Heatmap
        x0 = (0.015 * nv + 0.08) * scale
        ax = fig.add_axes([0.06, 0.1, x0 - 0.06, 0.75])

        cols1 = sns.cubehelix_palette(40, rot=(-0.2), light=0.2, dark=0.9).as_hex()
        cols1 = list(zip(np.linspace(1 / 3, 1, len(cols1)), cols1))
        cols = [(0, "#000000"), (0.33, "#000000")]
        cols.extend(cols1)
        custom_color_map = LinearSegmentedColormap.from_list(
            name='vaf_map',
            colors=cols,
        )

        ims = ax.imshow(vaf.iloc[clust_keep, :].astype(float), aspect='auto', origin='high', vmin=-0.5, vmax=1, cmap=custom_color_map)

        if nv < 40:
            for i in range(nc):
                for j in range(nv):
                    if vaf.iloc[clust_keep[i], j] < 0:
                        ax.text(j, i, 'MIS', va='center', ha='center', color=[1, 1, 1, 1], fontsize=titlefontsize * 0.7)
                    elif vaf.iloc[clust_keep[i], j] < .1:
                        ax.text(j, i, 'WT', va='center', ha='center', color=[1, 1, 1, 1], fontsize=titlefontsize * 0.7)
                    elif vaf.iloc[clust_keep[i], j] > 0.9:
                        ax.text(j, i, 'HOM', va='center', ha='center', fontsize=titlefontsize * 0.7)
                    else:
                        ax.text(j, i, 'HET', va='center', ha='center', fontsize=titlefontsize * 0.7)

        ax.hlines(np.arange(nc) - 0.5, ax.get_xlim()[0], ax.get_xlim()[1], color=[0, 0, 0, 1], linewidth=1)

        xticklab = [s.replace(' - ', '\n') for s in vaf.columns]

        plt.title('Genotype', fontsize=titlefontsize)
        ax.set_xticks(np.arange(nv))
        ax.set_xticklabels(xticklab, rotation=90, fontsize=titlefontsize * 0.7)
        ax.set_yticks([])
        ax.set_ylim([nc - 0.5, -0.5])
        ax.set_xlabel('Variants', fontsize=titlefontsize)
        ax.grid(False)

        ax.annotate('Sample: ' + samplename, xy=(0.01, 1), xycoords='figure fraction', fontsize=18, verticalalignment='top')

        cbaxes = fig.add_axes([.97, .6, .02, .3])
        cbar = plt.colorbar(ims, cax=cbaxes, ticks=[-0.5, 0, 0.5, 1])
        cbar.set_ticklabels(['missing', '0', '50', '100'])
        plt.title('VAF')

        # Clonal bar graph
        ax = fig.add_axes([x0 + gap, 0.1, 0.04 * scale, 0.75])
        y = [sum(clustind == clust_keep[nc - 1 - i]) / len(clustind) * 100 for i in range(nc)]
        ax.barh(np.arange(nc), y, color=[dna.get_palette()[i] for i in clust_names][::-1])
        mn = np.mean(ax.get_xlim())
        for i in range(nc):
            ax.text(y[i] + (-mn * 0.08 if y[i] > 3 / 2 * mn else 0), i, s='{:,}'.format(np.sum(clustind == clust_keep[nc - 1 - i])),
                    ha='left' if y[i] < 3 / 2 * mn else 'right', va='center', fontweight='bold', rotation=-90,
                    fontsize=titlefontsize * 0.8)

        ax.set_xlim(ax.get_xlim())

        ax.vlines(np.arange(0, ax.get_xlim()[1], 10), ax.get_ylim()[0], ax.get_ylim()[1], color=[0, 0, 0, .3], linewidth=1)
        ax.set_yticks([])
        plt.xticks(fontsize=titlefontsize)
        ax.set_ylim([-0.5, nc - 0.5])

        plt.title('Counts', fontsize=titlefontsize)
        ax.set_xlabel('(%)', fontsize=titlefontsize)

        x0 = x0 + 0.04 * scale + gap

        # Analyte map
        if analyte is None:
            return

        if analyte.name == PROTEIN_ASSAY:
            protein = analyte
            w = 0.92 - x0

            pn = protein.shape[1]

            for p in range(pn):
                dat = {'Cluster': clust_names[clustind], 'Counts': protein.layers['normalized_counts'][:, p]}
                df = pd.DataFrame(data=dat)

                ax = fig.add_axes([x0 + gap + w / pn * p, 0.1, w / pn - 0.002, 0.75])
                ax = sns.violinplot(data=df, x='Counts', y='Cluster', width=0.9, ax=ax, orient='h',
                                    palette=dna.get_palette(), order=clust_names[clust_keep], scale='width')

                plt.xticks(fontsize=titlefontsize * 0.6)
                ax.set_yticks([])
                ax.set_ylim(ax.get_ylim())
                ax.set(xlabel=None)

                plt.title(protein.col_attrs['id'][p], fontsize=titlefontsize, rotation=90, va='bottom')
                if p == int(pn / 2):
                    plt.xlabel('Normalized Protein Reads')

                plt.ylabel('')

        if analyte.name == DNA_READ_COUNTS_ASSAY:
            cnv = analyte
            ax = fig.add_axes([x0 + gap, 0.1, 0.92 - x0, 0.75])
            cnt = 0

            if PLOIDY not in cnv.layers:
                raise Exception("Ploidy has not yet been computed for this assay. Run sample.cnv.compute_ploidy")

            cnv_dat = cnv.layers[PLOIDY].copy()
            na = cnv_dat.shape[1]
            nh = 50
            Hs = np.zeros((0, na))

            ids, features = cnv._get_ids_from_features('genes')
            order = cnv.col_attrs[GENE_NAME].copy().argsort()

            for c in range(nc):

                in_clust = (c == clustind)
                cnv_clust = cnv_dat[in_clust, :][:, order]

                x = np.arange(na)
                y = np.median(cnv_clust, axis=0)
                ax.plot(x, y + (nc - cnt - 1) * 6 + 1, color=[0, 0, 0, 1], linewidth=0, marker='s', markersize=3)

                nb = np.sum(in_clust)

                xx = np.array([[i for i in range(na)] for j in range(nb)])
                xx = xx.reshape(na * nb, 1)[:, 0]

                yy = cnv_clust.reshape(na * nb, 1)[:, 0]
                ybins = np.linspace(-1, 5, nh + 1)
                ybins[-1] = 100

                H, xedges, yedges = np.histogram2d(xx, yy, bins=(np.linspace(0.5, na + 0.5, na + 1), ybins))
                H = np.minimum(H, np.quantile(H, .995))
                H = H / np.max(H)
                Hs = np.concatenate((H.T, Hs), axis=0)

                cnt = cnt + 1

            bone = cm.get_cmap('bone', nh * 2)
            cmap = col.ListedColormap(bone(np.linspace(0, 1, nh * 2))[2 * nh:nh:-1])
            ax.imshow(Hs, origin='low', extent=[xedges[0], xedges[-1], 0, 6 * nc], cmap=cmap, aspect='auto', interpolation='none')

            un, ind, cnts = np.unique(features, return_index=True, return_counts=True)
            ticks = (ind + cnts / 2).astype(int)
            ax.set_xticks(ticks)
            ax.set_xticklabels(features[ticks], rotation='vertical')
            ax.yaxis.tick_right()
            yt = np.arange(1, 6 * nc)
            yt2 = (yt - 1) % 6
            ax.set_yticks(yt)
            ax.set_yticklabels(yt2.astype(str))
            ax.set_xlim([-0.5, na - 0.5])
            ax.set_ylim([0, 6 * nc])
            plt.title('Amplicon Read Distribution vs Subclone', fontsize=14)

            for line in ind:
                ax.plot([line - 0.5, line - 0.5], ax.get_ylim(), color=[.5, .5, .5, 1], linewidth=.5)

            for lines in range(6 * nc):
                ax.plot(ax.get_xlim(), [lines, lines], color=[0, 0, 0, 1], linewidth=.5, linestyle='dotted')
                ax.plot(ax.get_xlim(), [lines + 0.1, lines + 0.1], color=[1, 1, 1, 1], linewidth=.5, linestyle='dotted')
            for lines in range(0, 6 * nc, 6):
                ax.plot(ax.get_xlim(), [lines, lines], color=[0, 0, 0, 1], linewidth=1)

    def umaps(self):
        """
        Plots umap arrays from assay objects.

        Plots 4 umaps, 2 clustered by DNA and 2 clustered by Protein,
        and then colored by both DNA and Protein cluster labels.

        Returns
        -------
        fig : plotly.graph_objects.Figure
        """

        if self.protein is None:
            dat = [self.dna]
        else:
            dat = [self.dna, self.protein]

        titles = ['DNA', 'Protein']
        fig = make_subplots(rows=len(dat), cols=len(dat),
                            subplot_titles=[(titles[int(ii / 2)] + ' UMAP, Colored by ' + titles[ii % 2]) for ii in range(len(dat)**2)])

        for ii in range(len(dat)):
            for jj in range(len(dat)):

                u, lbli = np.unique(dat[jj].get_labels(), return_inverse=True)
                umap_dat = dat[ii].row_attrs['umap']

                nu = len(u)
                ls = np.linspace(0, 1, nu)

                cmap = (cm.rainbow(ls) * 255).astype(int)

                cmap = [[ls[i], f'rgb({cmap[i,0]},{cmap[i,1]},{cmap[i,2]})'] for i in range(nu)]

                data = go.Scattergl(x=umap_dat[:, 0], y=umap_dat[:, 1], mode='markers',
                                    marker=dict(size=2, color=lbli, colorscale=cmap),
                                    showlegend=False)

                fig.add_trace(data, row=ii + 1, col=jj + 1)

        xax = dict(linecolor='black', linewidth=1, mirror=True,
                   ticks='inside', tickfont={'size': 9})
        fig.update_xaxes(xax)
        fig.update_yaxes(xax)
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', width=1000, height=1000)

        return fig

    def raw_heatmaps(self):
        """
        Protein and DNA read distribution.

        Three heatmaps with barcodes mapped by total DNA reads (x-axis) and total protein reads (y-axis).
            The first shows the number of barcodes in each DNA/protein read bin.
            The second shows the number of protein reads in each DNA/protein read bin.
            The third shows the number of DNA reads in each DNA/protein read bin.

        This is useful for understanding the distribution barcodes/protein reads/DNA reads within certain clusters.

        Requires
        --------
        cnv_raw and protein_raw to be loaded.
        """

        if (self.cnv_raw is not None) & (self.protein_raw is not None):
            cnv = self.cnv_raw
            prot = self.protein_raw

            barcodes, cnv_ind, prot_ind = np.intersect1d(cnv.barcodes(), prot.barcodes(), return_indices=True)

            fig = plt.figure(figsize=(20, 15))

            xx = np.sum(cnv.layers['read_counts'][cnv_ind], axis=1) + 1
            yy = np.sum(prot.layers['read_counts'][prot_ind], axis=1) + 1

            h, xedges, yedges = np.histogram2d(np.log10(yy), np.log10(xx), bins=50)

            scale = [np.zeros((len(yedges) - 1, len(xedges) - 1)),
                     xedges[:-1, None], yedges[None, :-1]]

            titles = ['Barcode Count', 'Protein Reads', 'DNA Reads']

            for ii in range(3):
                ax = fig.add_subplot(2, 2, ii + 1)
                im = ax.imshow(np.log10(h * (10 ** scale[ii]) + 1),
                               cmap=plt.get_cmap('jet'),
                               origin='lower', vmin=0, vmax=6.5, extent=[0, max(xedges), 0, max(yedges)])
                ax.set_ylabel('Protein Reads')
                ax.set_xlabel('DNA Reads')
                ax.set_yticklabels(['10$^{' + str(ii) + '}$' for ii in np.arange(7)])
                ax.set_xticklabels(['10$^{' + str(ii) + '}$' for ii in np.arange(7)])
                plt.title('Distribution of ' + titles[ii])

                cbar = fig.colorbar(im, ax=ax)

                cbar.set_ticks(np.arange(7))
                cbar.ax.set_yticklabels(['10$^{' + str(ii) + '}$' for ii in np.arange(7)])
                cbar.ax.set_ylabel(titles[ii] + ' Per Pixel',
                                   rotation=270, labelpad=20)

    def read_data(self):
        """
        Plot read statistics of each barcode.

        A total of 6 plots are generated:
            1) Protein reads vs DNA reads, colored by clone.
            2) Genotyped knee plot, showing clonal distribution along the rank-ordered knee.
            3) Distribution of each clone type by rank-ordered barcode, useful to identify doublets/mergers.
            4) 1X/10X plot colored by clone, useful for identifying partial/mixed barcodes.
            5) DNA reads vs unique DNA targets.
            6) Protein reads vs unique protein targets.
        """
        fig = plt.figure(figsize=(25, 22))
        cnt = 1
        dna = self.dna

        clust_names = np.unique(dna.get_labels())
        clust_names = np.append(clust_names, 'Ungen')
        nc = len(clust_names)

        _, dnalbl = np.unique(dna.get_labels(), return_inverse=True)

        colors = cm.rainbow(np.linspace(0, 1, nc - 1))
        colors = np.concatenate((colors, [[0, 0, 0, 1]]), axis=0)
        colors2 = deepcopy(colors)
        colors2[-1, -1] = .2

        if (self.cnv_raw is None) | (self.protein_raw is None):
            return

        cnv = deepcopy(self.cnv_raw)
        cnv.dat = cnv.layers[READS]
        prot = deepcopy(self.protein_raw)
        prot.dat = prot.layers[READS]

        barcodes, cnv_ind, prot_ind = np.intersect1d(cnv.barcodes(), prot.barcodes(), return_indices=True)
        ax = fig.add_subplot(3, 3, cnt)

        _, ind2, filt_ind = np.intersect1d(barcodes, dna.barcodes(), return_indices=True)

        c = np.ones(len(cnv_ind)) * (nc - 1)

        c[ind2] = dnalbl[filt_ind]

        jitter = np.random.rand(len(cnv_ind), 2)

        cc = (np.sum(cnv.dat[cnv_ind, :], axis=1) > 0) & (np.sum(prot.dat[prot_ind, :], axis=1) > 0)

        plt.scatter(np.sum(cnv.dat[cnv_ind[cc], :], axis=1) + 1 + jitter[cc, 0],
                    np.sum(prot.dat[prot_ind[cc], :], axis=1) + jitter[cc, 1],
                    c=colors2[c[cc].astype(int), :], s=1)

        ax.set_xlabel('DNA Reads')
        ax.set_ylabel('Protein Reads')
        ax.set_title('Protein vs DNA Reads')
        ax.set_xlim([1, ax.get_xlim()[1]])
        plt.xscale('log')
        plt.yscale('log')

        cnt = cnt + 1

        self._add_legend(ax, clust_names, colors)

        # Calculate the read statistics for rank ordering
        cnv.dat_sum = np.sum(cnv.dat, axis=1)

        keep = cnv.dat_sum >= 20
        cnv.dat = cnv.dat[keep, :]
        cnv.dat_sum = cnv.dat_sum[keep]
        cnv.row_attrs[BARCODE] = cnv.barcodes()[keep]
        cnv.num_bar = len(cnv.barcodes())

        sr = cnv.dat_sum

        ss = np.sort(sr)[::-1]
        sind = np.argsort(sr)[::-1]
        ordr = np.argsort(sind)
        f0 = np.arange(cnv.num_bar)

        # Generate rank-ordered histogram bins used in the barcode distribution
        xb = np.concatenate(([1, 50], np.arange(100, 2000, 150), np.arange(2200, 1e4, 300), np.arange(1e4, 1e6, 1e3)))
        xb = xb[:(xb < np.max(ordr)).sum() + 1]

        nb = len(xb)

        _, unfilt_ind, filt_ind = np.intersect1d(
            cnv.barcodes(), dna.barcodes(), return_indices=True)
        c = np.ones(cnv.num_bar) * (nc - 1)
        c[unfilt_ind] = dnalbl[filt_ind]

        typedistr = np.zeros((nb - 1, nc + 1))

        # Connect genotype classifications to rank order
        read_type = []
        for ii in range(nc + 1):
            ord_ind = np.array(np.where(c == ii))[0]
            read_type.append(ord_ind)
            typedistr[:, ii] = np.histogram(ordr[ord_ind], xb)[0]

        # Normalize the genotype distributions
        rng = np.sum(typedistr, axis=1) > 0
        typedistr = typedistr / (np.sum(typedistr, axis=1) + 0.01)[:, None]

        # Cut off the histogram where there are no more counts
        xb = xb[:-1][rng]
        typedistr = typedistr[rng]

        # Genotyped knee plot
        ax = fig.add_subplot(3, 3, cnt)
        ax.plot(f0 + 1, ss, color='black', linewidth=4)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.title('Barcodes vs Reads')
        plt.xlabel('Rank-Ordered Barcodes by DNA Reads')
        plt.ylabel('DNA Reads')
        for ii in range(nc):
            xx = ordr[read_type[ii]] + 1
            if len(xx) > 0:
                yy = 10 ** (5 - ii * 1.5 + np.random.rand(read_type[ii].shape[0]))
                s = np.maximum(30 - 10 * np.log10(xx), 1)
                plt.scatter(xx, yy, s=s, linewidth=0,
                            marker='o', color=colors2[ii, :])
            ax.text(x=0.2, y=10 ** (5 - (ii - 0.1) * 1.5 + 0.5),
                    s=clust_names[ii], color=colors[ii, :], ha='left', fontsize=12)
            ax.text(x=0.8, y=10 ** (5 - (ii + 0.1) * 1.5 + 0.5), s=len(xx),
                    color=colors[ii, :], ha='right', fontsize=12)
        xl = ax.get_xlim()
        yl = ax.get_ylim()
        ax.set_xlim([.1, max(xl)])
        ax.set_ylim(yl)
        ax.vlines([1, len(dna.barcodes())], ax.get_ylim()[0],
                  ax.get_ylim()[1], color='black', linestyle='dotted')

        self._add_legend(ax, clust_names, colors)

        cnt = cnt + 1

        # Barcode type distribution by rank-ordered barcode
        ax = fig.add_subplot(3, 3, cnt)
        xx = np.concatenate(([1], 10 ** ((np.log10(xb[0:-1]) + np.log10(xb[1:])) / 2), [xb[-1]]))
        hand = []
        for ii in range(nc):
            yy = np.concatenate(([typedistr[0, ii]], typedistr[:, ii]))
            h = ax.plot(xx, yy, color=colors[ii, :])
            hand = np.append(hand, h)
        ax.vlines([1, len(dna.barcodes())], ax.get_ylim()[0],
                  ax.get_ylim()[1], color='black', linestyle='dotted')

        ax.set_xlabel('Rank-Ordered Barcodes by DNA Reads')
        ax.set_ylabel('Fraction of Barcodes')
        ax.set_title('Distribution of Clones vs Rank-Ordered Barcodes')
        ax.set_xscale('log')
        ax.set_xlim([.1, max(xl)])
        ax.set_ylim([0, 1.1])
        ax.legend(handles=hand.tolist(), labels=clust_names.tolist())

        cnt = cnt + 1

        del cnv

        analytes = [1, 1, 2, 1, 1]

        plottype = [1, 2, 2, 1, 2]

        for ii in range(len(analytes)):

            if analytes[ii] == 1:
                name = 'DNA'
                omic = self.cnv_raw
                omic.dat = omic.layers[READS]
            else:
                name = 'Protein'
                omic = self.protein_raw
                omic.dat = omic.layers[READS]

            na = omic.dat.shape[1]
            nb = omic.dat.shape[0]
            ax = fig.add_subplot(3, 3, cnt)

            jitter = np.random.rand(nb, 2)

            _, unfilt_ind, filt_ind = np.intersect1d(omic.barcodes(), dna.barcodes(), return_indices=True)
            c = np.ones(nb) * (nc - 1)
            c[unfilt_ind] = dnalbl[filt_ind]

            omic.dat_sum = np.sum(omic.dat, axis=1)

            cc = omic.dat_sum > 0

            if plottype[ii] == 1:
                plt.scatter(np.sum(omic.dat[cc, :] >= 1, axis=1) + jitter[cc, 0],
                            np.sum(omic.dat[cc, :] >= 10, axis=1) + jitter[cc, 1],
                            c=colors2[c[cc].astype(int), :], s=2)
                ax.plot(np.array([.30, .88, .79, .21, .30]) * na,
                        np.array([0.17, 0.75, 0.84, 0.26, 0.17]) * na,
                        color=[1, 0, 0, .5], linestyle='dotted')
                ax.text(x=50, y=50, s='Partial Cells', color='r',
                        rotation=45, va='bottom', ha='center')
                ax.plot([0, na], [0, na], color='k', linestyle='dotted')
                ax.axis('equal')
                ax.set_xlabel('%s Targets by Barcode with Counts >=1' % name)
                ax.set_ylabel('%s Targets by Barcode with Counts >=10' % name)
                ax.set_title('%s Targets by Barcode with 10X Coverage vs 1X Coverage' % name)
                ax.set_xlim([0, na * 1.05])

                if ii > 2:
                    plt.xscale('log')
                    plt.yscale('log')
                    ax.set_xlim([.5, na * 1.05])
                    ax.set_ylim([.5, na * 1.05])

            elif plottype[ii] == 2:
                plt.scatter(omic.dat_sum[cc] + 1 + jitter[cc, 0],
                            np.sum(omic.dat[cc, :] >= 10, axis=1) + jitter[cc, 1],
                            c=colors2[c[cc].astype(int), :], s=2)
                ax.set_xlabel('%s Reads by Barcode' % name)
                ax.set_ylabel('%s Targets by Barcode with Counts >=10' % name)
                ax.set_title(
                    '%s Targets by Barcode with 10X Coverage vs %s Read Count' % (name, name))
                ax.set_xlim([1, np.max(omic.dat_sum[cc]) * 2])
                ax.set_ylim([0, na * 1.05 + 1])
                plt.xscale('log')

                if ii > 2:
                    plt.yscale('log')
                    ax.set_ylim([0.5, na * 1.05])

            self._add_legend(ax, clust_names, colors)

            cnt = cnt + 1

    @staticmethod
    def _heatmap_setup(data=None, clusterby=None, sortby=None):
        """
        Setup for :meth:`mosaic.sample.Sample.heatmap`.

        Takes a dictionary of assay objects, and prepares the parameters for
        plotting heatmaps, generate spacing, ticks, colorbar parameters, title, etc.

        Parameters
        ----------
        data : dictionary object
                Has three possible keys: a dna object, a cnv object, and a prot object.
        clusterby : string
                Assay name ('dna', 'protein') and corresponding labels used to cluster barcodes.
        sortby : string
                Assay name ('dna', 'protein') and corresponding labels used to sort barcodes within each cluster.


        Returns
        -------
        labels : string array
                Contains the labels from the assay indicated by "clusterby".
        omics : class
                A list of objects (same length as the data parameter) of class 'omic' with several plotting parameters.
        """

        lbl = []
        lbl2 = []

        class omic:
            def __init__(self, **kwargs):
                self.data = None
                self.ztickvals = None
                self.ticklabels = None
                self.xtickgap = None
                self.title = None
                self.coltitles = None
                self.colweight = None
                self.width = None

        omics = []

        lbl = data[clusterby].get_labels()

        if sortby is not None:
            lbl2 = data[sortby].get_labels()
        else:
            lbl2 = lbl

        u, lbli = np.unique(lbl, return_inverse=True)
        u2, lbli2 = np.unique(lbl2, return_inverse=True)

        a = np.argsort(lbli * 1e6 + lbli2)

        labels = lbl[a]

        if 'dna' in data:
            om = omic()

            dna = data['dna']
            vaf = dna.layers['NGT']

            if vaf.T.shape[0] > 1:
                D = pdist(vaf.T, 'correlation')
                D[np.isnan(D)] = 0
                tree = hierarchy.linkage(D)
                s = hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(tree, D))
            else:
                s = [0]

            xticklab = np.array([s.replace(' - ', '<br>') for s in dna.ids()])

            om.data = vaf[a, :][:, s]
            om.ztickvals = clipped_values(om.data.flatten())
            om.xtickgap = 1
            om.title = 'SNV'
            om.coltitles = 'VAF'
            om.ticklabels = xticklab[s]
            om.colweight = 1
            om.width = om.data.shape[1]
            omics = omics + [om]

        if 'cnv' in data:
            om = omic()

            cnv = data['cnv']
            cnv_norm = cnv.layers['read_counts']
            tot = np.sum(cnv_norm, axis=1)
            s = np.argsort(tot)[np.maximum(len(tot) - 1000, 0):]

            om.data = cnv.layers[NORMALIZED_READS][a, :]
            om.ztickvals = clipped_values(om.data.flatten())
            om.xtickgap = 1
            om.title = 'CNV'
            om.coltitles = 'Norm Counts'
            om.ticklabels = cnv.ids()
            om.colweight = .75
            om.width = om.data.shape[1]
            omics = omics + [om]

        if 'protein' in data:
            om = omic()
            prot = data['protein']
            prot_norm = prot.layers['normalized_counts']
            D = pdist(prot_norm.T, 'correlation')
            D[np.isnan(D)] = 0
            tree = hierarchy.linkage(D)
            s = hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(tree, D))

            om.data = prot_norm[a, :][:, s]
            om.ztickvals = clipped_values(om.data.flatten())
            om.xtickgap = 1
            om.title = 'Protein'
            om.coltitles = 'Norm Counts'
            om.ticklabels = prot.col_attrs['id'][s]
            om.colweight = 1
            om.width = om.data.shape[1]
            omics = omics + [om]

        return labels, omics

    @staticmethod
    def _add_legend(ax, clust_names, colors, loc=None):
        """
        Adds a nicely-formatted legend to the scatter plot.

        Parameters
        ----------
        ax: axis
                The axis to which to add the legend.
        clust_names: string array
                The legend entry names.
        colors: float array
                n x 4 array of the colors of the legend entries.
        loc: string (optional)
                The location of the legend.
        """
        ax.set_xlim(ax.get_xlim())
        ax.set_ylim(ax.get_ylim())

        hand = []
        for ii in range(0, len(clust_names)):
            h = plt.scatter(1e9, 1e9, c=[colors[ii, :]], s=10)
            hand = np.append(hand, h)
        if loc is not None:
            ax.legend(handles=hand.tolist(), labels=clust_names.tolist(), loc=loc)
        else:
            ax.legend(handles=hand.tolist(), labels=clust_names.tolist())

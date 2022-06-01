import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture as GMM
from sklearn.neighbors import KernelDensity

from missionbio.mosaic.assay import _Assay
from missionbio.mosaic.constants import COLORS, NORMALIZED_READS, READS
from missionbio.mosaic.plotting import plt, require_seaborn, sns
from missionbio.mosaic.utils import extend_docs


@extend_docs
class Protein(_Assay):
    """
    Container for Protein data

    Inherits most methods from :class:`missionbio.mosaic.assay._Assay`.
    See that for the documentation on other methods and visualizations.

    .. rubric:: Algorithms
    .. autosummary::

       get_signal_profile
       get_scaling_factor
       normalize_reads

    .. rubric:: Protein-specific visualizations
    .. autosummary::

       reads_to_ab
       cells_per_ab

    .. rubric:: Extended methods
    .. autosummary::

        run_umap
    """

    assay_color = COLORS[2]

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        args : list
            To be passed to missionbio.mosaic.assay object.

        kwargs : dict
            To be passed to missionbio.mosaic.assay object.
        """
        super().__init__(*args, **kwargs)

    def get_signal_profile(self, ab, attribute=None, features=None, bw=0.02, show_plot=False):
        """
        Identify the peaks and valleys in the antibody signal

        Parameters
        ----------
        ab : str
            The id of the antibdoy for which
            the profile is to be ascertained.
        attribute : str / np.array
            The layer to use to get the signal profile.
            Uses :meth:`_Assay.get_attribute` to retrieve
            the values constrained by `row+col`. If None,
            then the signal profile is calculated using
            the asinh normalizatioin.
        features : list-like
            Passed to :meth:`_Assay.get_attribute` if the
            attribute is not `None`
        bw : float
            The bandwidth used for fitting the kernel
            density on the signal profile.
        show_plot : bool
            Whether to show diagnostic plots or not.

        Returns
        -------
        float
            The raw count value of the peak or valley
            for the given antibodies
        """

        if attribute is None:
            read_counts = self.layers[READS].astype(float)
            read_counts += np.random.normal(0, 0.5, size=self.shape)
            expr = np.arcsinh(read_counts)[:, np.where(self.ids() == ab)].reshape(self.shape[0], 1)
        else:
            expr = self.get_attribute(attribute, constraint='row', features=features)
            expr = expr[ab].values[:, None]

        expr_norm = expr / np.max(expr)
        kde = KernelDensity(bandwidth=bw, kernel='gaussian')
        kde.fit(expr_norm)

        # score_samples returns the log of the probability density
        x_d = np.linspace(np.min(expr_norm), np.max(expr_norm), 1000)
        logprob = kde.score_samples(x_d[:, None])

        # Find peaks and valleys
        peaks = []
        valleys = []
        for i in range(len(x_d)):
            center = x_d[i]
            window = np.logical_and(center - bw / 2 < x_d, x_d < center + bw / 2)

            num_expr = np.logical_and(center - 2 * bw < expr_norm, expr_norm < center + 2 * bw).sum()
            if logprob[i] == np.max(logprob[window]):
                if (num_expr > 5):  # Greater than 5 cells needed to call a peak
                    peaks.append([center, np.exp(logprob[i])])
            if logprob[i] == np.min(logprob[window]):
                if (num_expr > 5):  # Greater than 5 cells needed to call a valley
                    valleys.append([center, np.exp(logprob[i])])

        peaks = pd.DataFrame(peaks, columns=['x', 'p'])
        peaks['kind'] = 'peak'
        peaks['x'] = peaks['x'] * np.max(expr)
        valleys = pd.DataFrame(valleys, columns=['x', 'p'])
        valleys['kind'] = 'valley'
        valleys['x'] = valleys['x'] * np.max(expr)

        # Plotting
        if show_plot:
            plt.figure(figsize=(10, 5))
            ax = plt.gca()
            ax.fill_between(x_d * np.max(expr), np.exp(logprob), alpha=0.5)
            ax.plot(expr_norm * np.max(expr), np.full_like(expr_norm, -0.01), '|k', markeredgewidth=1, alpha=0.2)

            ax.vlines(peaks['x'].values, *ax.get_ylim(), color='r', linewidth=2, linestyle='dashed')
            ax.vlines(valleys['x'].values, *ax.get_ylim(), color='g', linewidth=2, linestyle='dashed')
            ax.set_xlabel('Normalized expression')
            ax.set_ylabel('Distribution')
            ax.set_title(f'{self.title} - {ab}')

        return pd.concat([peaks, valleys])

    def get_scaling_factor(self):
        """
        Identifies the appropriate scaling factor for oversequenced runs.

        Returns
        -------
        float
            The factor to scale down the reads by.
        list
            The antibodies in which the peak amplification
            was observed at the identified scaling factor.
        """

        peaks = []
        for ab in self.ids():
            profile = self.get_signal_profile(ab)
            profile = profile[profile['kind'] == 'peak']
            peaks.append(np.array(profile['x'].values))

        # Filter for non-zero peaks
        peaks = [p[p > 1] for p in peaks]

        def get_max_bin(bins, peaks):
            binned_peaks = [np.digitize(p, bins=bins) for p in peaks]
            most_peaks = []
            for p in binned_peaks:
                most_peaks.extend(set(p))
            un, cnt = np.unique(most_peaks, return_counts=True)
            return(un[cnt == max(cnt)][0])

        # Find the bin where the most peaks are present
        binwidth = 1
        bins = np.arange(0, 20, binwidth)
        sub_peaks = peaks
        max_bin = get_max_bin(bins, peaks)
        max_vals = max([len(p) for p in sub_peaks])

        # Iteratively identify the bin where all antibodies show <= 1 peak
        while max_vals > 1:
            sub_peaks = [p[np.logical_and(bins[max_bin - 1] <= p, p < bins[max_bin])] for p in sub_peaks]
            max_vals = max([len(p) for p in sub_peaks])
            bins = np.linspace(bins[max_bin - 1], bins[max_bin], 3)
            max_bin = get_max_bin(bins, sub_peaks)

        # Find the average of the peaks and call it the scaling factor
        scale = 0
        ab = []
        for i in range(len(sub_peaks)):
            if (len(sub_peaks[i]) > 0):
                scale += sub_peaks[i][0]
                ab.append(self.ids()[i])

        scale = np.sinh(scale / len(ab))

        return scale, ab

    def normalize_reads(self, method='CLR', jitter=0.5, scale=None, show_plot=False):
        """
        Normalize read counts.

        This adds `normalized_counts` to the assay layers.

        Parameters
        ----------
        method : str
            CLR, asinh, or NSP which stand for
            Centered Log Ratio, Inverse Hyperbolic
            transformation and Noise corrected and Scaled
            Protein counts respectively
        jitter : float
            The standard deviation of the jitter to be added
            to the read counts before applying the normalization.
            The jitter is sampled from a normal distribution
            cenetered at 0. This is only applicable for NSP and asinh
        scale : float
            The amount by which the read counts are scaled down.
            This is applicable only for NSP. If 'None' then the
            algorithm tries to estimate it from the read distribution.
        show_plot : bool
            Whether to show diagnostic plots for NSP or not.

        Raises
        ------
        Exception
            When one of the supported methods is not provided.
        """

        read_counts = self.layers[READS].copy().astype(float)

        if method == 'NSP':
            # Find a scaling factor
            if scale is None:
                print('Estimating scaling factor', end='')
                scale, ab = self.get_scaling_factor()
                if len(ab) > len(self.ids()) / 2:
                    print(f' - Scaling down by {scale:.2f}')
                else:
                    print(' - Could not confidently find a scaling factor. No scaling applied.')
                    scale = 1

            # Normalize the read counts
            jitter = np.random.normal(loc=0, scale=jitter, size=self.shape)
            normal_counts = np.arcsinh((self.layers[READS] / scale) + jitter)

            reads = np.log10(self.layers[READS].sum(axis=1))

            # Use a Gaussian Mixture Model on each cell to identify its background and signal means
            means = []
            gmm = GMM(n_components=2)
            for i in range(normal_counts.shape[0]):
                gmm.fit(normal_counts[i, :].reshape(-1, 1))
                means.append(gmm.means_[:, 0])

            means = np.array(means)
            background = np.min(means, axis=1)
            signal = np.max(means, axis=1)

            # Find linear fits for the background and signal
            f_sig = np.poly1d(np.polyfit(reads, signal, 1))
            f_back = np.poly1d(np.polyfit(reads, background, 1))

            # Find the factor for the antibodies
            ab_factor = (self.layers[READS] > 0).sum(axis=0) / self.shape[0]
            cell_factor = f_back(reads)[:, None]

            # Correct read depth dependence such that mean background is at 0 and mean signal is at 1
            normal_counts = normal_counts - cell_factor * ab_factor
            normal_counts = normal_counts / np.maximum((f_sig(reads) - f_back(reads)), 1)[:, None]

            # Plotting
            if show_plot:
                try:
                    require_seaborn()

                    sns.set(style='whitegrid', font_scale=1.2)

                    fig, axs = plt.subplots(1, 2, figsize=(30, 10))
                    sns.scatterplot(reads, background, ax=axs[0])
                    sns.scatterplot(reads, signal, ax=axs[0])

                    # Apply transformation for signal and background
                    background = background - f_back(reads)
                    background = background / np.maximum((f_sig(reads) - f_back(reads)), 1)
                    signal = signal - f_back(reads)
                    signal = signal / np.maximum((f_sig(reads) - f_back(reads)), 1)

                    sns.scatterplot(reads, background, ax=axs[1])
                    sns.scatterplot(reads, signal, ax=axs[1])
                    sns.lineplot(reads[reads.argsort()], f_back(reads[reads.argsort()]), ax=axs[0], color=COLORS[20])
                    sns.lineplot(reads[reads.argsort()], f_sig(reads[reads.argsort()]), ax=axs[0], color=COLORS[20])

                    axs[0].set_ylabel('Mean expression')
                    axs[1].set_ylabel('Corrected mean expression')
                    axs[0].set_xlabel('log$_{10}$(Total reads)')
                    axs[1].set_xlabel('log$_{10}$(Total reads)')
                    axs[0].set_title('Mean of the signal and background from the GMM on each cell')
                    axs[1].set_title('Corrected signal and background')
                    fig.suptitle('Correction for read depth dependence')
                    plt.show()
                except ValueError:
                    pass

        elif method == 'CLR':
            normal_counts = np.log1p(read_counts)
            normal_counts = normal_counts - normal_counts.mean(axis=1)[:, None]
        elif method == 'asinh':
            read_counts += np.random.normal(0, jitter, size=self.shape)
            normal_counts = np.arcsinh(read_counts)
        else:
            raise Exception(f"Please provide one of {['CLR', 'NSP', 'asinh']}")

        self.add_layer(NORMALIZED_READS, normal_counts)

    # --------- Tertiary analysis methods

    def run_umap(self, **kwargs):
        """
        Extends :meth:`missionbio.mosaic.assay._Assay.run_umap`

        Sets the default values for the dna UMAP.
        n_neighbors=50, metric='cosine', min_dist=0
        """

        defaults = dict(n_neighbors=50,
                        metric='cosine',
                        min_dist=0)

        for key in defaults:
            if key not in kwargs:
                kwargs[key] = defaults[key]

        super().run_umap(**kwargs)

    # --------- Plotting

    def reads_to_ab(self, **kwargs):
        """
        Violin plot for reads taken by each antibody.

        Parameters
        ----------
        kwargs : dict
            Passed to violinplot.

        Returns
        -------
        fig : plotly.graph_objects.Figure

        See also
        --------
        :meth:`missionbio.mosaic.assay._Assay.violinplot`
        """

        ab_reads = np.log10(self.layers[READS] + 1)
        order = np.lexsort((np.median(ab_reads, axis=0), np.mean(ab_reads, axis=0)))
        id_order = self.ids()[order][::-1]

        self.add_layer('_ab_reads', ab_reads)

        fig = self.violinplot(attribute='_ab_reads', **kwargs)
        if 'features' not in kwargs:
            fig.update_layout(xaxis_categoryarray=id_order)

        self.del_layer('_ab_reads')

        return fig

    @require_seaborn
    def cells_per_ab(self, title='', **kwargs):
        """
        Bar plot of non-zero read cells to antibodies.

        Parameters
        ----------
        title : str
            Appended to the name of the sample in the title.
        kwargs : dict
            Passed to the seaborn barplot method.

        Returns
        -------
        ax : matplotlib.pyplot.axis
        """

        ab_reads = self.layers[READS].astype(bool).sum(axis=0)
        ab_names = self.ids()

        data = pd.DataFrame([ab_reads, ab_names], index=['reads', 'ab']).T
        data.reads = data.reads.astype(int)
        data = data.sort_values(by='reads', ascending=False)
        num_ab = self.ids().shape[0]

        sns.set(style='whitegrid', font_scale=1.5)
        plt.figure(figsize=(2 * num_ab, 10))

        ax = sns.barplot(data=data, x='ab', y='reads', color=self.assay_color, **kwargs)
        ax.hlines(self.shape[0], *ax.get_xlim())

        ax.set_xticklabels(ax.get_xticklabels(), fontsize=16, rotation=90)
        ax.set_ylabel('Number of cells', fontsize=19)
        ax.set_xlabel('')

        if title == '':
            title = self.title
        ax.set_title(title, fontsize=19)

        return ax

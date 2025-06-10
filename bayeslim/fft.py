"""
Module for Fourier transforms and related tools
"""
import torch
import numpy as np
from scipy.signal import windows

from . import dataset, utils


class FFT(utils.Module):
    """
    A 1D FFT block for tensors, VisData, MapData or CalData
    """
    def __init__(self, dim=0, abs=False, peaknorm=False, N=None, dx=None,
                 ndim=None, window=None, fftshift=True, ifft=False, norm=None,
                 edgecut=None, square=False, device=None, **kwargs):
        """
        Parameters
        ----------
        dim : int, optional
            Dimension to take FFT.
        abs : bool, optional
            Take abs after FFT
        peaknorm : bool, optional
            Peak normalize after FFT along dim
        N : int, optional
            Number of channels along FFT dim, used
            for computing fftfreqs and when using a window
        dx : float, optional
            Channel spacing along dim, used for fftfreqs
        ndim : int, optional
            Total dimensionality of input tensors, required
            for window
        window : str, optional
            Windowing to use before FFT across dim.
        fftshift : bool, optional
            if True, fftshift along dim after fft.
        ifft : bool, optional
            If True use the ifft instead of fft. If fftshift
            is set to True, then perform ifftshift *BEFORE* ifft()
        norm : str, optional
            The FFT convention. 'forward', 'backward', or 'ortho'.
            Default is torch.fft.fft default.
        edgecut : tuple of int, optional
            Number of channels to give zero weight on either end of
            the FFT (edgecut_start, edgecut_end). If providing
            a window, this ensures the window smoothly connects
            to the edgecut (not the same as only giving zero weight
            to edge channels).
        square : bool, optional
            If True, take abs(fft)**2 before output
        kwargs : dict, optional
            Kwargs to pass to gen_window()
        """
        super().__init__()
        self.dim = dim
        self.abs = abs
        self.peaknorm = peaknorm
        self.dx = dx if dx is not None else 1.0
        self.fftshift = fftshift
        self.ifft = ifft
        self.norm = norm
        self.square = square
        if N is not None:
            self.freqs = torch.fft.fftfreq(N, d=self.dx)
            if fftshift:
                self.freqs = torch.fft.fftshift(self.freqs)
            self.start = self.freqs[0]
            self.df = self.freqs[1] - self.freqs[0]
        else:
            self.start = 0.0
            self.dx, self.freqs, self.df = None, None, None
        if isinstance(edgecut, (int, np.integer)):
            edgecut = (edgecut, edgecut)
        elif edgecut is None:
            edgecut = (0, 0)
        self.edgecut = edgecut
        self.win = None
        if window is not None:
            assert N is not None
            assert ndim is not None
            Nwin = N - self.edgecut[0] - self.edgecut[1]
            win = gen_window(window, Nwin, **kwargs)
            win = torch.cat([torch.zeros(self.edgecut[0]),
                             win,
                             torch.zeros(self.edgecut[1])])
            shape = [1 for i in range(ndim)]
            shape[dim] = N
            self.win = win.reshape(*shape)

        if device is not None:
            self.push(device)

    def forward(self, inp, ifft=None, win=None, **kwargs):
        """
        Take the FFT of the inp and return
        """
        if isinstance(inp, np.ndarray):
            inp = torch.as_tensor(inp)

        elif isinstance(inp, (dataset.VisData, dataset.CalData, dataset.MapData)):
            out = inp.copy()
            out.data = self.forward(inp.data, **kwargs)
            return out

        win = win if win is not None else self.win
        if win is not None:
            inp = inp * win

        ifft = ifft if ifft is not None else self.ifft

        if self.fftshift and ifft:
            inp = torch.fft.ifftshift(inp, dim=self.dim)

        if ifft:
            inp_fft = torch.fft.ifft(inp, norm=self.norm, dim=self.dim)
        else:
            inp_fft = torch.fft.fft(inp, norm=self.norm, dim=self.dim)

        if self.fftshift and not ifft:
            inp_fft = torch.fft.fftshift(inp_fft, dim=self.dim)

        if self.abs:
            inp_fft = torch.abs(inp_fft)

        if self.peaknorm:
            inp_fft = inp_fft / torch.max(torch.abs(inp_fft), dim=self.dim, keepdim=True).values

        if self.square:
            inp_fft = torch.abs(inp_fft)**2

        return inp_fft

    def push(self, device):
        """
        Push self to device
        """
        self.win = self.win.to(device)


class PeakDelay(FFT):
    """
    Compute peak delay across dim, using
    Quinn's 2nd estimator
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def k(self, x):
        return 0.25 * torch.log(3 * x**2 + 6 * x + 1) \
                - np.sqrt(6) / 24 \
                * torch.log((x + 1 - np.sqrt(2./3.)) / (x + 1 + np.sqrt(2./3.)))

    def get_peak(self, y):
        """
        Use Quinn 2nd estimator to get peak ybin
        """
        argmax = torch.argmax(torch.abs(y))
        argmax_pos = argmax + 1 if argmax != len(y) - 1 else 0
        argmax_neg = argmax - 1 if argmax != 0 else -1
        cast = torch.real if torch.is_complex(y) else torch.as_tensor
        rpos = cast(y[argmax_pos] / y[argmax])
        rneg = cast(y[argmax_neg] / y[argmax])
        dpos = -rpos / (1 - rpos)
        dneg = rneg / (1 - rneg)
        max_bin = argmax + ((dneg + dpos) / 2 + self.k(dneg**2) - self.k(dpos**2))

        return self.start + max_bin * self.df

    def _iter_peak(self, inp, dim, out):
        if inp.ndim == 1:
            # estimate peak
            out[:] = self.get_peak(inp)
        else:
            # iterate
            for i in range(len(inp)):
                self._iter_peak(inp[i], dim+1, out[i])

    def forward(self, inp):

        if isinstance(inp, (dataset.VisData, dataset.MapData)):
            out = inp.copy()
            out.data = self.forward(inp.data)
            return out

        # take fft
        inp = super().forward(inp)

        # iterate over all dims
        shape = list(inp.shape)
        shape[self.dim] = 1
        out = torch.zeros(shape, dtype=utils._float())
        out = out.moveaxis(self.dim, -1)
        self._iter_peak(inp, 0, out)
        out = out.moveaxis(-1, self.dim)

        return out


def vis_wedge(vd, ravg_kwgs=None, **kwargs):
    """
    Given a VisData object, take its FFT along frequency
    and average redundant groups to form a wedge

    Parameters
    ----------
    vd : VisData object
    ravg_kwgs : dict, optional
        Keyword arguments to pass to vd.bl_average()
        for redundant averaging.
    kwargs : dict, optional
        Additional keyword args to send to fft.FFT(**kwargs)

    Returns
    -------
    VisData
        Red-averaged and FFT'd data
    FFT object
        FFT object holding Fourier modes as FFT.freqs
    """
    # average redundancies
    ravg_kwgs = ravg_kwgs if ravg_kwgs is not None else {}
    vd = vd.bl_average(inplace=False, **ravg_kwgs)

    # setup FT object
    dfreq = vd.freqs[1] - vd.freqs[0]
    Nfreqs = vd.Nfreqs
    FT = FFT(dim=4, ndim=5, dx=dfreq, N=Nfreqs, **kwargs)

    # take FT
    vd = FT(vd)

    return vd, FT


def gen_window(window, N, alpha=None, **kwargs):
    """
    Generate a window function of len N

    Parameters
    ----------
    window : str
        window function
    N : int
        number of channels
    alpha : float, optional
        alpha parameter for tukey window,
        std parameter for gaussian window

    Returns
    -------
    tensor
        window function
    """
    if window in ['none', None, 'None', 'boxcar', 'tophat']:
        w = windows.boxcar(N)
    elif window in ['blackmanharris', 'blackman-harris', 'bh', 'bh4']:
        w = windows.blackmanharris(N)
    elif window in ['hanning', 'hann']:
        w =  windows.hann(N)
    elif window == 'tukey':
        w =  windows.tukey(N, alpha=alpha, **kwargs)
    elif window == 'gaussian':
        w = windows.gaussian(N, std=alpha, **kwargs)
    elif window in ['blackmanharris-7term', 'blackman-harris-7term', 'bh7']:
        # https://ieeexplore.ieee.org/document/293419
        a_k = [0.27105140069342, 0.43329793923448, 0.21812299954311, 0.06592544638803, 0.01081174209837,
              0.00077658482522, 0.00001388721735]
        w = windows.general_cosine(N, a_k, True)
    elif window in ['cosinesum-9term', 'cosinesum9term', 'cs9']:
        # https://ieeexplore.ieee.org/document/940309
        a_k = [2.384331152777942e-1, 4.00554534864382e-1, 2.358242530472107e-1, 9.527918858383112e-2,
               2.537395516617152e-2, 4.152432907505835e-3, 3.68560416329818e-4, 1.38435559391703e-5,
               1.161808358932861e-7]
        w = windows.general_cosine(N, a_k, True)
    elif window in ['cosinesum-11term', 'cosinesum11term', 'cs11']:
        # https://ieeexplore.ieee.org/document/940309
        a_k = [2.151527506679809e-1, 3.731348357785249e-1, 2.424243358446660e-1, 1.166907592689211e-1,
               4.077422105878731e-2, 1.000904500852923e-2, 1.639806917362033e-3, 1.651660820997142e-4,
               8.884663168541479e-6, 1.938617116029048e-7, 8.482485599330470e-10]
        w = windows.general_cosine(N, a_k, True)
    else:
        try:
            # return any single-arg window from windows
            w  = getattr(windows, window)(N, **kwargs)
        except AttributeError:
            raise ValueError("Didn't recognize window {}".format(window))

    w = torch.as_tensor(w, dtype=utils._float())

    return w

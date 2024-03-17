import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from numpy import linalg

def psd(trials, sfreq):
    """
    Compute the Power Spectral Density (PSD) for each trial.
    
    Parameters
    ----------
    trials : 3d-array (channels x samples x trials)
        The EEG data for all the trials.

    Returns
    -------
    trials_PSD : 3d-array (channels x PSD x trials)
        The PSD data for all the trials.
    freqs : list of floats
        The frequencies for which the PSD was computed (useful for plotting later)
    """

    # Define the lower and upper bounds of the frequencies of interest
    # You can also specify the frequencies in Hz directly (e.g. [0, 4, 8, 12, 30, 100])
    freqs = np.linspace(0, sfreq / 2, 129)  # define frequencies of interest based on sampling frequency In frequency analysis, the highest frequency of interest is the Nyquist frequency, which is half of the sampling frequency.
    ntrials = trials.shape[2]  # get number of trials
    nchannels = trials.shape[0]  # get number of channels
    trials_PSD = np.zeros((nchannels, len(freqs), ntrials))  # to store the power spectral density

    # Iterate over trials and channels
    for trial in range(ntrials):
        for ch in range(nchannels):
            # Compute the PSD
            freq, PSD = signal.welch(trials[ch, :, trial], sfreq, nperseg=256)
            # Store the results
            trials_PSD[ch, :, trial] = PSD

    return trials_PSD, freq

# Plot the PSD
def plot_PSD(psd_all, freqs, chan_names, cl):
    """
    Plot the power spectral density (PSD) for a specific channel.
    The plot is done for only two channels C3 and C4.

    Parameters
    ----------
    psd_all : dict
        The PSD data for each class. The keys are the class names.
    freqs : list of floats
        The frequencies for which the PSD was computed.
    """
    channels_ind = [chan_names.index('C3'), chan_names.index('C4')]  # C3 and C4 channels
    class_labels = [cl[0], cl[1]]  # Two different class labels

    # Create a new figure
    plt.figure(figsize=(10, 4))

    # Loop over the channels
    for i, ch_ind in enumerate(channels_ind):
        plt.subplot(1, 2, i + 1)
        # Loop over the classes
        for cl, psd in psd_all.items():
            # Compute the mean PSD over trials
            mean_PSD = np.mean(psd[ch_ind, :, :], axis=1)
            # Plot the PSD
            plt.plot(freqs, mean_PSD, label=cl)

        # Add labels and grid
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density (dB)')
        plt.title('PSD for channel {}'.format(chan_names[ch_ind]))
        plt.xlim(0,40)
        plt.ylim(0, 500)
        plt.legend()
        plt.grid(True)

def butter_bandpass(trials, lowcut, highcut, fs, nsamples, order=5):
    """
    Design a band-pass filter using the Butterworth method.

    Parameters
    ----------
    trials : 3d-array (channels x samples x trials)
        The EEGsignal for one class.
    lowcut : float
        The lower cut-off frequency of the band-pass filter.
    highcut : float
        The higher cut-off frequency of the band-pass filter.

    fs : float
        The sampling frequency of the signal.
    
    order : int
        The order of the filter.
    
    Returns
    -------
    trials_filt : 3d-array (channels x samples x trials)
        The band-pass filtered signal for one class.
    """
    from scipy.signal import butter, lfilter
    
    nchannels = trials.shape[0]  # get number of channels

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    ntrials = trials.shape[2]
    trials_filt = np.zeros((nchannels, nsamples, ntrials))
    for i in range(ntrials):
        trials_filt[:,:,i] = lfilter(b, a, trials[:,:,i], axis=1)

    return trials_filt

def logvar(trials):
    """
    Calculates the log-variance of each channel.

    Parameters
    ----------
    trials : 3d-array (channels x samples x trials)
        The EEGsignal

    Returns
    -------
    feat : 2d-array (channels x trials)
        The log-variance features for each channel 
    """
    return np.log(np.var(trials, axis=1))

def plot_logvar(trials, cl_lab, cl1, cl2, nchannels):
    """
    Plots the log-variance features for each channel as a bar chart.
    
    Parameters
    ----------
    trials : 2d-array (channels x trials)
        The EEGsignal
    
    """
    plt.figure(figsize=(12,5))
    
    x0 = np.arange(nchannels)
    x1 = np.arange(nchannels) + 0.4

    y0 = np.mean(trials[cl1], axis=1)
    y1 = np.mean(trials[cl2], axis=1)

    plt.bar(x0, y0, width=0.5, color='b')
    plt.bar(x1, y1, width=0.4, color='orange')

    plt.xlim(-0.5, nchannels+0.5)

    plt.gca().yaxis.grid(True)
    plt.title('Log-variance features for each channel')
    plt.xlabel('Channels')
    plt.ylabel('Log-variance')
    plt.legend(cl_lab)

def scatter_logvar(logvar_trials, cl_lab, chan_ind):
    """
    Scatter plot of the log-variance features for 2 channels..
    Each data point on the plot represents the log-variance values of C3 and C4 for a single trial.
    X-Axis: The x-axis represents the log-variance of C3 for each trial. So, the horizontal position of a data point indicates the log-variance value of C3 for that trial.

    Y-Axis: The y-axis represents the log-variance of C4 for each trial. So, the vertical position of a data point indicates the log-variance value of C4 for that trial.

    If I choose to include more than two channels in the scatter plot, each additional channel will be represented by an additional dimension on the plot.

    Parameters
    ----------
    logvar_trials : dict
        The log-variance features for each class
    cl_lab : list of str
        The class labels
    chan_names : list of str
        The channel names

    Returns
    -------
    Scatter plot of the log-variance features for 2 channels.
    """
    # channels_ind = [chan_names.index('C3'), chan_names.index('C4')]  # C3 and C4 channels
    channels_ind = chan_ind
    cl1 = cl_lab[0]
    cl2 = cl_lab[1]

    plt.figure(figsize=(6,4))
    plt.scatter(logvar_trials[cl1][channels_ind[0],:], logvar_trials[cl1][channels_ind[1],:], label=cl1)
    plt.scatter(logvar_trials[cl2][channels_ind[0],:], logvar_trials[cl2][channels_ind[1],:], label=cl2)
    plt.xlabel('First Component')
    plt.ylabel('Last Component')
    plt.title('Scatter Plot of the log-variance features')
    plt.legend()
    plt.show()

def cov(trials, nsamples):
    """
    Calculates the covariance for each trial and return their average.

    Parameters
    ----------
    trials : 3d-array (channels x samples x trials)
        The EEG data for all the trials.

    Returns
    -------
    cov_matrix : 2d-array
        The covariance matrix
    """
    ntrials = trials.shape[2]
    covs = [ trials[:,:,i].dot(trials[:,:,i].T)/ nsamples for i in range(ntrials) ]
    
    return np.mean(covs, axis=0)

def whitening(sigma):
    """ calculate whitening matrix for covariance matrix sigma. """

    U, l, _ = linalg.svd(sigma)

    return U.dot(np.diag(l ** -0.5))

def csp(trials_r, trials_l, nsamples):
    """
    Calculates the CSP transformation matrix W.

    Parameters
    ----------
    trials_r, trials_l : 3d-arrays (channels x samples x trials)
        The EEGsignal for right and left hand

    Returns
    -------
    W : 2d-array
        The CSP transformation matrix
    """
    # CSP requires the estimation of the covariance matrix for each class
    cov_r = cov(trials_r, nsamples)
    cov_l = cov(trials_l, nsamples)

    P = whitening(cov_r + cov_l)
    B, _, _ = linalg.svd(P.T.dot(cov_l).dot(P))
    
    W = P.dot(B)

    return W

def apply_mix(W, trials, nchannels, nsamples):
    """
    Apply a mixing matrix to each trial (basically multiply W with the EEG signal matrix)
    """
    ntrials = trials.shape[2]
    trials_csp = np.zeros((nchannels, nsamples, ntrials))
    for i in range(ntrials):
        trials_csp[:,:,i] = W.T.dot(trials[:,:,i])

    return trials_csp
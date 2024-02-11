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
        plt.xlim(0,30)
        plt.ylim(0, 350)
        plt.legend()
        plt.grid(True)

def rms(trials):
    """
    Calculates the Root Mean Square (RMS) of each channel.

    Parameters
    ----------
    trials : 3d-array (channels x samples x trials)
        The EEG signal

    Returns
    -------
    rms_values : 2d-array (channels x trials)
        The RMS values for each channel 
    """
    return np.sqrt(np.mean(trials**2, axis=1))

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

def std(trials):
    """
    Calculates the standard deviation of each channel.

    Parameters
    ----------
    trials : 3d-array (channels x samples x trials)
        The EEG signal

    Returns
    -------
    feat : 2d-array (channels x trials)
        The standard deviation features for each channel 
    """
    return np.std(trials, axis=1)

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

def plot_rms(trials, cl_lab, cl1, cl2, nchannels):

    """
    Plots the RMS features for each channel as a bar chart.

    Parameters
    ----------
    trials : 2-d array (channels x trials)
        The RMS values for each channel

    """
    plt.figure(figsize=(12, 5))

    x0 = np.arange(nchannels)
    x1 = np.arange(nchannels) + 0.4

    y0 = np.mean(trials[cl1], axis=1)
    y1 = np.mean(trials[cl2], axis=1)

    plt.bar(x0, y0, width=0.5, color='b')
    plt.bar(x1, y1, width=0.4, color='orange')

    plt.xlim(-0.5, nchannels+0.5)

    plt.gca().yaxis.grid(True)
    plt.title('RMS features for each channel')
    plt.xlabel('Channels')
    plt.ylabel('RMS')
    plt.legend(cl_lab) 

def plot_std(trials, cl_lab, cl1, cl2, nchannels):
    """
    Plots the standard deviation features for each channel as a bar chart.

    Parameters
    ----------
    trials : 2-d array (channels x trials)
        The standard deviation values for each channel

    """
    plt.figure(figsize=(12, 5))

    x0 = np.arange(nchannels)
    x1 = np.arange(nchannels) + 0.4

    y0 = np.mean(trials[cl1], axis=1)
    y1 = np.mean(trials[cl2], axis=1)

    plt.bar(x0, y0, width=0.5, color='b')
    plt.bar(x1, y1, width=0.4, color='orange')

    plt.xlim(-0.5, nchannels+0.5)

    plt.gca().yaxis.grid(True)
    plt.title('Standard deviation features for each channel')
    plt.xlabel('Channels')
    plt.ylabel('Standard deviation')
    plt.legend(cl_lab)

def scatter_logvar(logvar_trials, cl_lab, chan_ind):
    """
    Scatter plot of the log-variance features for 2 channels (C3 and C4).
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
    Scatter plot of the log-variance features for 2 channels (C3 and C4).
    """
    # channels_ind = [chan_names.index('C3'), chan_names.index('C4')]  # C3 and C4 channels
    channels_ind = chan_ind
    cl1 = cl_lab[0]
    cl2 = cl_lab[1]

    plt.figure(figsize=(6,4))
    plt.scatter(logvar_trials[cl1][channels_ind[0],:], logvar_trials[cl1][channels_ind[1],:], label=cl1)
    plt.scatter(logvar_trials[cl2][channels_ind[0],:], logvar_trials[cl2][channels_ind[1],:], label=cl2)
    plt.xlabel('Log-variance of C3')
    plt.ylabel('Log-variance of C4')
    plt.title('Scatter plot of the log-variance features')
    plt.legend()
    plt.show()

def scatter_rms(rms_trials, cl_lab, chan_ind):
    """
    Scatter plot of the RMS features for 2 channels (C3 and C4).
    Each data point on the plot represents the RMS values of C3 and C4 for a single trial.
    X-Axis: The x-axis represents the RMS of C3 for each trial. So, the horizontal position of a data point indicates the RMS value of C3 for that trial.

    Y-Axis: The y-axis represents the RMS of C4 for each trial. So, the vertical position of a data point indicates the RMS value of C4 for that trial.

    If I choose to include more than two channels in the scatter plot, each additional channel will be represented by an additional dimension on the plot.

    Parameters
    ----------
    rms_trials : dict
        The RMS features for each class
    cl_lab : list of str
        The class labels
    chan_names : list of str
        The channel names

    Returns
    -------
    Scatter plot of the RMS features for 2 channels (C3 and C4).
    """
    # channels_ind = [chan_names.index('C3'), chan_names.index('C4')]  # C3 and C4 channels
    channels_ind = chan_ind
    cl1 = cl_lab[0]
    cl2 = cl_lab[1]

    plt.figure(figsize=(6,4))
    plt.scatter(rms_trials[cl1][channels_ind[0],:], rms_trials[cl1][channels_ind[1],:], label=cl1)
    plt.scatter(rms_trials[cl2][channels_ind[0],:], rms_trials[cl2][channels_ind[1],:], label=cl2)
    plt.xlabel('RMS of C3')
    plt.ylabel('RMS of C4')
    plt.title('Scatter plot of the RMS features')
    plt.legend()
    plt.show()

def scatter_std(std_trials, cl_lab, chan_ind):
    """
    Scatter plot of the standard deviation features for 2 channels (C3 and C4).
    Each data point on the plot represents the standard deviation values of C3 and C4 for a single trial.
    X-Axis: The x-axis represents the standard deviation of C3 for each trial. So, the horizontal position of a data point indicates the standard deviation value of C3 for that trial.

    Y-Axis: The y-axis represents the standard deviation of C4 for each trial. So, the vertical position of a data point indicates the standard deviation value of C4 for that trial.

    If I choose to include more than two channels in the scatter plot, each additional channel will be represented by an additional dimension on the plot.

    Parameters
    ----------
    std_trials : dict
        The standard deviation features for each class
    cl_lab : list of str
        The class labels
    chan_names : list of str
        The channel names

    Returns
    -------
    Scatter plot of the standard deviation features for 2 channels (C3 and C4).
    """
    # channels_ind = [chan_names.index('C3'), chan_names.index('C4')]  # C3 and C4 channels
    channels_ind = chan_ind
    cl1 = cl_lab[0]
    cl2 = cl_lab[1]

    plt.figure(figsize=(6,4))
    plt.scatter(std_trials[cl1][channels_ind[0],:], std_trials[cl1][channels_ind[1],:], label=cl1)
    plt.scatter(std_trials[cl2][channels_ind[0],:], std_trials[cl2][channels_ind[1],:], label=cl2)
    plt.xlabel('Standard deviation of C3')
    plt.ylabel('Standard deviation of C4')
    plt.title('Scatter plot of the standard deviation features')
    plt.legend()
    plt.show()

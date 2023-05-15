import numpy as np

def neurons_in_pc(weights):
    stdw = np.zeros(weights.shape[0])
    for i_sv in np.arange(weights.shape[0]):
        weights_sorted = np.sort(np.abs(weights[i_sv]))[::-1]

        # Compute an estimate of the number of the neurons involved in each
        # PC: Make a sorted bar plot of the absolute values of the weights,
        # compute the standard deviation of the bar plot, and then divide 
        # by the total number of neurons in the recording.
        avgw = np.sum(weights_sorted*np.arange(len(weights_sorted))) / np.sum(weights_sorted)
        stdw[i_sv] = np.sqrt(np.sum((weights_sorted-avgw)**2)/len(weights_sorted))
        # Normalize by number of weights
        stdw[i_sv] /= len(weights_sorted)
    
    return stdw

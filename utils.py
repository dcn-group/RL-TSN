import matplotlib.pyplot as plt
import numpy as np


def plotLearning(scores, number, filename, x=None, window=5):
    #N = len(number)
    #running_avg = np.empty(N)
    #for t in range(N):
    #    running_avg[t] = np.mean(scores[max(0, t - window):(t + 1)])
    #if x is None:
    #    x = [i for i in range(N)]
    plt.ylabel('Reward')
    plt.xlabel('Testing Step')
    #plt.plot(x, running_avg)
    plt.plot(number, scores)
    plt.savefig(filename)
    plt.show()

    #plt.plot(x,running_avg)
    #plt.ylabel('Score2')
    #plt.xlabel('Game')
    #plt.show()

def plot_figure(data, xlabel, ylabel, filename):
    plt.figure()
    plt.ylim([0, 105])
    plt.cla()
    plt.plot(range(len(data)), data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(f"{filename}", format="png")
    plt.close()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys


def plot_computations(folder_name, nb_computations, alpha, process_avg_over=20, confint=True, ra=True, save_figure=True):
    """
    Plot the bootstrapped mean and associated confidence interval at alpha% over rolling avg of the nb ocomputations chosen in the selected folder 
    """
    path = 'Results/' + folder_name + "/"
    data = []
    for i in range(1, nb_computations + 1):
        df = pd.read_csv(path + "comp%d.csv" % i, index_col=None, sep=';')
        df.sort_values(by=['timestamps'], inplace=True)  # sort in good order...
        df["training_score_ra"] = df["training_score"].rolling(window=process_avg_over, min_periods=1).mean()
        df["testing_score_ra"] = df["testing_score"].rolling(window=process_avg_over, min_periods=1).mean()
        df.index += 1
        df.index.names = ["ep"]
        data.append(df)

    alldf = pd.concat(data, axis=1, keys=range(len(data)))
    keydf = alldf.swaplevel(0, 1, axis=1).groupby(level=0, axis=1)
    meandf = keydf.mean()
    lower_df = keydf.quantile((1 - alpha) / 2)
    upper_df = keydf.quantile(1 - (1 - alpha) / 2)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2)
    if ra:
        ax1.plot(meandf.index, meandf['training_score_ra'], c='b', linewidth=1, label='Train Score RA')
        ax1.plot(meandf.index, meandf['testing_score_ra'], c='g', linewidth=1, label='Test Score RA')
        if confint:
            ax1.fill_between(meandf.index, lower_df['training_score_ra'], upper_df['training_score_ra'], color='b', alpha=0.2, label='%.2f%% CI Train RA' % alpha)
            ax1.fill_between(meandf.index, lower_df['testing_score_ra'], upper_df['testing_score_ra'], color='g', alpha=0.2, label='%.2f%% CI Test RA' % alpha)
    else:
        ax1.plot(meandf.index, meandf['training_score'], c='b', linewidth=1, label='Train Score')
        ax1.plot(meandf.index, meandf['testing_score'], c='g', linewidth=1, label='Test Score')
        if confint:
            ax1.fill_between(meandf.index, lower_df['training_score'], upper_df['training_score'], color='b', alpha=0.2, label='%.2f%% CI Train' % alpha)
            ax1.fill_between(meandf.index, lower_df['testing_score'], upper_df['testing_score'], color='g', alpha=0.2, label='%.2f%% CI Test' % alpha)
    ax1.set(xlabel='Episodes', ylabel='Score per Episode')
    ax2.plot(meandf.index, meandf['timestamps'], c='b', linewidth=1, label='Time')
    if confint:
        ax2.fill_between(meandf.index, lower_df['timestamps'], upper_df['timestamps'], color='b', alpha=0.2, label='%.2f%% CI Time' % alpha)
    ax2.set(xlabel='Episodes', ylabel='Time (s)')
    fig.suptitle('Training and Testing Performances bootstraped over %d computations, with %.2f CI' % (nb_computations, alpha))
    ax1.legend(loc='upper left', prop={'size': 9})
    ax2.legend(loc='upper left', prop={'size': 9})
    if save_figure:
        plt.savefig(path + 'bootstrapped_mean_and_CI', dpi=500)
    plt.show()

def plot_first_over(folder_names, nb_computations, testing_or_training='testing', first_over=150, process_avg_over=20, display_mean=False, save_figure=True):
    """
    For each folder, plot the ep (x) and timestamps (y) at which the rolling avg for each computations first reached selected "first_over" score.
    Possible to plot the mean point and associated std
    """
    # cmap = plt.get_cmap('jet_r')
    # colors = cmap(np.arange(len(folder_names)) /  len(folder_names))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    max_ep = 0
    plt.figure()
    for folder_name, c in zip(folder_names, colors):
        path = 'Results/' + folder_name + "/"
        data = []
        for i in range(1, nb_computations + 1):
            df = pd.read_csv(path + "comp%d.csv" % i, index_col=None, sep=';')
            df.sort_values(by=['timestamps'], inplace=True)  # sort in good order...
            df["training_score_ra"] = df["training_score"].rolling(window=process_avg_over, min_periods=1).mean()
            df["testing_score_ra"] = df["testing_score"].rolling(window=process_avg_over, min_periods=1).mean()
            df['ep'] = df.index + 1
            max_ep = max(len(df.index), max_ep)
            df = df.loc[df['testing_score_ra'] >= first_over].head(1)
            data.append(df)
        alldf = pd.concat(data, keys=range(len(data)), ignore_index=True)
        mean_df = alldf.mean()
        std_df = alldf.std()
        # plot
        plt.scatter(alldf['ep'], alldf['timestamps'], color=c, s=15, alpha=0.5, label=folder_name)
        if display_mean:
            plt.errorbar(mean_df['ep'], mean_df['timestamps'], xerr=std_df['ep'], yerr=std_df['timestamps'], color = c, fmt='o', elinewidth=.5, capsize=2, label='Mean with std')
    
    plt.xlabel('Episodes')
    plt.ylabel('Time (s)')
    plt.xlim([0, max_ep])
    plt.suptitle('{} Performance bootstrapped over {} computations: first time each computation reached a reward of at least {}'.format(
        testing_or_training.title(),
        nb_computations,
        first_over
    ))
    plt.legend(loc='upper left', prop={'size': 9})
    if save_figure:
        if len(folder_names) == 1:
            plt.savefig('Results/' + folder_names[0] + '/plot_first_over_%d' % first_over, dpi=500)
        else:
            plt.savefig('Results/new_plot_first_over_%d' % first_over, dpi=500)
    plt.show()

if __name__ == "__main__":
    nb_computations = 50            # Number of computations for bootstrapping
    alpha = 0.95                    # Confidence interval
    folder_names = [
        'AgentDQL_comp50_maxep500_update250_doubleTrue_duelingFalse_perFalse_epssteps1000_rms20000'
    ]
    for folder_name in folder_names:
        plot_computations(folder_name, nb_computations, alpha, process_avg_over=20, ra=True, confint=True, save_figure=True)
    plot_first_over(folder_names, nb_computations, testing_or_training='testing', first_over=180, display_mean=True, save_figure=True)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys


def plot_computations(folder_name, nb_computations, alpha, process_average_over=20, confint=True, ra=True, save_figure=True):
    path = 'Results/' + folder_name + "/"
    data = []
    for i in range(1, nb_computations + 1):
        df = pd.read_csv(path + "comp%d.csv" % i, index_col=None, sep=';')
        df.sort_values(by=['timestamps'], inplace=True)  # sort in good order...
        df["training_score_ra"] = df["training_score"].rolling(window=process_average_over, min_periods=1).mean()
        df["testing_score_ra"] = df["testing_score"].rolling(window=process_average_over, min_periods=1).mean()
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
    fig.suptitle('Training and Testing Performances (CI bootstraped over %d computations)' % nb_computations)
    ax1.legend(loc='upper left', prop={'size': 9})
    ax2.legend(loc='upper left', prop={'size': 9})
    if save_figure:
        plt.savefig(path + folder_name, dpi=500)
    plt.show()


if __name__ == "__main__":
    nb_computations = 10            # Number of computations for bootstrapping
    alpha = 0.95                    # Confidence interval
    folder_name = 'AgentPG_comp%d_maxep200_entropy0001_ppo02_lambd1' % nb_computations
    # filename = ['AgentDQL_comp%d_maxep200_update200_doubleTrue_duelingFalse_perTrue_epssteps1000_rms20000.csv' % comp for comp in range(1, nb_computations+1)]
    plot_computations(folder_name, nb_computations, alpha, process_average_over=20, ra=True, confint=True, save_figure=True)

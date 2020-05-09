import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os


def process_folders(folders, alpha=0.90, nb_computations=10, process_avg_over=20, first_over=180):
    folders_data = {}
    for f in folders:
        folders_data[f] = {}
        folders_data[f]["first_over_test"] = []
        folders_data[f]["first_over_train"] = []
        folders_data[f]["first_over_test_ra"] = []
        folders_data[f]["first_over_train_ra"] = []
        path = 'Results/' + f + "/"
        data = []
        for i in range(1, nb_computations + 1):
            df = pd.read_csv(path + "comp%d.csv" % i, index_col=None, sep=';')
            df.sort_values(by=['timestamps'], inplace=True)  # sort in good order...
            df.index += 1
            df["ep"] = df.index
            df["training_score_ra"] = df["training_score"].rolling(window=process_avg_over, min_periods=1).mean()
            df["testing_score_ra"] = df["testing_score"].rolling(window=process_avg_over, min_periods=1).mean()
            folders_data[f]["first_over_test"] += [df.loc[df['testing_score'] >= first_over].head(1)]
            folders_data[f]["first_over_train"] += [df.loc[df['training_score'] >= first_over].head(1)]
            folders_data[f]["first_over_test_ra"] += [df.loc[df['testing_score_ra'] >= first_over].head(1)]
            folders_data[f]["first_over_train_ra"] += [df.loc[df['training_score_ra'] >= first_over].head(1)]
            data += [df]
        folders_data[f]["first_over_test"] = pd.concat(folders_data[f]["first_over_test"], keys=range(nb_computations), ignore_index=True)
        folders_data[f]["first_over_train"] = pd.concat(folders_data[f]["first_over_train"], keys=range(nb_computations), ignore_index=True)
        folders_data[f]["first_over_test_ra"] = pd.concat(folders_data[f]["first_over_test_ra"], keys=range(nb_computations), ignore_index=True)
        folders_data[f]["first_over_train_ra"] = pd.concat(folders_data[f]["first_over_train_ra"], keys=range(nb_computations), ignore_index=True)

        alldf = pd.concat(data, axis=1, keys=range(len(data)))
        keydf = alldf.swaplevel(0, 1, axis=1).groupby(level=0, axis=1)

        folders_data[f]['meandf'] = keydf.mean()
        folders_data[f]['lowerdf'] = keydf.quantile((1 - alpha) / 2)
        folders_data[f]['upperdf'] = keydf.quantile(1 - (1 - alpha) / 2)
    return folders_data


def plot_all(folders_data, ra=True):
    bba = [0.5, 0.93]
    figscore, axesscore = plt.subplots(2, 5, sharex=True, sharey=True)
    for idx, (method, f_data) in enumerate(folders_data.items()):
        meandf = f_data["meandf"]
        lowerdf = f_data["lowerdf"]
        upperdf = f_data["upperdf"]
        i = idx // 5
        j = idx % 5
        ax = axesscore[i][j]
        if ra:
            ax.plot(meandf.index, meandf['training_score_ra'], c='b', linewidth=1, label='Mean Train Score RA')
            ax.plot(meandf.index, meandf['testing_score_ra'], c='g', linewidth=1, label='Mean Test Score RA')
            ax.fill_between(meandf.index, lowerdf['training_score_ra'], upperdf['training_score_ra'], color='b', alpha=0.2, label='%d%% CI Train RA' % int(100 * alpha))
            ax.fill_between(meandf.index, lowerdf['testing_score_ra'], upperdf['testing_score_ra'], color='g', alpha=0.2, label='%d%% CI Test RA' % int(100 * alpha))
        else:
            ax.plot(meandf.index, meandf['training_score'], c='b', linewidth=1, label='Mean Train Score')
            ax.plot(meandf.index, meandf['testing_score'], c='g', linewidth=1, label='Mean Test Score')
            ax.fill_between(meandf.index, lowerdf['training_score'], upperdf['training_score'], color='b', alpha=0.2, label='%d%% CI Train' % int(100 * alpha))
            ax.fill_between(meandf.index, lowerdf['testing_score'], upperdf['testing_score'], color='g', alpha=0.2, label='%d%% CI Test' % int(100 * alpha))
        ax.grid(True, which="both", linestyle='--', color='k', alpha=0.5)
        ax.set_title(method)
        if i == 1:
            ax.set_xlabel('Episodes')
        if j == 0:
            ax.set_ylabel('Score per Episode')
    handles, labels = axesscore[0][0].get_legend_handles_labels()
    figscore.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=bba)
    figscore.subplots_adjust(top=0.85, bottom=0.23, wspace=0.06, hspace=0.16)
    figscore.suptitle('Training and Testing Score Performances\n(%d%% Confidence Intervals and Mean bootstrapped over %d computations)' % (int(100 * alpha), nb_computations))

    figtime, axestime = plt.subplots(2, 5, sharex=True, sharey=True)
    for idx, (method, f_data) in enumerate(folders_data.items()):
        meandf = f_data["meandf"]
        lowerdf = f_data["lowerdf"]
        upperdf = f_data["upperdf"]
        i = idx // 5
        j = idx % 5
        ax = axestime[i][j]
        ax.plot(meandf.index, meandf['timestamps'], c='b', linewidth=1, label='Time')
        ax.fill_between(meandf.index, lowerdf['timestamps'], upperdf['timestamps'], color='b', alpha=0.2, label='%d%% CI Time' % int(100 * alpha))
        ax.grid(True, which="both", linestyle='--', color='k', alpha=0.5)
        ax.set_title(method)
        if i == 1:
            ax.set_xlabel('Episodes')
        if j == 0:
            ax.set_ylabel('Time (s)')
    handles, labels = axestime[0][0].get_legend_handles_labels()
    figtime.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=bba)
    figtime.subplots_adjust(top=0.85, bottom=0.23, wspace=0.06, hspace=0.16)
    figtime.suptitle('Training and Testing Time Performances\n(%d%% Confidence Intervals and Mean bootstrapped over %d computations)' % (int(100 * alpha), nb_computations))


def plot_comparison(folders_data, ra=True, test_or_train='test'):

    plt.figure()
    for method, f_data in folders_data.items():
        meandf = f_data["meandf"]
        if ra:
            plt.plot(meandf.index, meandf['%sing_score_ra' % test_or_train], linewidth=1, label=method)
        else:
            plt.plot(meandf.index, meandf['%sing_score' % test_or_train], linewidth=1, label=method)

    plt.xlabel('Episodes')
    if ra:
        plt.ylabel('Rolling Average Score')
        plt.title('Mean Rolling Average Score Evolution during %sing Phase\n(Bootstrapped over %d computations)' % (test_or_train.title(), nb_computations))
    else:
        plt.ylabel('Score')
        plt.title('Mean Score Evolution during %sing Phase\n(Bootstrapped over %d computations)' % (test_or_train.title(), nb_computations))
    plt.grid(True, which="both", linestyle='--', color='k', alpha=0.5)
    plt.legend(prop={'size': 9})


def plot_first_over(folders_data, ra=True, test_or_train='test', first_over=180, display_mean=True):
    """
    For each folder, plot the ep (x) and timestamps (y) at which the rolling avg for each computations first reached selected "first_over" score.
    Possible to plot the mean point and associated std
    """
    plt.figure()
    for method, f_data in folders_data.items():
        if ra:
            all_df = f_data["first_over_%s_ra" % test_or_train]
        else:
            all_df = f_data["first_over_%s" % test_or_train]
        mean_df = all_df.mean()
        std_df = all_df.std()
        plt.scatter(all_df['ep'], all_df['timestamps'], s=10, alpha=0.6, label=method)
        if display_mean:
            plt.errorbar(mean_df['ep'], mean_df['timestamps'], xerr=std_df['ep'], yerr=std_df['timestamps'], fmt='o', elinewidth=.5, capsize=2)

    plt.xlabel('Episodes')
    plt.ylabel('Time (s)')
    if ra:
        plt.title('First Time a Rolling Average Score of %d is Reached during %sing Phase\n(Scale and Mean bootstrapped over %d computations)' % (first_over, test_or_train.title(), nb_computations))
    else:
        plt.title('First Time a Score of %d is Reached during %sing Phase\n(Scale and Mean bootstrapped over %d computations)' % (first_over, test_or_train.title(), nb_computations))
    plt.grid(True, which="both", linestyle='--', color='k', alpha=0.5)
    plt.legend(prop={'size': 9})


if __name__ == "__main__":
    nb_computations = 10            # Number of computations for bootstrapping
    alpha = 0.90                    # Confidence interval
    process_avg_over = 100          # Rolling Average Window
    ra = True
    first_over = 180
    folders_data = process_folders(os.listdir('Results/'), alpha, nb_computations, process_avg_over, first_over)
    # plot_all(folders_data, ra=ra)
    plot_comparison(folders_data, test_or_train='test', ra=ra)
    # plot_first_over(folders_data, ra=ra, test_or_train='test', first_over=first_over, display_mean=True)
    plt.show()

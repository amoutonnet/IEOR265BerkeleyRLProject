import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os


def process_folders(folders, alpha=0.90, nb_computations=10, process_avg_over=20, first_over=180):
    folders_data = {}
    for f in folders:
        if "excluded" not in f:
            folders_data[f] = {}
            folders_data[f]["first_over_test"] = []
            folders_data[f]["first_over_train"] = []
            folders_data[f]["first_over_test_ra"] = []
            folders_data[f]["first_over_train_ra"] = []
            folders_data[f]["mean_time_per_ep"] = []
            path = 'Results/' + f + "/"
            data = []
            for i in range(1, nb_computations + 1):
                df = pd.read_csv(path + "comp%d.csv" % i, index_col=None, sep=';')
                df.sort_values(by=['timestamps'], inplace=True)  # sort in good order...
                df.index += 1
                df["ep"] = df.index
                df["training_score_ra"] = df["training_score"].rolling(window=process_avg_over, min_periods=1).mean()
                df["testing_score_ra"] = df["testing_score"].rolling(window=process_avg_over, min_periods=1).mean()
                folders_data[f]["mean_time_per_ep"] += [df["timestamps"].diff().fillna(df["timestamps"]).mean()]
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


def time_table(folders_data, latex=True):
    if not latex:
        width = 25
        print("-" * (5 * width + 6))
        print("|" + "Mean Time Per Episode Table (Time in secondes)".center(5 * width + 4) + "|")
        print("-" * (5 * width + 6))
        print("|" + "Method".center(width) + "|" + "Mean Time Per Ep".center(width) + "|" + "Min Mean Time Per Ep".center(
            width) + "|" + "Max Mean Time Per Ep".center(width) + "|" + "Std Mean Time Per Ep".center(width) + "|")
        print("-" * (5 * width + 6))
        for (method, f_data) in folders_data.items():
            data = np.array(f_data["mean_time_per_ep"])
            mean = data.mean()
            low = data.min()
            up = data.max()
            std = data.std()
            print("|" + "{:s}".format(method).center(width) + "|" + "{:.4f}".format(mean).center(width) + "|" + "{:.4f}".format(
                low).center(width) + "|" + "{:.4f}".format(up).center(width) + "|" + "{:.4f}".format(std).center(width) + "|")
        print("-" * (5 * width + 6))
    else:
        for (method, f_data) in folders_data.items():
            data = np.array(f_data["mean_time_per_ep"])
            mean = data.mean()
            low = data.min()
            up = data.max()
            std = data.std()
            print("{:s}".format(method) + " & " + "{:.4f}".format(mean) + " & " + "{:.4f}".format(
                low) + " & " + "{:.4f}".format(up) + " & " + "{:.4f}".format(std) + "\\\\")


def plot_all(folders_data, ra=True, test_or_train="test"):
    bba = [0.5, 0.93]
    figscore, axesscore = plt.subplots(2, 4, sharey=True)
    for idx, (method, f_data) in enumerate(folders_data.items()):
        meandf = f_data["meandf"]
        lowerdf = f_data["lowerdf"]
        upperdf = f_data["upperdf"]
        i = idx // 4
        j = idx % 4
        ax = axesscore[i][j]
        if ra:
            ax.plot(meandf.index, meandf['%sing_score_ra' % test_or_train], c='b', linewidth=1, label='Mean %s Score RA' % test_or_train.title())
            ax.fill_between(meandf.index, lowerdf['%sing_score_ra' % test_or_train], upperdf['%sing_score_ra' % test_or_train],
                            color='b', alpha=0.2, label='%d%% CI %s RA' % (int(100 * alpha), test_or_train.title()))
        else:
            ax.plot(meandf.index, meandf['%sing_score' % test_or_train], c='b', linewidth=1, label='Mean %s Score' % test_or_train.title())
            ax.fill_between(meandf.index, lowerdf['%sing_score' % test_or_train], upperdf['%sing_score' % test_or_train],
                            color='b', alpha=0.2, label='%d%% CI %s' % (int(100 * alpha), test_or_train.title()))
        ax.grid(True, which="both", linestyle='--', color='k', alpha=0.5)
        ax.set_title(method)
        if i == 1:
            ax.set_xlabel('Episodes')
        if j == 0:
            ax.set_ylabel('Score per Episode')
    handles, labels = axesscore[0][0].get_legend_handles_labels()
    figscore.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=bba)
    # figscore.subplots_adjust(top=0.85, bottom=0.23, wspace=0.06, hspace=0.16)
    # figscore.suptitle('Performances during %s Phase\n(%d%% Confidence Intervals and Mean bootstrapped over %d computations)' % (test_or_train.title(), int(100 * alpha), nb_computations))


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
        # plt.title('Mean Rolling Average Score Evolution during %sing Phase\n(Bootstrapped over %d computations)' % (test_or_train.title(), nb_computations))
    else:
        plt.ylabel('Score')
        # plt.title('Mean Score Evolution during %sing Phase\n(Bootstrapped over %d computations)' % (test_or_train.title(), nb_computations))
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
    # if ra:
    #     plt.title('First Time a Rolling Average Score of %d is Reached during %sing Phase\n(Scale and Mean bootstrapped over %d computations)' % (first_over, test_or_train.title(), nb_computations))
    # else:
    #     plt.title('First Time a Score of %d is Reached during %sing Phase\n(Scale and Mean bootstrapped over %d computations)' % (first_over, test_or_train.title(), nb_computations))
    plt.grid(True, which="both", linestyle='--', color='k', alpha=0.5)
    plt.legend(prop={'size': 9})
    plt.ylim(bottom=10)
    plt.yscale('log')


def plot_ppo_alone(folder_name, process_avg_over=50, test_or_train='test', alpha = 0.9):
    path = "Results/excluded/" + folder_name
    data = []
    for i in range(1, nb_computations + 1):
        df = pd.read_csv(path + "/comp%d.csv" %i, index_col=None, sep=';')
        df.sort_values(by=['timestamps'], inplace=True)  # sort in good order...
        df.index += 1
        df["ep"] = df.index
        df["training_score_ra"] = df["training_score"].rolling(window=process_avg_over, min_periods=1).mean()
        df["testing_score_ra"] = df["testing_score"].rolling(window=process_avg_over, min_periods=1).mean()
        data += [df]
    # df["training_score_ra"] = df["training_score"].rolling(window=process_avg_over, min_periods=1).mean()
    alldf = pd.concat(data, axis=1, keys=range(len(data)))
    keydf = alldf.swaplevel(0, 1, axis=1).groupby(level=0, axis=1)

    meandf = keydf.mean()
    lowerdf = keydf.quantile((1 - alpha) / 2)
    upperdf = keydf.quantile(1 - (1 - alpha) / 2)

    plt.figure()
    plt.plot(meandf.index, meandf[['%sing_score_ra' % test_or_train]], c='b', linewidth=1, label='Mean %s Score RA' % test_or_train.title())
    plt.fill_between(meandf.index, lowerdf['%sing_score_ra' % test_or_train], upperdf['%sing_score_ra' % test_or_train],
                            color='b', alpha=0.2, label='%d%% CI %s RA' % (int(100 * alpha), test_or_train.title()))
    df["testing_score_ra"] = df["testing_score"].rolling(window=process_avg_over, min_periods=1).mean()
    plt.xlabel("Episodes")
    plt.ylabel("Rolling Average Score")
    plt.grid(True, which="both", linestyle='--', color='k', alpha=0.5)
    plt.legend()


if __name__ == "__main__":
    nb_computations = 20            # Number of computations for bootstrapping
    alpha = 0.90                    # Confidence interval
    process_avg_over = 50          # Rolling Average Window
    ra = True
    first_over = 180
    # folders_data = process_folders(os.listdir('Results/'), alpha, nb_computations, process_avg_over, first_over)
    # plot_all(folders_data, test_or_train='test', ra=ra)
    # time_table(folders_data, latex=True)
    # plot_comparison(folders_data, test_or_train='test', ra=ra)
    # plot_first_over(folders_data, ra=ra, test_or_train='test', first_over=first_over, display_mean=True)
    plot_ppo_alone("PGA2C")
    plt.show()

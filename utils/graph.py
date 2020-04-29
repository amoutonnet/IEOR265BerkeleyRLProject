import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_computations(folder_name, nb_computations, alpha, confidence_interval=True, save_figure=True):
    path = 'Results/' + folder_name + "/"
    li = []
    file_paths = [ path + 'comp' + str(comp) +'.csv' for comp in range(1, nb_computations + 1)]
    for filename in file_paths:
        df = pd.read_csv(filename, index_col=None, sep=';')
        df.sort_values(by=['timestamps'], inplace=True)  # sort in good order...
        df.reset_index(drop=True, inplace=True)  # ... update the index...
        df['ep'] = df.index  # and set it as the episode #
        li.append(df)

    # Concatenate all files and bootstrap mean over episode for timestamp and scores
    all_df = pd.concat(li, ignore_index=True)  #concatenate all files (dataframes) into one
    stats_df = all_df.groupby(['ep'], as_index=False).mean()  # mean by ep

    # compute confidence interval
    quantile_low, quantile_upp = (1 - alpha) / 2, alpha + (1 - alpha) / 2
    lower, upper = [], []
    for ep in range(len(stats_df.index)):
        temp = all_df.loc[all_df['ep'] == ep]  # take all same episodes values to compute confidence interval
        lower.append(temp.quantile(quantile_low).to_frame().T)  # list of 1-line dataframes with lower percentile
        upper.append(temp.quantile(quantile_upp).to_frame().T)
    lower_df = pd.concat(lower, ignore_index=True)  # concatenate all dataframes in one of same length as stats_df
    upper_df = pd.concat(upper, ignore_index=True)
    lower_df.rename(columns={  # rename columns for clarity
        'timestamps': 'timestamps_low',
        'training_score':'training_score_low',
        'testing_score':'testing_score_low'
    }, inplace=True)
    upper_df.rename(columns={
        'timestamps': 'timestamps_upp',
        'training_score':'training_score_upp',
        'testing_score':'testing_score_upp'
    }, inplace=True)
    lower_df['ep'] = lower_df.ep.astype('int64')  # ep for converted to float64 when quantiles were computed, changing it back
    upper_df['ep'] = upper_df.ep.astype('int64')

    # Add it to the mean df and convert to dict of numpy
    stats_df = stats_df.join(lower_df, on='ep', lsuffix='_left', rsuffix='_right')  # join mean and lower CI based on ep
    stats_df.drop(columns=['ep_right'], inplace=True)  # drop one ep column
    stats_df.rename(columns={'ep_left':'ep'}, inplace=True)  # rename the otehr ep column to 'ep'
    stats_df = stats_df.join(upper_df, on='ep', lsuffix='_left', rsuffix='_right')
    stats_df.drop(columns=['ep_right'], inplace=True)
    stats_df.rename(columns={'ep_left':'ep'}, inplace=True)
    data = {}  # dictionnary containing numpy arrays to be ploted
    for column_name in stats_df.columns.values:
        data[column_name] = stats_df[column_name].to_numpy()

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.plot(data['ep'], data['training_score'], c='b', linewidth=1, label='Train Score')
    ax1.plot(data['ep'], data['testing_score'], c='g', linewidth=1, label='Test Score')
    if confidence_interval:
        ax1.fill_between(data['ep'], data['training_score_low'], data['training_score_upp'], color='b', alpha=0.2, label='%.2f%% CI training' %alpha)
        ax1.fill_between(data['ep'], data['testing_score_low'], data['testing_score_upp'], color='g', alpha=0.2, label='%.2f%% CI testing' %alpha)
    ax1.set(xlabel='Episodes', ylabel='Score')
    if confidence_interval:
        mask = data['ep'] % 20 == 0
        xerr_left, xerr_right = data['timestamps_low'] * mask, data['timestamps_upp'] * mask
        ax2.errorbar(data['timestamps'], data['training_score'], c='b', linewidth=1, xerr=[xerr_left, xerr_right], ecolor='tab:blue', elinewidth=.7, capsize=5)
        ax2.errorbar(data['timestamps'], data['testing_score'], c='g', linewidth=1, xerr=[xerr_left, xerr_right], ecolor='tab:green', elinewidth=.7, capsize=5)
        ax2.fill_between(data['timestamps'], data['training_score_low'], data['training_score_upp'], color='b', alpha=0.2)
        ax2.fill_between(data['timestamps'], data['testing_score_low'], data['testing_score_upp'], color='g', alpha=0.2)
    ax2.set(xlabel='Time (s)')
    fig.suptitle('Confidence interval for Training and Testing, bootstraped over %d computations' % nb_computations)
    fig.legend(ncol=5, loc='upper center', bbox_to_anchor=(0.5, 0.955), prop={'size': 9})
    if save_figure:
        plt.savefig(path + folder_name, dpi=500)
    plt.show()

    
if __name__ == "__main__":
    nb_computations = 10            # Number of computations for bootstrapping
    alpha = 0.95                    # Confidence interval
    folder_name = 'AgentPG_comp10_maxep200_entropy0001_ppo02_lambd1'
    # filename = ['AgentDQL_comp%d_maxep200_update200_doubleTrue_duelingFalse_perTrue_epssteps1000_rms20000.csv' % comp for comp in range(1, nb_computations+1)]
    plot_computations(folder_name, nb_computations, alpha)
    


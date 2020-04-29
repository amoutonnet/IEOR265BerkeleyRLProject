import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_computations(filenames, alpha, confidence_interval=True, save_figure=True, figname='newfig'):
    computations = len(filenames)
    li = []
    for filename in filenames:
        df = pd.read_csv('Results/' + filename, index_col=None, sep=';')
        df.sort_values(by=['timestamps'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        df['ep'] = df.index
        li.append(df)

    # Concatenate all files and bootstrap mean over episode for timestamp and scores
    all_df = pd.concat(li, ignore_index=True)
    stats_df = all_df.groupby(['ep'], as_index=False).mean()

    # compute confidence interval
    quantile_low, quantile_upp = (1 - alpha) / 2, alpha + (1 - alpha) / 2
    lower, upper = [], []
    for ep in range(len(stats_df.index)):
        temp = all_df.loc[all_df['ep'] == ep]
        lower.append(temp.quantile(quantile_low).to_frame().T)
        upper.append(temp.quantile(quantile_upp).to_frame().T)
    lower_df = pd.concat(lower, ignore_index=True)
    upper_df = pd.concat(upper, ignore_index=True)
    lower_df.rename(columns={
        'timestamps': 'timestamps_low',
        'training_score':'training_score_low',
        'testing_score':'testing_score_low'
    }, inplace=True)
    upper_df.rename(columns={
        'timestamps': 'timestamps_upp',
        'training_score':'training_score_upp',
        'testing_score':'testing_score_upp'
    }, inplace=True)
    lower_df['ep'] = lower_df.ep.astype('int64')
    upper_df['ep'] = upper_df.ep.astype('int64')

    # Add it to the mean df and convert to dict of numpy
    stats_df = stats_df.join(lower_df, on='ep', lsuffix='_left', rsuffix='_right')
    stats_df.drop(columns=['ep_right'], inplace=True)
    stats_df.rename(columns={'ep_left':'ep'}, inplace=True)
    stats_df = stats_df.join(upper_df, on='ep', lsuffix='_left', rsuffix='_right')
    stats_df.drop(columns=['ep_right'], inplace=True)
    stats_df.rename(columns={'ep_left':'ep'}, inplace=True)
    data = {}
    for column_name in stats_df.columns.values:
        data[column_name] = stats_df[column_name].to_numpy()

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.plot(data['ep'], data['training_score'], c='b', linewidth=1, label='Train Score')
    ax1.plot(data['ep'], data['testing_score'], c='g', linewidth=1, label='Test Score')
    if confidence_interval:
        ax1.fill_between(data['ep'], data['training_score_low'], data['training_score_upp'], color='b', alpha=0.2, label='%.2f%% training CI' %alpha)
        ax1.fill_between(data['ep'], data['testing_score_low'], data['testing_score_upp'], color='g', alpha=0.2, label='%.2f%% testing CI' %alpha)
    ax1.set(xlabel='Episodes', ylabel='Score')
    if confidence_interval:
        mask = data['ep'] % 20 == 0
        xerr_left, xerr_right = data['timestamps_low'] * mask, data['timestamps_upp'] * mask
        ax2.errorbar(data['timestamps'], data['training_score'], c='b', linewidth=1, xerr=[xerr_left, xerr_right], ecolor='tab:blue', elinewidth=.7, capsize=5)
        ax2.errorbar(data['timestamps'], data['testing_score'], c='g', linewidth=1, xerr=[xerr_left, xerr_right], ecolor='tab:green', elinewidth=.7, capsize=5)
        ax2.fill_between(data['timestamps'], data['training_score_low'], data['training_score_upp'], color='b', alpha=0.2)
        ax2.fill_between(data['timestamps'], data['testing_score_low'], data['testing_score_upp'], color='g', alpha=0.2)
    ax2.set(xlabel='Time (s)')
    fig.suptitle('Confidence interval for Training and testing, bootstraped over %d compuations' %computations)
    fig.legend(ncol=5, loc='upper center', bbox_to_anchor=(0.5, 0.955), prop={'size': 9})
    if save_figure:
        plt.savefig('Results/' + figname, dpi=500, format='png')
    plt.show()

    
if __name__ == "__main__":
    alpha = 0.95                # Confidence interval
    computations = 10            # Number of computations for bootstrapping
    filenames = [
        'AgentDQL_comp%d_maxep200_update200_doubleTrue_duelingFalse_perFalse_epssteps1000_rms20000.csv' %i for i in range(1, computations + 1)
    ]
    graph.plot_computations(filenames, alpha)
    


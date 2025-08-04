import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def create_rms_histogram(
        data: pd.DataFrame,
        save_name: str,
        axis_tick_size: int = 12,
        column: str = 'rms_error',
        num_bins: int = 25,
):
    # create histogram
    data[column].hist(bins=num_bins, color='purple', edgecolor='black', linewidth=1.2)

    # update ticks on x-axis to 0.05
    plt.xticks(np.arange(0, data[column].max() + 0.05, 0.05))

    # increase tick mark size
    plt.tick_params(axis='both', which='major', labelsize=axis_tick_size)

    # add vertical line w/ median
    median_value = data['rms_error'].median()
    plt.axvline(median_value, color='k', linestyle='dashed', linewidth=2)

    # set figure components
    plt.xlabel('RMSE (m)', fontdict={'size': 18})
    plt.ylabel('Frequency', fontdict={'size': 18})

    # add legend
    plt.legend([f'Median RMSE ({median_value:.3f} m)'])

    # remove the right and top spines
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    # save plot
    plt.savefig(f'figures/{save_name}.png')
    
    # show plot
    plt.show()

# plot all trajectories for a specified joint angle
def plot_all_trajectories(
        data: pd.DataFrame,
        angle: str,
        y_label: str = 'Elbow Flexion (degrees)',
):
    # check for subject ID column
    if 'subject_id' not in data.columns:
        data.insert(0, 'subject_id', data['study_id'].str.split('_').str[0])
    
    # plot each trajectory
    plt.figure(figsize=(10, 6))
    for _, study_id in list(data.groupby(['subject_id', 'study_id'])):
        plt.plot(study_id['normalized_time'], study_id[angle], linewidth=0.25)

    # axis labels
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.xlabel('Normalized Time', fontsize=16)
    plt.ylabel(y_label, fontsize=16)

    # remove the right and top spines
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
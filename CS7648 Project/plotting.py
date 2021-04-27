import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")


def plot_training_data():
    zero_hyperball = pd.read_csv('0_hyperball_experiment.csv')
    five_hyperball = pd.read_csv('5_hyperball_experiment.csv')
    ten_hyperball = pd.read_csv('10_hyperball_experiment.csv')
    twenty_hyperball = pd.read_csv('20_hyperball_experiment.csv')

    zero_hyperball['k-hyperballs'] = 0
    five_hyperball['k-hyperballs'] = 5
    ten_hyperball['k-hyperballs'] = 10
    twenty_hyperball['k-hyperballs'] = 20

    zero_hyperball = zero_hyperball[zero_hyperball.epoch <= 15]
    five_hyperball = five_hyperball[five_hyperball.epoch <= 15]
    ten_hyperball = ten_hyperball[ten_hyperball.epoch <= 15]
    twenty_hyperball = twenty_hyperball[twenty_hyperball.epoch <= 15]

    zero_hyperball['feedback'] = zero_hyperball['positive feedback'] + zero_hyperball['negative feedback']
    five_hyperball['feedback'] = five_hyperball['positive feedback'] + five_hyperball['negative feedback']
    ten_hyperball['feedback'] = ten_hyperball['positive feedback'] + ten_hyperball['negative feedback']
    twenty_hyperball['feedback'] = twenty_hyperball['positive feedback'] + twenty_hyperball['negative feedback']



    plot_kballs = sns.lineplot(data=pd.concat([zero_hyperball, five_hyperball, ten_hyperball, twenty_hyperball]), x="epoch", y="reward", err_style=None, hue='k-hyperballs', palette=sns.color_palette("hls", 4)).set_title('Ground-truth Reward Over Epochs - Incrementing HyperBalls')
    plt.show()

    zero_hyperball_only_feedback = zero_hyperball.groupby('epoch').feedback.mean().reset_index()
    five_hyperball_only_feedback = five_hyperball.groupby('epoch').feedback.mean().reset_index()
    ten_hyperball_only_feedback = ten_hyperball.groupby('epoch').feedback.mean().reset_index()
    twenty_hyperball_only_feedback = twenty_hyperball.groupby('epoch').feedback.mean().reset_index()

    zero_hyperball_only_feedback['k-hyperballs'] = 0
    five_hyperball_only_feedback['k-hyperballs'] = 5
    ten_hyperball_only_feedback['k-hyperballs'] = 10
    twenty_hyperball_only_feedback['k-hyperballs'] = 20

    sns.barplot(data=pd.concat([zero_hyperball_only_feedback, five_hyperball_only_feedback, ten_hyperball_only_feedback, twenty_hyperball_only_feedback]), x="epoch", y="feedback", hue='k-hyperballs', palette=sns.color_palette("hls", 4), errwidth=None, errcolor=None).set_title(
        'Ground-truth Reward Over Epochs - Incrementing HyperBalls')
    plt.show()


def main():
    plot_training_data()

if __name__ == "__main__":
    main()

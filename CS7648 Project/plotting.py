import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")


def plot_gym_ablation():
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 5)

    zero_hyperball = pd.read_csv('experiments/hyperball_puyao/no_hyperball_experiment.csv')
    five_hyperball = pd.read_csv('experiments/hyperball_puyao/5_hyperball_experiment.csv')
    ten_hyperball = pd.read_csv('experiments/hyperball_puyao/10_hyperball_experiment.csv')
    twenty_hyperball = pd.read_csv('experiments/hyperball_puyao/20_hyperball_experiment.csv')

    zero_hyperball['K-HyperSpheres'] = 0
    five_hyperball['K-HyperSpheres'] = 5
    ten_hyperball['K-HyperSpheres'] = 10
    twenty_hyperball['K-HyperSpheres'] = 20

    zero_hyperball = zero_hyperball[zero_hyperball.Epoch < 15]
    five_hyperball = five_hyperball[five_hyperball.Epoch < 15]
    ten_hyperball = ten_hyperball[ten_hyperball.Epoch < 15]
    twenty_hyperball = twenty_hyperball[twenty_hyperball.Epoch < 15]

    zero_hyperball['feedback'] = zero_hyperball['positive feedback'] + zero_hyperball['negative feedback']
    five_hyperball['feedback'] = five_hyperball['positive feedback'] + five_hyperball['negative feedback']
    ten_hyperball['feedback'] = ten_hyperball['positive feedback'] + ten_hyperball['negative feedback']
    twenty_hyperball['feedback'] = twenty_hyperball['positive feedback'] + twenty_hyperball['negative feedback']



    sns.lineplot(data=pd.concat([zero_hyperball, five_hyperball, ten_hyperball, twenty_hyperball]), x="Epoch", y="Reward", hue='K-HyperSpheres', palette=sns.color_palette("hls", 4), ax=ax)
    ax.figure.savefig('experiments/figures/avg_reward_training_gym_hypersphere')
    plt.show()

    zero_hyperball_only_feedback = zero_hyperball.groupby('Epoch').feedback.mean().reset_index()
    five_hyperball_only_feedback = five_hyperball.groupby('Epoch').feedback.mean().reset_index()
    ten_hyperball_only_feedback = ten_hyperball.groupby('Epoch').feedback.mean().reset_index()
    twenty_hyperball_only_feedback = twenty_hyperball.groupby('Epoch').feedback.mean().reset_index()

    zero_hyperball_only_feedback['K-HyperSpheres'] = 0
    five_hyperball_only_feedback['K-HyperSpheres'] = 5
    ten_hyperball_only_feedback['K-HyperSpheres'] = 10
    twenty_hyperball_only_feedback['K-HyperSpheres'] = 20
    feedback_data = pd.concat([zero_hyperball_only_feedback, five_hyperball_only_feedback, ten_hyperball_only_feedback, twenty_hyperball_only_feedback])
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 5)
    sns.barplot(data=feedback_data, x="K-HyperSpheres", y="feedback", palette=sns.color_palette("hls", 4), errwidth=None, errcolor=None, ax=ax)
    ax.figure.savefig('experiments/figures/avg_feedback_gym_hypersphere')
    plt.show()

    zero_hyperball_eval = pd.read_csv('experiments/hyperball_puyao/no_hyperball_verification.csv')
    five_hyperball_eval = pd.read_csv('experiments/hyperball_puyao/5_hyperball_verification.csv')
    ten_hyperball_eval = pd.read_csv('experiments/hyperball_puyao/10_hyperball_verification.csv')
    twenty_hyperball_eval = pd.read_csv('experiments/hyperball_puyao/20_hyperball_verification.csv')
    eval_data = pd.DataFrame(columns=['K-HyperSpheres', 'trial', 'Reward'])
    zero_hyperball_eval.columns = ['trial', 'Reward']
    five_hyperball_eval.columns = ['trial', 'Reward']
    ten_hyperball_eval.columns = ['trial', 'Reward']
    twenty_hyperball_eval.columns = ['trial', 'Reward']
    zero_hyperball_eval['K-HyperSpheres'] = 0
    five_hyperball_eval['K-HyperSpheres'] = 5
    ten_hyperball_eval['K-HyperSpheres'] = 10
    twenty_hyperball_eval['K-HyperSpheres'] = 20
    eval_data = pd.concat([zero_hyperball_eval, five_hyperball_eval, ten_hyperball_eval, twenty_hyperball_eval])
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 5)
    sns.barplot(data=eval_data, x="K-HyperSpheres", y="Reward", palette=sns.color_palette("hls", 4), errwidth=None, errcolor=None, ax=ax)
    ax.figure.savefig('experiments/figures/avg_reward_eval_gym_hypersphere')
    plt.show()

def language_model_gym():
    lang_with_hyperballs = pd.read_csv('experiments/language_model_gym/TAMER_HS20_LM_experiment.csv')
    lang_no_hyperballs = pd.read_csv('experiments/language_model_gym/TAMER_LM_experiment.csv')
    no_lang_with_hyperballs = pd.read_csv('experiments/hyperball_puyao/20_hyperball_experiment.csv')
    no_lang_no_hyperballs = pd.read_csv('experiments/hyperball_puyao/no_hyperball_experiment.csv')

    no_lang_with_hyperballs = no_lang_with_hyperballs[no_lang_with_hyperballs.Epoch < 15]
    no_lang_no_hyperballs = no_lang_no_hyperballs[no_lang_no_hyperballs.Epoch < 15]

    lang_with_hyperballs['Model'] = '20 HS - Verbal Feedback'
    lang_no_hyperballs['Model'] = '0 HS - Verbal Feedback'
    no_lang_with_hyperballs['Model'] = '20 HS - Keyboard Feedback'
    no_lang_no_hyperballs['Model'] = '0 HS - Keyboard Feedback'

    lang_with_hyperballs['feedback'] = lang_with_hyperballs['positive feedback'] + lang_with_hyperballs['negative feedback']
    lang_no_hyperballs['feedback'] = lang_no_hyperballs['positive feedback'] + lang_no_hyperballs['negative feedback']
    no_lang_with_hyperballs['feedback'] = no_lang_with_hyperballs['positive feedback'] + no_lang_with_hyperballs['negative feedback']
    no_lang_no_hyperballs['feedback'] = no_lang_no_hyperballs['positive feedback'] + no_lang_no_hyperballs['negative feedback']

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 5)
    sns.lineplot(data=pd.concat([lang_no_hyperballs, no_lang_no_hyperballs, lang_with_hyperballs, no_lang_with_hyperballs]), x="Epoch", y="Reward", hue='Model', palette=sns.color_palette("hls", 4), ax=ax)
    ax.legend(loc="upper left", title="Model")
    ax.figure.savefig('experiments/figures/avg_training_reward_gym')
    plt.show()

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 5)
    sns.lineplot(data=pd.concat([lang_no_hyperballs, no_lang_no_hyperballs, lang_with_hyperballs, no_lang_with_hyperballs]), x="Epoch", y="feedback",hue='Model', palette=sns.color_palette("hls", 4), ax=ax)
    ax.figure.savefig('experiments/figures/avg_feedback_lineplot_avg')
    plt.show()

    lang_no_hs_only_feedback = lang_no_hyperballs.groupby('Epoch').feedback.mean().reset_index()
    lang_with_hs_only_feedback = lang_with_hyperballs.groupby('Epoch').feedback.mean().reset_index()
    no_lang_no_hs_only_feedback = no_lang_no_hyperballs.groupby('Epoch').feedback.mean().reset_index()
    no_lang_hs_only_feedback = no_lang_with_hyperballs.groupby('Epoch').feedback.mean().reset_index()

    lang_no_hs_only_feedback['Model'] = '0 HS - Verbal'
    lang_with_hs_only_feedback['Model'] = '20 HS - Verbal'
    no_lang_no_hs_only_feedback['Model'] = '0 HS - Keyboard'
    no_lang_hs_only_feedback['Model'] = '20 HS - Keyboard'

    feedback_data = pd.concat([lang_no_hs_only_feedback, no_lang_no_hs_only_feedback, lang_with_hs_only_feedback, no_lang_hs_only_feedback])
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 5)
    sns.barplot(data=feedback_data, x="Model", y="feedback", palette=sns.color_palette("hls", 4), errwidth=None, errcolor=None, ax=ax)
    ax.figure.savefig('experiments/figures/avg_feedback_gym')
    plt.show()

    lang_with_hyperballs_eval = pd.read_csv('experiments/language_model_gym/TAMER_HS20_LM_verification.csv')
    lang_no_hyperballs_eval = pd.read_csv('experiments/language_model_gym/TAMER_LM_verification.csv')
    no_lang_no_hyperballs_eval = pd.read_csv('experiments/hyperball_puyao/no_hyperball_verification.csv')
    no_lang_with_hyperballs_eval = pd.read_csv('experiments/hyperball_puyao/20_hyperball_verification.csv')

    lang_with_hyperballs_eval.columns = ['trial', 'Reward']
    lang_no_hyperballs_eval.columns = ['trial', 'Reward']
    no_lang_no_hyperballs_eval.columns = ['trial', 'Reward']
    no_lang_with_hyperballs_eval.columns = ['trial', 'Reward']

    lang_with_hyperballs_eval['Model'] = '20 HS - Verbal'
    lang_no_hyperballs_eval['Model'] = '0 HS - Verbal'
    no_lang_no_hyperballs_eval['Model'] = '0 HS - Keyboard'
    no_lang_with_hyperballs_eval['Model'] = '20 HS - Keyboard'

    eval_data = pd.concat([lang_no_hyperballs_eval, no_lang_no_hyperballs_eval, lang_with_hyperballs_eval, no_lang_with_hyperballs_eval])
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 5)
    sns.barplot(data=eval_data, x="Model", y="Reward", palette=sns.color_palette("hls", 4), errwidth=None, errcolor=None, ax=ax)
    ax.figure.savefig('experiments/figures/avg_reward_eval_gym')
    plt.show()

def heatmaps_alto_experiment():
    subject1 = pd.read_csv('experiments/variance/plain_feedback_subject1.csv').set_index('Unnamed: 0').rename_axis("State")
    subject1_speech = pd.read_csv('experiments/variance/speech_feedback_subject1.csv').set_index('Unnamed: 0').rename_axis("State")
    subject2 = pd.read_csv('experiments/variance/plain_feedback_subject2.csv').set_index('Unnamed: 0').rename_axis("State")
    subject2_speech = pd.read_csv('experiments/variance/speech_feedback_subject2.csv').set_index('Unnamed: 0').rename_axis("State")


    fig, ax = plt.subplots()
    fig.set_size_inches(8, 5)
    ax = sns.heatmap(subject1, ax=ax, cmap=sns.color_palette("coolwarm", as_cmap=True), vmin=0, vmax=2)
    ax.set_xlabel('Action')
    ax.figure.savefig('experiments/figures/subject1_keyboard')
    plt.show()

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 5)
    ax = sns.heatmap(subject1_speech, ax=ax, cmap=sns.color_palette("coolwarm", as_cmap=True), vmin=0, vmax=2)
    ax.set_xlabel('Action')
    ax.figure.savefig('experiments/figures/subject1_speech')
    plt.show()

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 5)
    ax = sns.heatmap(subject2, ax=ax, cmap=sns.color_palette("coolwarm", as_cmap=True), vmin=0, vmax=2)
    ax.set_xlabel('Action')
    ax.figure.savefig('experiments/figures/subject2_keyboard')
    plt.show()

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 5)
    ax = sns.heatmap(subject2_speech, ax=ax, cmap=sns.color_palette("coolwarm", as_cmap=True), vmin=0, vmax=2)
    ax.set_xlabel('Action')
    ax.figure.savefig('experiments/figures/subject2_speech')
    plt.show()

def hyperballs_tgt_net():
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 5)

    zero_hyperball = pd.read_csv('experiments/hyperball_puyao/no_hyperball_experiment.csv')
    five_hyperball = pd.read_csv('experiments/hyperball_puyao/5_hyperball_experiment.csv')
    ten_hyperball = pd.read_csv('experiments/hyperball_puyao/10_hyperball_experiment.csv')
    twenty_hyperball = pd.read_csv('experiments/hyperball_puyao/20_hyperball_experiment.csv')

    zero_hyperball['K-HyperSpheres'] = 0
    five_hyperball['K-HyperSpheres'] = 5
    ten_hyperball['K-HyperSpheres'] = 10
    twenty_hyperball['K-HyperSpheres'] = 20

def main():
    # plot_gym_ablation()
    # language_model_gym()
    heatmaps_alto_experiment()

if __name__ == "__main__":
    main()

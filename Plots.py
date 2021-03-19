import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class Plots:
    @staticmethod
    def plot_distribution(df: pd.DataFrame, label: str, save: bool, figsize: tuple = (6.4, 4.8)):
        plt.figure(figsize=figsize)
        sns.barplot(x=df[label].value_counts().index, y=df[label].value_counts())
        plt.ylabel(label.upper())
        plt.savefig("./plots/distribution_plot.png") if save else ""
        plt.show()

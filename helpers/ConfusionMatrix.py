import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class ConfusionMatrix:
    
    def __init__(self):
        return self
    
    def display_confusion_matrix(confusion_matrix, axes, class_label, class_names, fontsize=14):
        df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
        try:
            heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
        except ValueError:
            raise ValueError("Confusion matrix values must be integers.")
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
        # axes.set_ylabel('True label')
        # axes.set_xlabel('Predicted label')
        axes.set_title("Confusion Matrix for the class - " + class_label)
        axes.margins(20)

    def print_confusion_matrix(matrixes, labels):
        fig, ax = plt.subplots(2, 2, figsize=(12, 7))
        for axes, cfs_matrix, label in zip(ax.flatten(), matrixes, labels):
            ConfusionMatrix.display_confusion_matrix(cfs_matrix, axes, label, ["0", "1"])
        fig.tight_layout()
        plt.margins(0.05, 0.1)
        plt.show()
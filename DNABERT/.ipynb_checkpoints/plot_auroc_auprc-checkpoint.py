import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np

def plot_auroc_auprc(nares_predictions, forehead_predictions, stool_predictions, inside_floor_predictions):
    sample_data = {
        "Nares": nares_predictions[0],
        "Forehead": forehead_predictions[0],
        "Stool": stool_predictions[0],
        "Inside floor": inside_floor_predictions[0],
    }

    # set up
    palette = ["#dc9766", "#d32f88", "#914f1f", "#bf64d7"]
    colors = sns.color_palette(palette)
    plt.figure(figsize=(10, 5))
    
    # AUROC
    plt.subplot(1, 2, 1)
    for (sample, (y_pred, y_true)), color in zip(sample_data.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, label=f"{sample}: AUROC={roc_auc:.2f}")
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.xticks(np.arange(0.0, 1.1, 0.25))
    plt.yticks(np.arange(0.0, 1.1, 0.25))
    plt.xticks(np.arange(0.0, 1.1, 0.125), minor=True)
    plt.yticks(np.arange(0.0, 1.1, 0.125), minor=True)
    plt.tick_params(which="minor", length=0) 
    plt.grid(True, linestyle="-", alpha=0.4)
    plt.grid(True, which="minor", linestyle="-", alpha=0.4)
    legend = plt.legend(title="Sample types", framealpha=1, facecolor="white", edgecolor="none", labelspacing=1.3, fontsize="medium")
    legend._legend_box.align = "left"
    
    # AUPRC
    plt.subplot(1, 2, 2)
    for (sample, (y_pred, y_true)), color in zip(sample_data.items(), colors):
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, color=color, label=f"{sample}: AUPRC={pr_auc:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xticks(np.arange(0.0, 1.1, 0.25))
    plt.yticks(np.arange(0.0, 1.1, 0.25))
    plt.xticks(np.arange(0.0, 1.1, 0.125), minor=True)
    plt.yticks(np.arange(0.0, 1.1, 0.125), minor=True)
    plt.tick_params(which="minor", length=0) 
    plt.grid(True, linestyle="-", alpha=0.4)
    plt.grid(True, which="minor", linestyle="-", alpha=0.4)
    legend = plt.legend(title="Sample types", framealpha=1, facecolor="white", edgecolor="none", labelspacing=1.3, fontsize="medium")
    legend._legend_box.align = "left"
    
    # adjust layout
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)
    
    plt.savefig('figures/auroc_auprc_dnabert.png')
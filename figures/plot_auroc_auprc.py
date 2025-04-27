import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np

def plot_auroc_auprc(model_predictions):    
    data = {
        "nares": model_predictions['nares'],
        "forehead": model_predictions['forehead'],
        "stool": model_predictions['stool'],
        "inside_floor": model_predictions['inside_floor'],
    }
    models = ['AAM', 'DNABERT', 'DNABERT-2', 'GROVER']

    # set up
    palette = ["#dc9766", "#d32f88", "#914f1f", "#bf64d7"]
    colors = sns.color_palette(palette)
    
    # AUROC
    for sample_type in data.keys():
      fig, axs = plt.subplots(1, 2, figsize=(12, 6))
      ax1 = axs[0]

      for (y_pred, y_true), color, model in zip(data[sample_type], colors, models):
          fpr, tpr, _ = roc_curve(y_true, y_pred)
          roc_auc = auc(fpr, tpr)
          ax1.plot(fpr, tpr, color=color, label=f"{model}: AUROC={roc_auc:.2f}")

      ax1.set_xlabel("1 - Specificity")
      ax1.set_ylabel("Sensitivity")
      ax1.set_xticks(np.arange(0.0, 1.1, 0.25))
      ax1.set_yticks(np.arange(0.0, 1.1, 0.25))
      ax1.set_xticks(np.arange(0.0, 1.1, 0.125), minor=True)
      ax1.set_yticks(np.arange(0.0, 1.1, 0.125), minor=True)
      ax1.tick_params(which="minor", length=0)
      ax1.grid(True, linestyle="-", alpha=0.4)
      ax1.grid(True, which="minor", linestyle="-", alpha=0.4)
      legend1 = ax1.legend(title="Model", framealpha=1, facecolor="white", edgecolor="none",
                          labelspacing=1.3, fontsize="medium")
      legend1._legend_box.align = "left"

      # AUPRC subplot
      ax2 = axs[1]
      for (y_pred, y_true), color, model in zip(data[sample_type], colors, models):
          precision, recall, _ = precision_recall_curve(y_true, y_pred)
          pr_auc = auc(recall, precision)
          ax2.plot(recall, precision, color=color, label=f"{model}: AUPRC={pr_auc:.2f}")

      ax2.set_xlabel("Recall")
      ax2.set_ylabel("Precision")
      ax2.set_xticks(np.arange(0.0, 1.1, 0.25))
      ax2.set_yticks(np.arange(0.0, 1.1, 0.25))
      ax2.set_xticks(np.arange(0.0, 1.1, 0.125), minor=True)
      ax2.set_yticks(np.arange(0.0, 1.1, 0.125), minor=True)
      ax2.tick_params(which="minor", length=0)
      ax2.grid(True, linestyle="-", alpha=0.4)
      ax2.grid(True, which="minor", linestyle="-", alpha=0.4)
      legend2 = ax2.legend(title="Model", framealpha=1, facecolor="white", edgecolor="none",
                          labelspacing=1.3, fontsize="medium")
      legend2._legend_box.align = "left"

      # Adjust layout
      plt.tight_layout()
      plt.subplots_adjust(wspace=0.3)

      # Save the figure
      plt.savefig(f'figures/auroc_auprc_{sample_type}.png')
      plt.close()
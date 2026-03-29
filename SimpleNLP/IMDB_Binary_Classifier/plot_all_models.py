import os
import pandas as pd
import matplotlib.pyplot as plt


record_dir = "record"
all_records = {}

for file in os.listdir(record_dir):
    if file.endswith("_record.csv"):
        model_name = file.replace("_record.csv", "")
        path = os.path.join(record_dir, file)

        df = pd.read_csv(path)
        all_records[model_name] = df


plt.style.use('seaborn-v0_8-whitegrid')

colors = [
    "#7FB3D5",  # light blue
    "#F7CAC9",  # light pink
    "#82E0AA",  # light green
    "#F8C471",  # light orange
    "#C39BD3",  # light purple
    "#76D7C4"   # cyan
]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))


for i, (model, df) in enumerate(all_records.items()):
    color = colors[i % len(colors)]

    axes[0].plot(
        df['epoch'], df['train_loss'],
        label=f"{model}-train",
        color=color,
        linestyle='-',
        linewidth=2.2,
        alpha=0.9
    )

    axes[0].plot(
        df['epoch'], df['test_loss'],
        label=f"{model}-test",
        color=color,
        linestyle='--',
        linewidth=2.2,
        alpha=0.9
    )

axes[0].set_title("Loss Curve", fontsize=16)
axes[0].set_xlabel("Epoch", fontsize=12)
axes[0].set_ylabel("Loss", fontsize=12)
axes[0].legend(fontsize=9)


for i, (model, df) in enumerate(all_records.items()):
    color = colors[i % len(colors)]

    axes[1].plot(
        df['epoch'], df['train_acc'],
        label=f"{model}-train",
        color=color,
        linestyle='-',
        linewidth=2.2,
        alpha=0.9
    )

    axes[1].plot(
        df['epoch'], df['test_acc'],
        label=f"{model}-test",
        color=color,
        linestyle='--',
        linewidth=2.2,
        alpha=0.9
    )

axes[1].set_title("Accuracy Curve", fontsize=16)
axes[1].set_xlabel("Epoch", fontsize=12)
axes[1].set_ylabel("Accuracy", fontsize=12)
axes[1].legend(fontsize=9)


plt.suptitle("Model Performance Comparison", fontsize=18, y=0.95)

plt.tight_layout()

plt.savefig("comparison_all_models.png", dpi=300, bbox_inches='tight')
plt.show()
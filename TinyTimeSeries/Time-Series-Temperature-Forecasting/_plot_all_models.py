import os
import pandas as pd
import matplotlib.pyplot as plt

record_dir = "TrainRecord"

all_records = {}

 
for file in os.listdir(record_dir):
    if file.endswith("_record.csv"):
        model_name = file.replace("_record.csv", "")
        df = pd.read_csv(os.path.join(record_dir, file))
        all_records[model_name] = df

plt.style.use('seaborn-v0_8-whitegrid')



colors = [
    "#7FB3D5",  # Soft Blue
    "#F7CAC9",  # Soft Pink
    "#82E0AA",  # Soft Green
    "#F8C471",  # Soft Orange Yellow
    "#C39BD3",  # Soft Purple
    "#76D7C4",  # Soft Green
    "#A9CCE3",  # Pale Sky Blue
    "#F5B7B1",  # Light Coral Pink
    "#A3E4D7",  # Light mint green
    "#FAD7A0",  # Soft Apricot Yellow
    "#D7BDE2",  # Light Lavender Purple
    "#85C1E9",  # Bright Soft Blue
    "#ABEBC6",  # Fresh Light Green
    "#F9E79F",  # Soft Cream Yellow
    "#D2B4DE",  # Lilac Purple
    "#7FC3C3"   # Soft Gray-Blue
]

 
plt.figure(figsize=(10, 6))

for i, (model, df) in enumerate(all_records.items()):
    color = colors[i % len(colors)]

    plt.plot(
        df["epoch"],
        df["train_mae_c"],
        label=f"{model}-train",
        color=color,
        linestyle='-',
        linewidth=2.2
    )

    plt.plot(
        df["epoch"],
        df["val_mae_c"],
        label=f"{model}-val",
        color=color,
        linestyle='--',
        linewidth=2.2
    )

plt.title("Temperature Forecasting Performance (MAE)", fontsize=16)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("MAE (°C)", fontsize=12)
plt.legend(fontsize=9)

plt.tight_layout()
plt.savefig("temperature_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

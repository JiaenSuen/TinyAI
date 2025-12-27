import os
import pandas as pd
import matplotlib.pyplot as plt


def visual_EDA():
    df = pd.read_csv("laptop_price.csv", encoding="latin1")

    
    os.makedirs("EDA/numeric", exist_ok=True)
    os.makedirs("EDA/categorical", exist_ok=True)
    os.makedirs("EDA/target", exist_ok=True)

    plt.rcParams["figure.figsize"] = (8, 5)
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in num_cols:
        plt.figure()
        df[col].dropna().hist(bins=30)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(f"EDA/numeric/{col}_distribution.png")
        plt.close()
    for col in num_cols:
        plt.figure()
        plt.boxplot(df[col].dropna(), vert=False)
        plt.title(f"Boxplot of {col}")
        plt.xlabel(col)
        plt.tight_layout()
        plt.savefig(f"EDA/numeric/{col}_boxplot.png")
        plt.close()

    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        plt.figure()
        df[col].value_counts().head(20).plot(kind="bar")
        plt.title(f"Top Categories of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(f"EDA/categorical/{col}_count.png")
        plt.close()


    plt.figure()
    df["Price_euros"].hist(bins=40)
    plt.title("Price Distribution")
    plt.xlabel("Price")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("EDA/target/price_distribution.png")
    plt.close()


    top_companies = df["Company"].value_counts().head(10).index

    plt.figure()
    df[df["Company"].isin(top_companies)] \
        .boxplot(column="Price_euros", by="Company", rot=45)

    plt.title("Price by Company (Top 10)")
    plt.suptitle("")
    plt.xlabel("Company")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.savefig("EDA/target/price_by_company.png")
    plt.close()


    for col in num_cols:
        if col == "Price_euros":
            continue
        plt.figure()
        plt.scatter(df[col], df["Price_euros"], alpha=0.5)
        plt.xlabel(col)
        plt.ylabel("Price")
        plt.title(f"{col} vs Price")
        plt.tight_layout()
        plt.savefig(f"EDA/target/{col}_vs_price.png")
        plt.close()

    
    corr = df[num_cols].corr()

    plt.figure(figsize=(10, 8))
    im = plt.imshow(corr, aspect="auto")
    plt.colorbar(im)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.title("Correlation Heatmap (Numerical Features)")
    plt.tight_layout()
    plt.savefig("EDA/numeric/correlation_heatmap.png")
    plt.close()

 
import pickle
import pandas as pd


class CategoricalEncoder:
    def __init__(self):
        self.encoders = {}   # col -> {category: int}
        self.decoders = {}   # col -> {int: category}
        self.fitted = False

    def fit(self, df: pd.DataFrame):
        cat_cols = df.select_dtypes(include=["object"]).columns
        for col in cat_cols:
            unique_vals = (
                df[col]
                .astype(str)
                .fillna("__UNK__")
                .unique()
            )
            encoder = {"__UNK__": 0}
            decoder = {0: "__UNK__"}
            idx = 1
            for val in unique_vals:
                if val not in encoder:
                    encoder[val] = idx
                    decoder[idx] = val
                    idx += 1
            self.encoders[col] = encoder
            self.decoders[col] = decoder
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Encoder has not been fitted yet.")

        df_encoded = df.copy()

        for col, encoder in self.encoders.items():
            df_encoded[col] = (
                df_encoded[col]
                .astype(str)
                .fillna("__UNK__")
                .map(lambda x: encoder.get(x, 0))
            )

        return df_encoded

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Encoder has not been fitted yet.")
        df_decoded = df.copy()
        for col, decoder in self.decoders.items():
            df_decoded[col] = df_decoded[col].map(
                lambda x: decoder.get(x, "__UNK__")
            )
        return df_decoded

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str):
        with open(path, "rb") as f:
            return pickle.load(f)









 




import re
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class FeatureEncoderV2:
    """
    FeatureEncoderV2
    - Deeply structured CPU/GPU features
    - Product uses TF-IDF
    - All outputs are numerical (XGBoost-safe)
    - Column names are automatically normalized (GPU/Gpu/gpu)
    """

    def __init__(self, max_tfidf_features=100):
        self.max_tfidf_features = max_tfidf_features
        self.tfidf = TfidfVectorizer(
            max_features=max_tfidf_features,
            ngram_range=(1, 2),
            stop_words="english"
        )
        self.fitted = False

    # Column normalization
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.rename(columns={
            "GPU": "Gpu",
            "gpu": "Gpu"
        })

    # Parsing helpers
    def _parse_ram(self, x):
        m = re.search(r"(\d+)", str(x))
        return int(m.group(1)) if m else 0

    def _parse_weight(self, x):
        m = re.search(r"([\d\.]+)", str(x))
        return float(m.group(1)) if m else 0.0

    def _parse_cpu(self, x):
        x = str(x).lower()

        # brand
        if "intel" in x:
            brand = 0
        elif "amd" in x:
            brand = 1
        else:
            brand = 2

        # tier
        if "i9" in x or "ryzen 9" in x:
            tier = 4
        elif "i7" in x or "ryzen 7" in x:
            tier = 3
        elif "i5" in x or "ryzen 5" in x:
            tier = 2
        elif "i3" in x or "ryzen 3" in x:
            tier = 1
        else:
            tier = 0

        # generation (Intel-like)
        m = re.search(r"(\d{4,5})", x)
        gen = int(m.group(1)[0]) if m else 0

        return brand, tier, gen

    def _parse_gpu(self, x):
        x = str(x).lower()

        if "intel" in x or "uhd" in x or "iris" in x:
            return 0  # integrated
        if "mx" in x or "vega" in x:
            return 1  # low
        if "gtx" in x:
            return 2  # mid
        if "rtx" in x:
            return 3  # high
        return 0

    def _parse_screen(self, x):
        x = str(x).lower()
        m = re.search(r"(\d+)x(\d+)", x)
        w, h = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
        pixels = w * h
        is_4k = int(pixels >= 8_000_000)
        ips = int("ips" in x)
        touch = int("touch" in x)
        return w, h, pixels, is_4k, ips, touch

    def _parse_memory(self, x):
        x = str(x).lower()
        total = sum(int(s) for s in re.findall(r"(\d+)\s*gb", x))
        has_ssd = int("ssd" in x)
        has_hdd = int("hdd" in x)
        return total, has_ssd, has_hdd

    # FIT
    def fit(self, df: pd.DataFrame):
        df = self._normalize_columns(df)
        self.tfidf.fit(df["Product"].fillna(""))
        self.fitted = True
        return self

    # TRANSFORM
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Encoder not fitted")

        df = self._normalize_columns(df.copy())

        # CPU
        df[["cpu_brand", "cpu_tier", "cpu_gen"]] = df["Cpu"].apply(
            lambda x: pd.Series(self._parse_cpu(x))
        )

        # GPU
        df["gpu_level"] = df["Gpu"].apply(self._parse_gpu)

        # RAM / Weight
        df["ram_gb"] = df["Ram"].apply(self._parse_ram)
        df["weight_kg"] = df["Weight"].apply(self._parse_weight)

        # Screen
        df[["screen_w", "screen_h", "screen_pixels",
            "is_4k", "is_ips", "is_touch"]] = df["ScreenResolution"].apply(
            lambda x: pd.Series(self._parse_screen(x))
        )

        # Memory
        df[["mem_total_gb", "has_ssd", "has_hdd"]] = df["Memory"].apply(
            lambda x: pd.Series(self._parse_memory(x))
        )

        # TF-IDF (Product)
        tfidf_mat = self.tfidf.transform(df["Product"].fillna(""))
        tfidf_df = pd.DataFrame(
            tfidf_mat.toarray(),
            columns=[f"prod_tfidf_{i}" for i in range(tfidf_mat.shape[1])],
            index=df.index
        )

        df = pd.concat([df, tfidf_df], axis=1)

        # Drop raw / non-ML columns
        drop_cols = [
            "laptop_ID",
            "Company", "Product", "TypeName",
            "Cpu", "Ram", "Memory", "Gpu",
            "OpSys", "Weight", "ScreenResolution"
        ]

        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        # Final safety: remove any object dtype
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = 0

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # SAVE / LOAD
    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str):
        with open(path, "rb") as f:
            return pickle.load(f)

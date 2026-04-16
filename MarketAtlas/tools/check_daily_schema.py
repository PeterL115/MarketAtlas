import pandas as pd

def describe(path):
    df = pd.read_parquet(path)
    print("\n==", path, "==")
    print("shape:", df.shape)
    print("cols:", list(df.columns))
    print("dtypes:\n", df.dtypes)
    print("head:\n", df.head(3))

describe(r"C:\Users\70588\OneDrive\Desktop\MarketAtlas\data\daily\SPY.parquet")
describe(r"C:\Users\70588\OneDrive\Desktop\MarketAtlas\data\daily\META.parquet")

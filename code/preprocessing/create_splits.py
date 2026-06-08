"""Train/Val Split — 85/15 stratified random split with fixed seed."""
import pandas as pd

# --- Configuration ---
INPUT_PATH = '/kaggle/input/amazon-ml-dataset-csv/dataset/train.csv'
TRAIN_OUT = '/kaggle/working/train.csv'
VAL_OUT = '/kaggle/working/val.csv'
SPLIT_RATIO = 0.85
SEED = 42

def main():
    df = pd.read_csv(INPUT_PATH)
    train_df = df.sample(frac=SPLIT_RATIO, random_state=SEED)
    val_df = df.drop(train_df.index)

    train_df.to_csv(TRAIN_OUT, index=False)
    val_df.to_csv(VAL_OUT, index=False)

    print(f"Total: {len(df)} | Train: {len(train_df)} | Val: {len(val_df)}")
    print(f"Saved → {TRAIN_OUT}, {VAL_OUT}")

if __name__ == "__main__":
    main()

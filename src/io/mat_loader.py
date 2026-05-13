import os
import pandas as pd


def load_mat_csv(file_path):
    """Load a pressure mat CSV (skip the 15-row header, tag each row as left/right)."""
    df = pd.read_csv(file_path).iloc[15:].reset_index(drop=True)
    df["Side"] = df.iloc[:, 1].apply(lambda x: "left" if "Left" in str(x) else "right")
    return df


def convert_mat_txt_folder(folder_path):
    """Convert all .txt MAT files in a folder to CSV files alongside them."""
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".txt"):
                path = os.path.join(root, file)
                df = pd.read_csv(path, delimiter=";", skiprows=5)
                out_path = os.path.join(root, f"_{file.replace('.txt', '.csv')}")
                df.to_csv(out_path, index=False)
                print(f"Saved: {out_path}")

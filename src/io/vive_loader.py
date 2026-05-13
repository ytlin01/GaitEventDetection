import os
import msgpack
import pandas as pd


def load_vive_csv(file_path):
    """Load a pre-exported VIVE CSV and keep only synced frames."""
    df = pd.read_csv(file_path).astype(float)
    df = df[df["Sync"] == 1].reset_index(drop=True)
    return df


def convert_msgpack_folder(folder_path):
    """Convert all .msgpack files in a folder to CSV files alongside them."""
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".msgpack"):
                path = os.path.join(root, file)
                raw_df = _read_msgpack(path)
                expanded = _expand_msgpack_df(raw_df)
                out_df = raw_df.drop(columns=["H", "Info", "Infof"], errors="ignore").join(expanded)
                out_path = path.replace(".msgpack", "_expanded.csv")
                out_df.to_csv(out_path, index=False)
                print(f"Saved: {out_path}")


def _read_msgpack(file_path):
    data = []
    with open(file_path, "rb") as f:
        unpacker = msgpack.Unpacker(f, raw=False)
        for item in unpacker:
            data.append(item)
    return pd.DataFrame(data)


def _expand_msgpack_df(df):
    rows = []
    for _, row in df.iterrows():
        row_data = {}
        for item in row["H"]:
            tag = item["Tag"]
            for i, v in enumerate(item["O"], 1):
                row_data[f"O{i} ({tag})"] = v
            for i, v in enumerate(item["P"], 1):
                row_data[f"P{i} ({tag})"] = v
        if isinstance(row.get("Info"), dict):
            row_data.update(row["Info"])
        rows.append(row_data)
    return pd.DataFrame(rows)

from pathlib import Path
import pandas as pd


def export_frames(frames: dict, output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for name, df in frames.items():
        filename = f"{name}.csv"
        df.to_csv(out / filename, index=False)
        manifest_rows.append({'artifact': filename, 'type': 'csv'})
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(out / 'analysis_manifest.csv', index=False)
    return manifest

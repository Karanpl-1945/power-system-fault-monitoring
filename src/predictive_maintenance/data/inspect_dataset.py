from __future__ import annotations

import argparse

from predictive_maintenance.data.protect90 import load_labels, load_waveform, missing_waveforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect PROTECT-90 labels and waveform files.")
    parser.add_argument("--labels", default="hv_double_line_90kv_labels.csv")
    parser.add_argument("--waveform-dir", default="hv_double_line_90kv_preprocessed_data")
    parser.add_argument("--sample-id", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = load_labels(args.labels)
    missing = missing_waveforms(labels, args.waveform_dir)

    print(f"labels_shape={labels.shape}")
    print(f"label_columns={list(labels.columns)}")
    print(f"missing_waveform_count={len(missing)}")
    if missing:
        print(f"first_missing_ids={missing[:10]}")

    waveform = load_waveform(args.waveform_dir, args.sample_id)
    print(f"sample_id={args.sample_id}")
    print(f"waveform_shape={waveform.shape}")
    print(f"waveform_columns={list(waveform.columns)}")
    print(waveform.head(3).to_string(index=False))


if __name__ == "__main__":
    main()


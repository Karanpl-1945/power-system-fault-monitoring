from __future__ import annotations

import argparse
import json
import time

from confluent_kafka import Producer

from predictive_maintenance.data.protect90 import load_labels, load_waveform
from predictive_maintenance.features.dataset import waveform_channels
from predictive_maintenance.features.windowing import iter_windows, window_fault_label
from predictive_maintenance.streaming.schemas import WaveformWindowEvent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay PROTECT-90 waveform windows into Kafka.")
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--topic", default="power.waveform.windows")
    parser.add_argument("--labels", default="hv_double_line_90kv_labels.csv")
    parser.add_argument("--waveform-dir", default="hv_double_line_90kv_preprocessed_data")
    parser.add_argument("--sample-id", type=int, default=0)
    parser.add_argument("--window-samples", type=int, default=640)
    parser.add_argument("--stride-samples", type=int, default=128)
    parser.add_argument("--channel-limit", type=int, default=48)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    return parser.parse_args()


def delivery_report(error, message) -> None:
    if error is not None:
        print(f"delivery_failed={error}", flush=True)
    else:
        print(
            f"delivered topic={message.topic()} partition={message.partition()} offset={message.offset()}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    labels = load_labels(args.labels)
    label_row = labels.set_index("sample_id").loc[args.sample_id]
    waveform = load_waveform(args.waveform_dir, args.sample_id)
    channels = waveform_channels(waveform)[: args.channel_limit]

    producer = Producer({"bootstrap.servers": args.bootstrap_servers})

    sent = 0
    for window_index, (start_idx, end_idx, window) in enumerate(
        iter_windows(waveform, args.window_samples, args.stride_samples),
        start=1,
    ):
        if args.max_windows is not None and window_index > args.max_windows:
            break

        start_time = float(window.iloc[0]["time_s"])
        end_time = float(window.iloc[-1]["time_s"])
        true_label = window_fault_label(
            start_time,
            end_time,
            float(label_row["t_evnt_start"]),
            float(label_row["t_evnt_end"]),
        )
        event = WaveformWindowEvent(
            sample_id=args.sample_id,
            window_index=window_index,
            start_idx=start_idx,
            end_idx=end_idx,
            window_start_time=start_time,
            window_end_time=end_time,
            true_window_label=true_label,
            channels={
                channel: window[channel].astype(float).tolist()
                for channel in channels
            },
            context={
                "phase_select": int(label_row["phase_select"]),
                "fault_resistance": float(label_row["fault_resistance"]),
                "sc_location": float(label_row["sc_location"]),
            },
        )
        producer.produce(
            args.topic,
            key=str(args.sample_id),
            value=event.model_dump_json(),
            callback=delivery_report,
        )
        producer.poll(0)
        sent += 1
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    producer.flush()
    print(f"sent_windows={sent}", flush=True)


if __name__ == "__main__":
    main()

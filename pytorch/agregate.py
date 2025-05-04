import os
import shutil
import argparse
import re


import os
import shutil
import re
from collections import defaultdict


def aggregate_dataset(input_dir, output_dir):
    pattern = re.compile(r'^[FM]\d+-(?P<emotion>[^-]+)-')
    counters = defaultdict(int)

    if not os.path.isdir(input_dir):
        raise ValueError(f"{input_dir} does not exist")
    os.makedirs(output_dir, exist_ok=True)

    for root, _, files in os.walk(input_dir):
        folder = os.path.basename(root)
        m = pattern.match(folder)
        if not m:
            continue

        emotion = m.group('emotion').lower()
        dest_dir = os.path.join(output_dir, emotion)
        os.makedirs(dest_dir, exist_ok=True)

        # Use folder name as a tag, sanitized
        tag = re.sub(r'[^A-Za-z0-9]+', '_', folder)

        for fname in files:
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            counters[emotion] += 1
            ext = os.path.splitext(fname)[1]
            # e.g. anger_000123_F01-Anger-Turn_Away.jpg
            new_name = f"{emotion}_{counters[emotion]:06d}_{tag}{ext}"
            src = os.path.join(root, fname)
            dst = os.path.join(dest_dir, new_name)
            shutil.copy2(src, dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate dataset by emotion")
    parser.add_argument("--input_dir", required=True,
                        help="Root directory containing nested F##-Emotion-... folders")
    parser.add_argument("--output_dir", required=True,
                        help="Target directory for structured dataset")
    args = parser.parse_args()
    aggregate_dataset(args.input_dir, args.output_dir)

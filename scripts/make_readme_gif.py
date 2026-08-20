"""Convert a recorded Contra evaluation video into a compact README GIF."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Input MP4 file")
    parser.add_argument("output", type=Path, help="Output GIF file")
    parser.add_argument("--fps", type=float, default=8.0, help="GIF frame rate")
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--max-seconds", type=float, default=12.0)
    parser.add_argument("--width", type=int, default=256, help="Output width in pixels")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fps <= 0 or args.max_seconds <= 0 or args.width <= 0:
        raise ValueError("fps, max-seconds, and width must be positive")

    capture = cv2.VideoCapture(str(args.source))
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if not capture.isOpened() or source_fps <= 0:
        raise ValueError(f"could not open video: {args.source}")

    start_frame = round(args.start_seconds * source_fps)
    stop_frame = start_frame + round(args.max_seconds * source_fps)
    sample_every = max(1, round(source_fps / args.fps))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames: list[Image.Image] = []
    frame_index = start_frame
    while frame_index < stop_frame:
        success, frame = capture.read()
        if not success:
            break
        if (frame_index - start_frame) % sample_every == 0:
            height, width = frame.shape[:2]
            target_height = round(height * args.width / width)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (args.width, target_height), interpolation=cv2.INTER_AREA)
            frames.append(Image.fromarray(rgb).quantize(colors=256))
        frame_index += 1
    capture.release()

    if not frames:
        raise ValueError("the selected video segment contains no frames")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=round(1000 / args.fps),
        loop=0,
        disposal=2,
        optimize=True,
    )
    print(f"wrote {args.output} ({len(frames)} frames at {args.fps:g} FPS)")


if __name__ == "__main__":
    main()

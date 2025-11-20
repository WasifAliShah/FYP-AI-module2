"""
Video Preprocessing Script (CPU-Friendly)
Upscales, denoises, deblocks, and fixes compression issues
before sending video into your face/ReID pipeline.
"""

import subprocess
import shlex
import sys
import os

def preprocess_video(input_path, output_path,
                     width=1280, crf=18, preset="slow", bitrate="4M"):
    """
    Preprocess video using FFmpeg:
    - Upscale to given width (height auto)
    - Remove compression noise / block artifacts
    - Improve color clarity slightly
    - Re-encode cleanly for better model performance
    """

    # Video filter pipeline: scale + slight color/tone fix + denoise
    vf = (
        f"scale={width}:-2,"        # upscale while preserving aspect ratio
        "format=yuv420p,"          # safe pixel format
        "eq=gamma=1.0:saturation=1.05,"  # subtle clarity & saturation boost
        "hqdn3d=1.5:1.5:6:6"       # spatial+temporal denoiser
    )

    # Final FFmpeg command
    cmd = (
        f'ffmpeg -y -i {shlex.quote(input_path)} '
        f'-vf "{vf}" '
        f'-c:v libx264 -preset {preset} -crf {crf} -b:v {bitrate} '
        f'-c:a copy '
        f'{shlex.quote(output_path)}'
    )

    print("\n-------------------------")
    print("Running FFmpeg Preprocessing")
    print("-------------------------")
    print(cmd)
    print()

    # Run FFmpeg
    process = subprocess.run(cmd, shell=True)

    if process.returncode == 0:
        print("\n✅ Preprocessing complete!")
        print(f"Output saved to: {output_path}")
    else:
        print("\n❌ Preprocessing failed.")
        print("Check your FFmpeg installation.")


# -------------------------------
# Command-line usage
# -------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("\nUsage:")
        print("  python preprocess_video.py input.mp4 output_upscaled.mp4\n")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Check input exists
    if not os.path.isfile(input_file):
        print(f"❌ Error: Input file not found: {input_file}")
        sys.exit(1)

    preprocess_video(input_file, output_file)

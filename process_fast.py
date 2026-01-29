"""
FastBlend Fast Processing Script

Quick processing with reduced quality settings.
Estimated time: ~15-25 minutes per video (depending on resolution)
"""

import os
import sys

# Add torch CUDA libraries to PATH before importing cupy
script_dir = os.path.dirname(os.path.abspath(__file__))
torch_lib_path = os.path.join(script_dir, '.venv', 'Lib', 'site-packages', 'torch', 'lib')
if os.path.exists(torch_lib_path):
    os.environ['PATH'] = torch_lib_path + os.pathsep + os.environ.get('PATH', '')
    os.add_dll_directory(torch_lib_path)

sys.path.insert(0, script_dir)

from FastBlend.api import smooth_video

# ============================================
# CONFIGURATION - Edit these paths as needed
# ============================================

# Single video mode
VIDEO_PATH = r"C:\Users\sa095\Desktop\antiflick\sources\test2.mp4"
OUTPUT_DIR = r"C:\Users\sa095\Desktop\antiflick\results\test2_fast"

# Fast settings (lower quality, faster processing)
SETTINGS = {
    "mode": "Fast",
    "window_size": 5,        # Lower = faster, less smooth
    "batch_size": 2,         # Reduce if out of memory
    "minimum_patch_size": 5,
    "num_iter": 3,           # Lower = faster, less quality
    "guide_weight": 10.0,
    "initialize": "identity"
}

# ============================================
# MAIN
# ============================================

def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"ERROR: Video not found: {VIDEO_PATH}")
        print("Please edit VIDEO_PATH in this script.")
        return

    video_name = os.path.basename(VIDEO_PATH)

    print(f"\n{'='*50}")
    print(f"FastBlend Fast Processing")
    print(f"{'='*50}")
    print(f"Video: {video_name}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Mode: {SETTINGS['mode']} | Window: {SETTINGS['window_size']} | Iterations: {SETTINGS['num_iter']}")
    print(f"{'='*50}\n")

    try:
        smooth_video(
            video_guide=VIDEO_PATH,
            video_guide_folder=None,
            video_style=VIDEO_PATH,
            video_style_folder=None,
            mode=SETTINGS['mode'],
            window_size=SETTINGS['window_size'],
            batch_size=SETTINGS['batch_size'],
            tracking_window_size=0,
            output_path=OUTPUT_DIR,
            fps=None,
            minimum_patch_size=SETTINGS['minimum_patch_size'],
            num_iter=SETTINGS['num_iter'],
            guide_weight=SETTINGS['guide_weight'],
            initialize=SETTINGS['initialize']
        )

        print(f"\n{'='*50}")
        print("SUCCESS!")
        print(f"Output: {os.path.join(OUTPUT_DIR, 'video.mp4')}")
        print(f"{'='*50}")

    except MemoryError as e:
        print(f"\nMEMORY ERROR: {e}")
        print("Try reducing batch_size in SETTINGS")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

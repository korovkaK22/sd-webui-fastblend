"""
FastBlend Quality Processing Script with Checkpoint Recovery

This script processes videos with high quality settings and supports
resuming from checkpoints if processing is interrupted.

Estimated time: ~45 minutes per video (depending on resolution and length)
"""

import os
import sys
import json
import shutil
from datetime import datetime

# Add torch CUDA libraries to PATH before importing cupy
script_dir = os.path.dirname(os.path.abspath(__file__))
torch_lib_path = os.path.join(script_dir, '.venv', 'Lib', 'site-packages', 'torch', 'lib')
if os.path.exists(torch_lib_path):
    os.environ['PATH'] = torch_lib_path + os.pathsep + os.environ.get('PATH', '')
    os.add_dll_directory(torch_lib_path)

sys.path.insert(0, script_dir)

from FastBlend.api import smooth_video
from FastBlend.runners import FastModeRunner, BalancedModeRunner, AccurateModeRunner
from FastBlend.data import VideoData, get_video_fps, save_video
import numpy as np

# ============================================
# CONFIGURATION - Edit these paths as needed
# ============================================

SOURCE_DIR = r"C:\Users\sa095\Desktop\antiflick\sources"
OUTPUT_DIR = r"C:\Users\sa095\Desktop\antiflick\results"
CHECKPOINT_DIR = os.path.join(script_dir, "checkpoints")

# Quality settings (higher quality, longer processing time ~45 min)
SETTINGS = {
    "mode": "Accurate",      # "Fast", "Balanced", or "Accurate"
    "window_size": 15,       # Smoothing window (10-30)
    "batch_size": 2,         # Reduce if out of memory
    "tracking_window_size": 1,  # For Accurate mode
    "minimum_patch_size": 7,    # Increase for higher resolution
    "num_iter": 5,           # Quality iterations (3-10)
    "guide_weight": 10.0,    # Motion guide weight
    "initialize": "identity"
}

# ============================================
# CHECKPOINT SYSTEM
# ============================================

def get_checkpoint_path(video_name):
    """Get checkpoint file path for a video."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    safe_name = video_name.replace(".", "_").replace(" ", "_")
    return os.path.join(CHECKPOINT_DIR, f"{safe_name}.checkpoint.json")


def load_checkpoint(video_name):
    """Load checkpoint for a video if exists."""
    checkpoint_path = get_checkpoint_path(video_name)
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r') as f:
                return json.load(f)
        except:
            return None
    return None


def save_checkpoint(video_name, data):
    """Save checkpoint for a video."""
    checkpoint_path = get_checkpoint_path(video_name)
    data['last_updated'] = datetime.now().isoformat()
    with open(checkpoint_path, 'w') as f:
        json.dump(data, f, indent=2)


def clear_checkpoint(video_name):
    """Remove checkpoint after successful completion."""
    checkpoint_path = get_checkpoint_path(video_name)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)


# ============================================
# PROCESSING WITH CHECKPOINTS
# ============================================

def process_video_with_checkpoints(video_path, output_dir):
    """
    Process a single video with checkpoint support.

    The processing is divided into stages:
    1. Frame extraction
    2. FastBlend processing (4 steps)
    3. Video encoding

    Checkpoints are saved after each major step.
    """
    video_name = os.path.basename(video_path)
    checkpoint = load_checkpoint(video_name)

    print(f"\n{'='*60}")
    print(f"Processing: {video_name}")
    print(f"Output: {output_dir}")
    print(f"Mode: {SETTINGS['mode']} | Window: {SETTINGS['window_size']} | Iterations: {SETTINGS['num_iter']}")
    print(f"{'='*60}")

    # Check for existing checkpoint
    if checkpoint:
        print(f"Found checkpoint from {checkpoint.get('last_updated', 'unknown')}")
        print(f"Stage: {checkpoint.get('stage', 'unknown')}")

        if checkpoint.get('stage') == 'completed':
            print("Video already processed. Skipping.")
            return True

    # Initialize checkpoint
    if not checkpoint:
        checkpoint = {
            'video_name': video_name,
            'video_path': video_path,
            'output_dir': output_dir,
            'stage': 'starting',
            'settings': SETTINGS
        }

    try:
        # Stage 1: Setup
        checkpoint['stage'] = 'setup'
        save_checkpoint(video_name, checkpoint)

        os.makedirs(output_dir, exist_ok=True)
        frames_path = os.path.join(output_dir, "frames")
        os.makedirs(frames_path, exist_ok=True)

        # Stage 2: Load video data
        checkpoint['stage'] = 'loading'
        save_checkpoint(video_name, checkpoint)

        print("Loading video frames...")
        frames_guide = VideoData(video_path, None)
        frames_style = VideoData(video_path, None)

        num_frames = len(frames_guide)
        height, width = frames_guide.shape()
        fps = get_video_fps(video_path)

        print(f"Frames: {num_frames} | Resolution: {width}x{height} | FPS: {fps}")

        checkpoint['num_frames'] = num_frames
        checkpoint['resolution'] = f"{width}x{height}"
        checkpoint['fps'] = fps

        # Stage 3: FastBlend processing
        checkpoint['stage'] = 'processing'
        save_checkpoint(video_name, checkpoint)

        print(f"\nStarting FastBlend ({SETTINGS['mode']} mode)...")

        ebsynth_config = {
            "minimum_patch_size": SETTINGS['minimum_patch_size'],
            "threads_per_block": 8,
            "num_iter": SETTINGS['num_iter'],
            "gpu_id": 0,
            "guide_weight": SETTINGS['guide_weight'],
            "initialize": SETTINGS['initialize'],
            "tracking_window_size": SETTINGS['tracking_window_size'],
        }

        mode = SETTINGS['mode']
        batch_size = SETTINGS['batch_size']
        window_size = SETTINGS['window_size']

        if mode == "Fast":
            FastModeRunner().run(
                frames_guide, frames_style,
                batch_size=batch_size,
                window_size=window_size,
                ebsynth_config=ebsynth_config,
                save_path=frames_path
            )
        elif mode == "Balanced":
            BalancedModeRunner().run(
                frames_guide, frames_style,
                batch_size=batch_size,
                window_size=window_size,
                ebsynth_config=ebsynth_config,
                save_path=frames_path
            )
        elif mode == "Accurate":
            AccurateModeRunner().run(
                frames_guide, frames_style,
                batch_size=batch_size,
                window_size=window_size,
                ebsynth_config=ebsynth_config,
                save_path=frames_path
            )

        # Stage 4: Encoding
        checkpoint['stage'] = 'encoding'
        save_checkpoint(video_name, checkpoint)

        print("\nEncoding video...")
        video_output_path = os.path.join(output_dir, "video.mp4")
        save_video(frames_path, video_output_path, num_frames=num_frames, fps=fps)

        # Stage 5: Completed
        checkpoint['stage'] = 'completed'
        checkpoint['output_video'] = video_output_path
        save_checkpoint(video_name, checkpoint)

        print(f"\n{'='*60}")
        print(f"SUCCESS: {video_name}")
        print(f"Output: {video_output_path}")
        print(f"{'='*60}")

        return True

    except MemoryError as e:
        checkpoint['stage'] = 'error_memory'
        checkpoint['error'] = str(e)
        save_checkpoint(video_name, checkpoint)
        print(f"\nMEMORY ERROR: {e}")
        print("Try reducing batch_size in SETTINGS")
        return False

    except Exception as e:
        checkpoint['stage'] = 'error'
        checkpoint['error'] = str(e)
        save_checkpoint(video_name, checkpoint)
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_all_videos():
    """Process all videos in the source directory."""
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')

    if not os.path.exists(SOURCE_DIR):
        print(f"ERROR: Source directory not found: {SOURCE_DIR}")
        return

    videos = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(video_extensions)]

    if not videos:
        print(f"No videos found in {SOURCE_DIR}")
        return

    print(f"\n{'#'*60}")
    print(f"FastBlend Quality Processing")
    print(f"{'#'*60}")
    print(f"Source: {SOURCE_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Videos found: {len(videos)}")
    print(f"Settings: {SETTINGS['mode']} mode, window={SETTINGS['window_size']}, iter={SETTINGS['num_iter']}")
    print(f"{'#'*60}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = []

    for i, video_name in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] {video_name}")

        video_path = os.path.join(SOURCE_DIR, video_name)
        video_output_dir = os.path.join(OUTPUT_DIR, os.path.splitext(video_name)[0])

        success = process_video_with_checkpoints(video_path, video_output_dir)
        results.append((video_name, success))

    # Summary
    print(f"\n{'#'*60}")
    print("PROCESSING COMPLETE")
    print(f"{'#'*60}")

    successful = sum(1 for _, s in results if s)
    failed = len(results) - successful

    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if failed > 0:
        print("\nFailed videos:")
        for name, success in results:
            if not success:
                print(f"  - {name}")
        print("\nTo retry failed videos, run this script again.")


def show_checkpoints():
    """Show status of all checkpoints."""
    if not os.path.exists(CHECKPOINT_DIR):
        print("No checkpoints found.")
        return

    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.checkpoint.json')]

    if not checkpoints:
        print("No checkpoints found.")
        return

    print(f"\n{'='*60}")
    print("CHECKPOINT STATUS")
    print(f"{'='*60}")

    for cp_file in checkpoints:
        cp_path = os.path.join(CHECKPOINT_DIR, cp_file)
        try:
            with open(cp_path, 'r') as f:
                data = json.load(f)
            print(f"\n{data.get('video_name', 'Unknown')}:")
            print(f"  Stage: {data.get('stage', 'unknown')}")
            print(f"  Updated: {data.get('last_updated', 'unknown')}")
            if data.get('error'):
                print(f"  Error: {data.get('error')}")
        except:
            print(f"\n{cp_file}: Unable to read")


def clear_all_checkpoints():
    """Clear all checkpoints."""
    if os.path.exists(CHECKPOINT_DIR):
        shutil.rmtree(CHECKPOINT_DIR)
        print("All checkpoints cleared.")
    else:
        print("No checkpoints to clear.")


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FastBlend Quality Processing")
    parser.add_argument('--status', action='store_true', help='Show checkpoint status')
    parser.add_argument('--clear', action='store_true', help='Clear all checkpoints')
    parser.add_argument('--source', type=str, help='Override source directory')
    parser.add_argument('--output', type=str, help='Override output directory')

    args = parser.parse_args()

    if args.status:
        show_checkpoints()
    elif args.clear:
        clear_all_checkpoints()
    else:
        if args.source:
            SOURCE_DIR = args.source
        if args.output:
            OUTPUT_DIR = args.output

        process_all_videos()

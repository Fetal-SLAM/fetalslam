"""Run a bounded real-data SLAM check, including nonblank OpenGL rendering."""
import argparse
import json

import cv2
import numpy as np
from OpenGL import GL
from fetal_slam.slam import XFeatSlamOffline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile', choices=('unity', 'exvivo'))
    parser.add_argument('dataset')
    parser.add_argument('--max-frames', type=int, default=120)
    args = parser.parse_args()
    if args.max_frames <= 0:
        parser.error('--max-frames must be positive')
    unity = args.profile == 'unity'
    slam = XFeatSlamOffline(
        'config/param/vo_unity.yaml' if unity else 'config/param/vo_fast.yaml',
        'config/param/feto_cam.yaml' if unity else 'config/param/feto_cam_exvivo.yaml',
        args.dataset, data_mode='image-sequence', image_subdir='images' if unity else 'undist_image')
    render = slam.viewer.render
    track = slam.tracking
    processed_frames = 0
    nonblank = False

    def checked_tracking(*values, **kwargs):
        nonlocal processed_frames
        result = track(*values, **kwargs)
        processed_frames += 1
        return result

    def checked_render(*values, **kwargs):
        nonlocal nonblank
        running = render(*values, **kwargs)
        if running and not nonblank:
            pixels = GL.glReadPixels(0, 0, slam.viewer.width, slam.viewer.height,
                                     GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
            pixels = np.frombuffer(pixels, dtype=np.uint8)
            nonblank = bool(pixels.size and np.ptp(pixels) > 0)
        return running

    slam.viewer.render = checked_render
    slam.tracking = checked_tracking
    try:
        slam.run(args.dataset, max_frames=args.max_frames)
        points = len(slam.map.get_all_map_points())
        report = dict(profile=args.profile, initialized=bool(slam.initial_flag),
                      tracking_state=slam.tracking_state, map_points=points,
                      nonblank_viewer=nonblank, processed_frames=processed_frames,
                      frame_limit=args.max_frames)
        print(json.dumps(report))
        if not (slam.initial_flag and points > 0 and nonblank
                and slam.tracking_state == 'TRACKING'
                and processed_frames == args.max_frames):
            raise RuntimeError('SLAM initialization/map/viewer smoke test failed')
    finally:
        slam.stop_workers()
        cv2.destroyAllWindows()
        slam.viewer.close()


if __name__ == '__main__':
    main()

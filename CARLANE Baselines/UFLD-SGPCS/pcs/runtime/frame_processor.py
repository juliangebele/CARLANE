import time
import torch


class FrameProcessor:
    """
    Helper class to process frame

    Provides simplified access to process_frame() method
    or a better encapsulation compared to functional approach (depending on implementation ;))
    """

    def __init__(self, model, cls_head, output_method, measure_time):
        self.model = model
        self.cls_head = cls_head
        self.output_method = output_method
        self.measure_time = measure_time
        if self.measure_time:
            self.timestamp = time.time()
            self.avg_fps = []

    def process_frames(self, frames, names, source_frames):
        """
        Process frames and pass result to output_method.

        Note: length of all supplied arrays (frames, names, source_frames) must be of the same length.

        Args:
            frames: frames to process, have to be preprocessed (scaled, as tensor, normalized)
            names: file paths - provide if possible, used by some out-modules
            source_frames: source images (only scaled to img_height & img_width, but not further processed, eg from camera) - provide if possible - used by some out modules
        """
        if self.measure_time:
            time1 = time.time()

        with torch.no_grad():
            feat = self.model(frames.cuda())
            output = self.cls_head(feat)
            output /= 0.5

        if self.measure_time:
            time2 = time.time()

        self.output_method(output, names, source_frames)

        if self.measure_time:
            real_time = (time.time() - self.timestamp) / len(output)
            real_time_wo_out = (time2 - self.timestamp) / len(output)
            inference_time = (time2 - time1) / len(output)
            print(f'input + inference + output: {round(1 / real_time)} FPS, '
                  f'input + inference: {round(1 / real_time_wo_out)} FPS, '
                  f'inference: {round(1 / inference_time)} FPS\n'
                  f'input + inference + output: {real_time:.4f}ms, '
                  f'input + inference: {real_time_wo_out:.4f}ms, '
                  f'inference: {inference_time:.4f}ms',
                  flush=True)
            self.avg_fps.append((1.0 / real_time, 1.0 / real_time_wo_out, 1.0 / inference_time))
            self.timestamp = time.time()

    def __del__(self):
        if self.measure_time:
            fps_average = [(round(sum(y) / len(y))) for y in zip(*self.avg_fps)]
            print(f'Average FPS: \n'
                  f'input + inference + output: {fps_average[0]} FPS\n'
                  f'input + inference: {fps_average[1]} FPS\n'
                  f'inference: {fps_average[2]} FPS')

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


class EpisodeVideoRecorder:
    """Stream rendered RGB frames into one episode MP4."""

    def __init__(self,
                 output_dir: str | os.PathLike,
                 fps: int = 20,
                 worker_id: str | None = None) -> None:
        if not isinstance(output_dir, (str, os.PathLike)):
            raise TypeError('video output_dir must be a path')
        if not str(output_dir).strip():
            raise ValueError('video output_dir cannot be empty')
        if isinstance(fps, bool) or not isinstance(fps, int):
            raise TypeError('video fps must be an integer')
        if fps < 1:
            raise ValueError('video fps must be positive')
        if worker_id is not None:
            if not isinstance(worker_id, str) or not worker_id.strip():
                raise ValueError('video worker_id must be non-empty or None')
            worker_id = worker_id.strip()

        self.output_dir = Path(output_dir).expanduser().resolve()
        self.fps = fps
        self.worker_id = worker_id
        self._writer = None
        self._frame_shape: tuple[int, int, int] | None = None
        self._temporary_path: Path | None = None
        self._base_stem: str | None = None
        self._frames_written = 0

    def start(self, task_id: str, episode_index: int) -> None:
        self.abort()
        if isinstance(episode_index,
                      bool) or not isinstance(episode_index, int):
            raise TypeError('episode_index must be an integer')
        if episode_index < 0:
            raise ValueError('episode_index must be non-negative')
        stem = f'{_safe_name(task_id)}_episode_{episode_index:07d}'
        if self.worker_id is not None:
            stem += f'_worker_{_safe_name(self.worker_id)}'
        self._base_stem = stem

    def capture(self, environment: Any) -> None:
        if self._base_stem is None:
            raise RuntimeError('video recorder must be started before capture')
        frame = environment.render()
        if frame is None:
            return
        rgb = _normalize_rgb_frame(frame)
        if self._writer is None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._temporary_path = (
                self.output_dir / f'.{self._base_stem}.recording.mp4')
            self._writer = _open_video_writer(self._temporary_path, self.fps,
                                              rgb.shape[1], rgb.shape[0])
            self._frame_shape = rgb.shape
        elif rgb.shape != self._frame_shape:
            raise ValueError(
                'Environment render frame shape changed within an episode: '
                f'{self._frame_shape} -> {rgb.shape}')
        self._writer.write(rgb)
        self._frames_written += 1

    def finish(self, success: bool) -> list[str]:
        writer = self._writer
        temporary_path = self._temporary_path
        base_stem = self._base_stem
        frames_written = self._frames_written
        if writer is None or temporary_path is None or base_stem is None:
            self._writer = None
            self._temporary_path = None
            self._base_stem = None
            self._frame_shape = None
            self._frames_written = 0
            return []
        try:
            writer.release()
            if frames_written == 0:
                temporary_path.unlink(missing_ok=True)
                return []
            tag = 'success' if success else 'fail'
            final_path = self.output_dir / f'{base_stem}_{tag}.mp4'
            os.replace(temporary_path, final_path)
            return [str(final_path)]
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
        finally:
            self._writer = None
            self._temporary_path = None
            self._base_stem = None
            self._frame_shape = None
            self._frames_written = 0

    def abort(self) -> None:
        if self._writer is not None:
            abort = getattr(self._writer, 'abort', None)
            if callable(abort):
                abort()
            else:
                self._writer.release()
        if self._temporary_path is not None:
            self._temporary_path.unlink(missing_ok=True)
        self._writer = None
        self._frame_shape = None
        self._temporary_path = None
        self._base_stem = None
        self._frames_written = 0


def _safe_name(value: Any) -> str:
    normalized = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value)).strip('._-')
    return (normalized or 'task')[:120]


def _normalize_rgb_frame(value: Any) -> np.ndarray:
    frame = np.asarray(value)
    if frame.ndim != 3 or frame.shape[2] not in {1, 3, 4}:
        raise ValueError(
            'Environment render frame must have shape [H, W, 1|3|4], got '
            f'{frame.shape}')
    if not np.issubdtype(frame.dtype, np.number) or np.issubdtype(
            frame.dtype, np.bool_):
        raise TypeError('Environment render frame must be numeric')
    if not np.isfinite(frame).all():
        raise ValueError('Environment render frame contains NaN or infinity')
    if frame.shape[2] == 1:
        frame = np.repeat(frame, 3, axis=2)
    elif frame.shape[2] == 4:
        frame = frame[..., :3]
    if np.issubdtype(frame.dtype, np.floating):
        minimum = float(frame.min())
        maximum = float(frame.max())
        if 0.0 <= minimum and maximum <= 1.0:
            frame = frame * 255.0
        elif -1.0 <= minimum and maximum <= 1.0:
            frame = (frame + 1.0) * 127.5
    return np.ascontiguousarray(np.clip(frame, 0, 255).astype(np.uint8))


def _open_video_writer(path: Path, fps: int, width: int, height: int) -> Any:
    return _FFmpegVideoWriter(path, fps, width, height)


class _FFmpegVideoWriter:

    def __init__(self, path: Path, fps: int, width: int, height: int) -> None:
        self.path = path
        self._process = subprocess.Popen(
            [
                _ffmpeg_executable(),
                '-y',
                '-loglevel',
                'error',
                '-f',
                'rawvideo',
                '-pixel_format',
                'rgb24',
                '-video_size',
                f'{width}x{height}',
                '-framerate',
                str(fps),
                '-i',
                '-',
                '-an',
                '-vf',
                'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                '-vcodec',
                'libx264',
                '-preset',
                'veryfast',
                '-crf',
                '23',
                '-pix_fmt',
                'yuv420p',
                '-movflags',
                '+faststart',
                str(path),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        process = self._require_process()
        if process.stdin is None:
            raise RuntimeError('ffmpeg video input pipe is unavailable')
        try:
            process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as exc:
            message = self._close_process(process)
            self._process = None
            raise OSError(f'ffmpeg stopped while encoding video: {message}') \
                from exc

    def release(self) -> None:
        process = self._process
        if process is None:
            return
        self._process = None
        message = self._close_process(process)
        if process.returncode != 0:
            raise OSError(
                f'ffmpeg failed while finalizing {self.path}: {message}')

    def abort(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        try:
            process.kill()
        except ProcessLookupError:
            pass
        if process.stdin is not None:
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass
        process.wait()

    def _require_process(self) -> subprocess.Popen:
        if self._process is None:
            raise RuntimeError('Cannot write to a closed video encoder')
        return self._process

    @staticmethod
    def _close_process(process: subprocess.Popen) -> str:
        if process.stdin is not None:
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass
        stderr = b''
        if process.stderr is not None:
            stderr = process.stderr.read()
        process.wait()
        return stderr.decode(errors='replace').strip() or 'unknown error'


def _ffmpeg_executable() -> str:
    executable = shutil.which('ffmpeg')
    if executable is not None:
        return executable
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            'H.264 video recording requires ffmpeg or imageio-ffmpeg') from exc
    return imageio_ffmpeg.get_ffmpeg_exe()

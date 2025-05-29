# We modified the original AVReader class of decord to solve the problem of memory leak.
# For more details, refer to: https://github.com/dmlc/decord/issues/208

import numpy as np
from decord.video_reader import VideoReader
from decord.audio_reader import AudioReader
import os
import torch

from decord.ndarray import cpu
from decord import ndarray as _nd
from decord.bridge import bridge_out


class AVReader(object):
    """Individual audio video reader with convenient indexing function.

    Parameters
    ----------
    uri: str
        Path of file.
    ctx: decord.Context
        The context to decode the file, can be decord.cpu() or decord.gpu().
    sample_rate: int, default is -1
        Desired output sample rate of the audio, unchanged if `-1` is specified.
    mono: bool, default is True
        Desired output channel layout of the audio. `True` is mono layout. `False` is unchanged.
    width : int, default is -1
        Desired output width of the video, unchanged if `-1` is specified.
    height : int, default is -1
        Desired output height of the video, unchanged if `-1` is specified.
    num_threads : int, default is 0
        Number of decoding thread, auto if `0` is specified.
    fault_tol : int, default is -1
        The threshold of corupted and recovered frames. This is to prevent silent fault
        tolerance when for example 50% frames of a video cannot be decoded and duplicate
        frames are returned. You may find the fault tolerant feature sweet in many cases,
        but not for training models. Say `N = # recovered frames`
        If `fault_tol` < 0, nothing will happen.
        If 0 < `fault_tol` < 1.0, if N > `fault_tol * len(video)`, raise `DECORDLimitReachedError`.
        If 1 < `fault_tol`, if N > `fault_tol`, raise `DECORDLimitReachedError`.
    """

    def __init__(self, uri, ctx=None, sample_rate=-1, mono=True, width=-1, height=-1, num_threads=0, fault_tol=-1):
        self.uri = uri
        self.ctx = ctx
        self.sample_rate = sample_rate
        self.mono = mono
        self.width = width
        self.height = height
        self.num_threads = num_threads
        self.fault_tol = fault_tol
        
        # Initialize readers with optimized settings
        self.video_reader = VideoReader(
            uri,
            ctx=ctx,
            width=width,
            height=height,
            num_threads=num_threads or os.cpu_count(),
            fault_tol=fault_tol
        )
        
        self.audio_reader = AudioReader(
            uri,
            sample_rate=sample_rate,
            mono=mono
        )
        
        # Cache for frequently accessed frames
        self._frame_cache = {}
        self._max_cache_size = 100  # Adjust based on available memory
        
    def _get_frame(self, idx):
        """Get frame with caching"""
        if idx in self._frame_cache:
            return self._frame_cache[idx]
            
        frame = self.video_reader[idx]
        
        # Update cache
        if len(self._frame_cache) >= self._max_cache_size:
            # Remove oldest entry
            self._frame_cache.pop(next(iter(self._frame_cache)))
        self._frame_cache[idx] = frame
        
        return frame
        
    def get_batch(self, indices):
        """Get multiple frames efficiently"""
        return [self._get_frame(idx) for idx in indices]
        
    def clear_cache(self):
        """Clear frame cache"""
        self._frame_cache.clear()
        torch.cuda.empty_cache()
        
    def __del__(self):
        """Cleanup"""
        self.clear_cache()
        del self.video_reader
        del self.audio_reader

    def __len__(self):
        """Get length of the video. Note that sometimes FFMPEG reports inaccurate number of frames,
        we always follow what FFMPEG reports.
        Returns
        -------
        int
            The number of frames in the video file.
        """
        return len(self.video_reader)

    def __getitem__(self, idx):
        """Get audio samples and video frame at `idx`.

        Parameters
        ----------
        idx : int or slice
            The frame index, can be negative which means it will index backwards,
            or slice of frame indices.

        Returns
        -------
        (ndarray/list of ndarray, ndarray)
            First element is samples of shape CxS or a list of length N containing samples of shape CxS,
            where N is the number of frames, C is the number of channels,
            S is the number of samples of the corresponding frame.

            Second element is Frame of shape HxWx3 or batch of image frames with shape NxHxWx3,
            where N is the length of the slice.
        """
        assert self.video_reader is not None and self.audio_reader is not None
        if isinstance(idx, slice):
            return self.get_batch(range(*idx.indices(len(self.video_reader))))
        if idx < 0:
            idx += len(self.video_reader)
        if idx >= len(self.video_reader) or idx < 0:
            raise IndexError("Index: {} out of bound: {}".format(idx, len(self.video_reader)))
        audio_start_idx, audio_end_idx = self.video_reader.get_frame_timestamp(idx)
        audio_start_idx = self.audio_reader._time_to_sample(audio_start_idx)
        audio_end_idx = self.audio_reader._time_to_sample(audio_end_idx)
        results = (self.audio_reader[audio_start_idx:audio_end_idx], self.video_reader[idx])
        self.video_reader.seek(0)
        return results

    def _get_slice(self, sl):
        audio_arr = np.empty(shape=(self.audio_reader.shape()[0], 0), dtype="float32")
        for idx in list(sl):
            audio_start_idx, audio_end_idx = self.video_reader.get_frame_timestamp(idx)
            audio_start_idx = self.audio_reader._time_to_sample(audio_start_idx)
            audio_end_idx = self.audio_reader._time_to_sample(audio_end_idx)
            audio_arr = np.concatenate(
                (audio_arr, self.audio_reader[audio_start_idx:audio_end_idx].asnumpy()), axis=1
            )
        results = (bridge_out(_nd.array(audio_arr)), self.video_reader.get_batch(sl))
        self.video_reader.seek(0)
        return results

    def _validate_indices(self, indices):
        """Validate int64 integers and convert negative integers to positive by backward search"""
        assert self.video_reader is not None and self.audio_reader is not None
        indices = np.array(indices, dtype=np.int64)
        # process negative indices
        indices[indices < 0] += len(self.video_reader)
        if not (indices >= 0).all():
            raise IndexError("Invalid negative indices: {}".format(indices[indices < 0] + len(self.video_reader)))
        if not (indices < len(self.video_reader)).all():
            raise IndexError("Out of bound indices: {}".format(indices[indices >= len(self.video_reader)]))
        return indices

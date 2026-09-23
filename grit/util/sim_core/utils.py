"""Small generic helpers: index lookup, timers, monitor size, FPS sampling."""
from __future__ import annotations

import time
import xml.dom.minidom
import xml.etree.ElementTree as ET

import numpy as np


def get_idxs(query_list, domain_list):
    """Indices in ``query_list`` of every item of ``domain_list`` that is present."""
    if not isinstance(query_list, list) or not isinstance(domain_list, list):
        raise TypeError("[get_idxs] both inputs must be lists")
    return [query_list.index(item) for item in domain_list if item in query_list]


def trim_scale(x, th):
    """Scale ``x`` down uniformly so that ``max(|x|) <= th``."""
    x = np.array(x, copy=True)
    m = np.abs(x).max()
    if m > th:
        x = x * th / m
    return x


def get_monitor_size():
    """``(width, height)`` of the primary monitor via GLFW.

    Falls back to 1920x1080 when there is no display (headless training
    never opens a window, so the value is irrelevant there).
    """
    try:
        import glfw
        if not glfw.init():
            return 1920, 1080
        mode = glfw.get_video_mode(glfw.get_primary_monitor())
        return int(mode.size.width), int(mode.size.height)
    except Exception:
        return 1920, 1080


def indent_xml(elem) -> str:
    """Pretty-printed XML string of an ``ElementTree`` element."""
    rough = ET.tostring(elem, encoding="unicode")
    return xml.dom.minidom.parseString(rough).toprettyxml(indent="  ")


class TicToc:
    """Wall-clock stopwatch: ``tt.tic(); ...; sec = tt.toc()``."""

    def __init__(self, name="tictoc"):
        self.name = name
        self.time_start = time.time()
        self.time_elapsed = 0.0
        self.cnt = 0

    def tic(self):
        self.time_start = time.time()

    def toc(self, label=None, cnt=None, print_every=None, verbose=False):
        self.time_elapsed = time.time() - self.time_start
        if cnt is not None:
            self.cnt = cnt
        if verbose and print_every is not None and (self.cnt % print_every) == 0:
            t = self.time_elapsed
            shown, unit = (t * 1e3, "ms") if t < 1.0 else ((t, "s") if t < 60.0 else (t / 60.0, "min"))
            print(f"{label or self.name} elapsed: [{shown:.2f}]{unit}")
        self.cnt += 1
        return self.time_elapsed


class SimpleTimer:
    """Fixed-rate scheduler for render / sim loops.

    ::

        tmr = SimpleTimer(name="Render", Hz=25)
        tmr.start()
        while running:
            if tmr.do_run():      # True once per 1/Hz seconds (no catch-up burst)
                draw()
                tmr.end()         # optional: records the block's run time
    """

    def __init__(self, name="timer", Hz=10.0, max_sec=np.inf, verbose=False):
        if Hz <= 0:
            raise ValueError("Hz must be positive")
        self.name = name
        self.Hz = float(Hz)
        self.sec_period = 1.0 / self.Hz
        self.max_sec = max_sec
        self.verbose = verbose
        self.start()

    def start(self):
        self.time_start = time.perf_counter()
        self.time_run_start = None
        self.sec_next = 0.0
        self.sec_elps = 0.0
        self.sec_elps_prev = 0.0
        self.sec_elps_diff = 0.0
        self.sec_elps_loop = None
        self.tick = 0
        self.force_finish = False
        self.delayed_flag = False
        if self.verbose:
            print(f"[{self.name}] start ({self.Hz:.1f} Hz, max {self.max_sec:.1f} s)")

    def finish(self):
        self.force_finish = True

    def is_finished(self):
        if self.force_finish:
            return True
        self.sec_elps = time.perf_counter() - self.time_start
        return self.sec_elps > self.max_sec

    def is_notfinished(self):
        return not self.is_finished()

    def do_run(self):
        time.sleep(1e-6)
        self.sec_elps = time.perf_counter() - self.time_start
        if self.sec_elps > self.sec_next:
            self.sec_next += self.sec_period          # advance by ONE period (no burst catch-up)
            self.tick += 1
            self.sec_elps_diff = self.sec_elps - self.sec_elps_prev
            self.sec_elps_prev = self.sec_elps
            self.delayed_flag = self.sec_elps_diff > self.sec_period * 1.1
            if self.delayed_flag and self.verbose:
                print(f"[{self.name}][{self.tick}] delayed: interval {self.sec_elps_diff*1e3:.1f} ms "
                      f"(period {self.sec_period*1e3:.1f} ms)")
            self.time_run_start = time.perf_counter()
            return True
        self.delayed_flag = False
        return False

    def end(self):
        if self.time_run_start is None:
            return None
        self.sec_elps_loop = time.perf_counter() - self.time_run_start
        self.time_run_start = None
        return self.sec_elps_loop


def farthest_point_sampling(pcd, n_sample):
    """Greedy farthest-point sampling → indices of shape ``(n_sample,)``.

    If ``n_sample >= N`` every point is returned once and the remainder is
    filled with random (with-replacement) draws, so the output size is always
    exactly ``n_sample``.
    """
    pcd = np.asarray(pcd)
    assert pcd.ndim == 2, "pcd must be (N, D)"
    N = pcd.shape[0]
    if n_sample >= N:
        extra = np.random.choice(N, n_sample - N, replace=True)
        return np.concatenate([np.arange(N, dtype=np.int32), extra.astype(np.int32)])
    idxs = np.empty(n_sample, dtype=np.int32)
    idxs[0] = np.random.randint(0, N)
    dist = np.full(N, np.inf)
    for i in range(1, n_sample):
        last = pcd[idxs[i - 1]]
        dist = np.minimum(dist, np.linalg.norm(pcd - last, axis=1))
        idxs[i] = int(np.argmax(dist))
    return idxs

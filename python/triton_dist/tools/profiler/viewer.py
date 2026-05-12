################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################

from collections import defaultdict
from typing import Any, Dict, List, Tuple
import json
import os
import numpy as np
from .language import (
    NUM_BITS_ID,
    NUM_BITS_TASK_TYPE,
    NUM_BITS_EVENT,
)
from .context import is_empty_slot

import torch
from dataclasses import dataclass

MAX_REASONABLE_SM_ID = 4096


# adapt from flashinfer/flashinfer/profiler/__init__.py
def decode_tag(tag, num_groups):
    """
    Decode a profiler tag into (block_idx, group_idx, task_type, is_start).
    Tag layout:  GLOBAL_ID | TASK TYPE | IS START
    """
    global_id = (tag >> (NUM_BITS_TASK_TYPE + NUM_BITS_EVENT)) & ((1 << NUM_BITS_ID) - 1)
    task_type = (tag >> NUM_BITS_EVENT) & ((1 << NUM_BITS_TASK_TYPE) - 1)
    is_start = tag & NUM_BITS_EVENT
    block_idx = global_id // num_groups
    group_idx = global_id % num_groups
    assert NUM_BITS_EVENT == 1
    return block_idx, group_idx, task_type, is_start


def _track_iter(profiler_buffer: np.ndarray, num_blocks, num_groups):
    empty_count = 0
    timestamp_offset = 0
    last_timestamp = None
    for i in range(len(profiler_buffer)):
        if is_empty_slot(profiler_buffer[i]):
            empty_count += 1
            if empty_count > num_blocks * num_groups:
                return
            continue
        empty_count = 0
        tag, timestamp = profiler_buffer[i:i + 1].view(np.uint32)
        tag = int(tag)
        timestamp = int(timestamp)
        if last_timestamp is not None and last_timestamp - timestamp > (1 << 31):
            timestamp_offset += 1 << 32
        last_timestamp = timestamp
        timestamp += timestamp_offset
        block_idx, group_idx, task_type, is_start = decode_tag(tag, num_groups)
        yield block_idx, group_idx, task_type, is_start, timestamp


def _verify_and_reorg_tracks(profiler_buffer: np.ndarray, num_blocks, num_groups):
    """
    return List[(block_idx, group_idx, task_type, start_time, end_time)] sorted by start_time
    """
    tracks = {}
    records = []
    for block_idx, group_idx, task_type, is_start, timestamp in _track_iter(profiler_buffer, num_blocks, num_groups):
        track_key = block_idx, group_idx, task_type
        if is_start:
            assert track_key not in tracks, f"track ({track_key}) is opened again when it's not closed"
            tracks[track_key] = timestamp
        else:
            ts_start = tracks[track_key]
            while timestamp < ts_start:
                timestamp += 1 << 32
            assert ts_start <= timestamp
            records.append((block_idx, group_idx, task_type, ts_start, timestamp))
            tracks.pop(track_key)

    assert not tracks, "some records is not closed"
    records.sort(key=lambda x: x[3])
    return records


class Tracker:
    """ this tracker contains multiple tracks to support overlaped tracks"""

    def __init__(self, parent, track_name):
        self.parent = parent
        self.tracks = []  # (track, track_ts_end)
        self.grp = self.parent.create_group(track_name)

    def track(self,
              ts_start,
              ts_end,
              annotation: str,
              kwargs=None,
              open_flow=None,
              close_flow=None,
              open_terminating_flow=None,
              close_terminating_flow=None):
        open_flow = [] if open_flow is None else open_flow
        close_flow = [] if close_flow is None else close_flow
        open_terminating_flow = [] if open_terminating_flow is None else open_terminating_flow
        close_terminating_flow = [] if close_terminating_flow is None else close_terminating_flow
        track = self._choose_track(ts_start, ts_end)
        self._track_open(track, ts_start, annotation, kwargs, open_flow, open_terminating_flow)
        self._track_close(track, ts_end, close_flow, close_terminating_flow)

    @staticmethod
    def _track_open(track, ts, annotation, kwargs, flow, terminating_flow):
        if not terminating_flow:
            track.open(ts, annotation, kwargs=kwargs, flow=flow)
            return
        from tg4perfetto import perfetto_trace_pb2 as pb2

        parent = track._parent
        pkt = parent.trace.packet.add()
        pkt.timestamp = ts
        pkt.track_event.name_iid = parent._get_iid_for(pkt, annotation)
        pkt.trusted_packet_sequence_id = 2
        pkt.sequence_flags = 2
        pkt.track_event.category_iids.append(1)
        pkt.track_event.type = pb2.TrackEvent.TYPE_SLICE_BEGIN
        pkt.track_event.track_uuid = track._uuid

        if kwargs is not None:
            parent._add_debug_annotation(pkt.track_event.debug_annotations, kwargs)
        for flow_id in flow:
            pkt.track_event.flow_ids.append(flow_id)
        for flow_id in terminating_flow:
            pkt.track_event.terminating_flow_ids.append(flow_id)
        parent._flush_if_necessary()

    @staticmethod
    def _track_close(track, ts, flow, terminating_flow):
        if not terminating_flow:
            track.close(ts, flow=flow)
            return
        from tg4perfetto import perfetto_trace_pb2 as pb2

        parent = track._parent
        pkt = parent.trace.packet.add()
        pkt.trusted_packet_sequence_id = 2
        pkt.sequence_flags = 2
        pkt.timestamp = ts
        pkt.track_event.track_uuid = track._uuid
        pkt.track_event.type = pb2.TrackEvent.TYPE_SLICE_END
        for flow_id in flow:
            pkt.track_event.flow_ids.append(flow_id)
        for flow_id in terminating_flow:
            pkt.track_event.terminating_flow_ids.append(flow_id)
        parent._flush_if_necessary()

    def _choose_track(self, ts_start, ts_end):
        for index, (track, track_ts_end) in enumerate(self.tracks):
            if track_ts_end <= ts_start:  # track is idle
                self.tracks[index][1] = ts_end
                return track
        self.tracks.append([self.grp.create_track(), ts_end])
        return self.tracks[-1][0]


def _lookup_task_name(task_names, task_type):
    if isinstance(task_names, dict):
        return task_names[task_type]
    return task_names[task_type]


def _metadata_tasks_by_block(dependency_metadata):
    tasks_by_block = {}
    if not dependency_metadata:
        return tasks_by_block
    for queue in dependency_metadata.get("queues", []):
        block_idx = int(queue["block_idx"])
        tasks_by_block[block_idx] = {int(task["queue_index"]): task for task in queue.get("tasks", [])}
    return tasks_by_block


def _build_profile_event_map(records, task_names, dependency_metadata):
    tasks_by_block = _metadata_tasks_by_block(dependency_metadata)
    queue_cursor = defaultdict(int)
    event_map = {}
    event_key_by_record = {}

    for block_idx, group_idx, task_type, ts_start, ts_end in records:
        cur_task_name = _lookup_task_name(task_names, task_type)
        if cur_task_name == "task_decoding":
            continue

        if cur_task_name == "scoreboard_wait_deps":
            queue_index = queue_cursor[block_idx]
            event_kind = "wait"
        else:
            queue_index = queue_cursor[block_idx]
            queue_cursor[block_idx] += 1
            event_kind = "task"

        task_metadata = tasks_by_block.get(block_idx, {}).get(queue_index)
        event_map[(block_idx, queue_index, event_kind)] = {
            "block_idx": block_idx,
            "group_idx": group_idx,
            "task_type": task_type,
            "task_name": cur_task_name,
            "ts_start": ts_start,
            "ts_end": ts_end,
            "queue_index": queue_index,
            "event_kind": event_kind,
            "task_metadata": task_metadata,
        }
        event_key_by_record[(block_idx, group_idx, task_type, ts_start, ts_end)] = (block_idx, queue_index, event_kind)

    return event_map, event_key_by_record


def _event_debug_annotations(event):
    task = event.get("task_metadata")
    if task is None:
        return None

    annotations = {
        "queue_index": int(task["queue_index"]),
        "layer_id": int(task["layer_id"]),
        "task_id": int(task["task_id"]),
        "tile_id_or_start": int(task["tile_id_or_start"]),
        "num_tiles": int(task["num_tiles"]),
    }
    if event["event_kind"] == "wait" or task.get("dependencies"):
        annotations["deps_entry_start"] = int(task["deps_entry_start"])
        annotations["deps_entry_end"] = int(task["deps_entry_end"])
        annotations["dependencies"] = task.get("dependencies", [])
    return annotations


def _parse_positive_int_env(name, default):
    try:
        return max(0, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _overlaps(lhs_start, lhs_end, rhs_start, rhs_end):
    return max(lhs_start, rhs_start) < min(lhs_end, rhs_end)


def _build_dependency_flow_maps(event_map, dependency_metadata, target_wait_events: bool = True):
    open_flows = defaultdict(list)
    close_flows = defaultdict(list)
    open_terminating_flows = defaultdict(list)
    close_terminating_flows = defaultdict(list)
    if not dependency_metadata:
        return open_flows, close_flows, open_terminating_flows, close_terminating_flows, 0

    max_flows = _parse_positive_int_env("MEGA_KERNEL_TRACE_MAX_DEP_FLOWS", 20000)
    if max_flows == 0:
        return open_flows, close_flows, open_terminating_flows, close_terminating_flows, 0

    flow_mode = os.getenv("MEGA_KERNEL_TRACE_FLOW_MODE", "range").lower()
    producer_events = defaultdict(list)
    for event_key, event in event_map.items():
        if event["event_kind"] != "task":
            continue
        task = event.get("task_metadata")
        if task is None:
            continue
        tile_start = int(task["tile_id_or_start"])
        tile_end = tile_start + max(1, int(task["num_tiles"]))
        producer_events[(int(task["layer_id"]), int(task["task_id"]))].append({
            "event_key": event_key,
            "tile_start": tile_start,
            "tile_end": tile_end,
            "ts_end": event["ts_end"],
        })
    for producers in producer_events.values():
        producers.sort(key=lambda item: (item["tile_start"], item["tile_end"]))

    flow_count = 0
    next_flow_id = 1
    for consumer_key, consumer_event in event_map.items():
        if consumer_event["event_kind"] != "task":
            continue
        consumer_task = consumer_event.get("task_metadata")
        if consumer_task is None:
            continue

        wait_key = (consumer_key[0], consumer_key[1], "wait")
        target_key = wait_key if target_wait_events and wait_key in event_map else consumer_key
        target_event = event_map[target_key]
        target_flow_on_close = target_event["event_kind"] == "wait"

        for dep in consumer_task.get("dependencies", []):
            producer_key = (int(dep["producer_layer_id"]), int(dep["producer_task_id"]))
            dep_start = int(dep["producer_tile_start"])
            dep_end = int(dep["producer_tile_end"])
            matched_producers = [
                producer for producer in producer_events.get(producer_key, [])
                if _overlaps(producer["tile_start"], producer["tile_end"], dep_start, dep_end)
            ]
            if not matched_producers:
                continue
            if flow_mode != "tile":
                matched_producers = [max(matched_producers, key=lambda item: (item["tile_end"], item["tile_start"]))]

            for producer in matched_producers:
                if flow_count >= max_flows:
                    return open_flows, close_flows, open_terminating_flows, close_terminating_flows, flow_count
                source_key = producer["event_key"]
                if source_key == target_key:
                    continue
                flow_id = next_flow_id
                next_flow_id += 1
                close_flows[source_key].append(flow_id)
                if target_flow_on_close:
                    close_terminating_flows[target_key].append(flow_id)
                else:
                    open_terminating_flows[target_key].append(flow_id)
                flow_count += 1

    return open_flows, close_flows, open_terminating_flows, close_terminating_flows, flow_count


def _read_profiler_header(profiler_buffer_host: torch.Tensor, verbose: bool = False):
    header = profiler_buffer_host[:1].view(dtype=torch.int32)
    if header.numel() < 2:
        raise ValueError(
            "Profiler buffer is too short to contain global metadata. "
            f"Expected one uint64 header slot, got {profiler_buffer_host.numel()} slots."
        )
    num_blocks, num_groups = header
    num_blocks = int(num_blocks)
    num_groups = int(num_groups)
    if num_blocks <= 0:
        raise ValueError(
            "Profiler buffer metadata is not initialized: "
            f"num_blocks={num_blocks}, num_groups={num_groups}. "
            "Make sure profiling is enabled and the kernel ran before exporting the trace."
        )
    if num_groups <= 0:
        if verbose:
            print(
                "Profiler buffer has non-positive num_groups "
                f"({num_groups}); falling back to the single-group trace layout."
            )
        num_groups = 1
    return num_blocks, num_groups


def _read_block_metadata(profiler_buffer_host: torch.Tensor, num_blocks: int, verbose: bool = False):
    block_idx_to_smid = {}
    block_meta_slots = 0
    metadata = profiler_buffer_host[1:]
    for i in range(num_blocks):
        entry = metadata[i:i + 1].view(dtype=torch.uint32)
        if entry.numel() < 2:
            if verbose:
                print(
                    "Profiler buffer ended before all block metadata was read: "
                    f"read {block_meta_slots}/{num_blocks} block metadata slots."
                )
            break

        block_idx, sm_id = int(entry[0]), int(entry[1])
        if block_idx != i or sm_id >= MAX_REASONABLE_SM_ID:
            if verbose:
                print(
                    "Profiler block metadata appears truncated or absent at slot "
                    f"{i}: block_idx={block_idx}, sm_id={sm_id}. "
                    "Treating the remaining buffer as event records."
                )
            break

        block_idx_to_smid[block_idx] = sm_id
        block_meta_slots += 1

    return block_idx_to_smid, block_meta_slots


def _lookup_sm_id(block_idx_to_smid, block_idx, verbose: bool = False):
    sm_id = block_idx_to_smid.get(block_idx)
    if sm_id is not None:
        return sm_id
    if verbose:
        print(f"Missing block metadata for block_idx={block_idx}; using block_idx as the trace lane id.")
    sm_id = block_idx
    block_idx_to_smid[block_idx] = sm_id
    return sm_id


def _is_scoreboard_wait_event(event):
    return event is not None and event.get("event_kind") == "wait"


# adapt from flashinfer/flashinfer/profiler/__init__.py
def export_to_perfetto_trace(profiler_buffer: torch.Tensor, task_names: List[str], file_name: str,
                             verbose: bool = False, dependency_metadata: Dict[str, Any] = None) -> None:
    from tg4perfetto import TraceGenerator

    if not file_name.endswith(".perfetto-trace"):
        file_name = file_name + ".perfetto-trace"
    assert profiler_buffer.dtype == torch.uint64
    profiler_buffer_host = profiler_buffer.cpu()
    num_blocks, num_groups = _read_profiler_header(profiler_buffer_host, verbose)

    tgen = TraceGenerator(file_name)

    pid_map = {}
    track_map: Dict[Tuple[int, int, int], Any] = {}

    block_idx_to_smid, block_meta_slots = _read_block_metadata(profiler_buffer_host, num_blocks, verbose)
    # for better view
    if num_groups == 1:
        pid_master = tgen.create_group("tracks of all SMs")

    for block_idx, sm_id in block_idx_to_smid.items():
        if num_groups > 1:
            if block_idx not in pid_map:
                pid_map[block_idx] = tgen.create_group(f"block_{block_idx}_sm_{sm_id}")
        else:
            pid_map[block_idx] = pid_master
    if verbose:
        print(f"block_idx_to_smid = {block_idx_to_smid}, {len(block_idx_to_smid)}")

    profiler_buffer_host = profiler_buffer_host[1 + block_meta_slots:].numpy()
    records = _verify_and_reorg_tracks(profiler_buffer_host, num_blocks, num_groups)
    event_map, event_key_by_record = _build_profile_event_map(records, task_names, dependency_metadata)
    (open_flows, close_flows, open_terminating_flows, close_terminating_flows,
     flow_count) = _build_dependency_flow_maps(event_map, dependency_metadata, target_wait_events=True)
    if verbose and dependency_metadata:
        print(f"dependency flow count = {flow_count}")

    for block_idx, group_idx, task_type, ts_start, ts_end in records:
        sm_id = _lookup_sm_id(block_idx_to_smid, block_idx, verbose)
        if verbose:
            print(
                f'block_idx = {block_idx}, group_idx: {group_idx}, task_type = {task_type}, range =[{ts_start}, {ts_end}]'
            )
        # create trackers
        if num_groups > 1 and block_idx not in pid_map:
            pid_map[block_idx] = tgen.create_group(f"block_{block_idx}_sm_{sm_id}")
        pid = pid_map[block_idx] if num_groups > 1 else pid_master
        cur_task_name = task_names[task_type]
        track_key = (sm_id, )

        if (track := track_map.get(track_key, None)) is None:
            if num_groups > 1:
                track = Tracker(pid, str(sm_id))
            else:
                track = Tracker(pid, str(sm_id))
            track_map[track_key] = track

        event_key = event_key_by_record.get((block_idx, group_idx, task_type, ts_start, ts_end))
        event = event_map.get(event_key)
        kwargs = _event_debug_annotations(event) if event is not None else None
        track.track(ts_start,
                    ts_end,
                    f"{cur_task_name}:{block_idx}",
                    kwargs=kwargs,
                    open_flow=open_flows.get(event_key, []),
                    close_flow=close_flows.get(event_key, []),
                    open_terminating_flow=open_terminating_flows.get(event_key, []),
                    close_terminating_flow=close_terminating_flows.get(event_key, []))

    tgen.flush()


class _ChromeTraceTrackAllocator:
    """Allocate non-overlapping Chrome trace lanes for each SM track."""

    def __init__(self):
        self.tracks = defaultdict(list)
        self.thread_names = {}
        self.thread_sort_indices = {}

    def choose_track(self, pid, sm_id, ts_start, ts_end):
        key = (pid, sm_id)
        lanes = self.tracks[key]
        for lane_idx, lane in enumerate(lanes):
            if lane["ts_end"] <= ts_start:
                lane["ts_end"] = ts_end
                return lane["tid"]

        lane_idx = len(lanes)
        tid = int(sm_id) * 1000 + lane_idx
        lanes.append({
            "tid": tid,
            "ts_end": ts_end,
        })
        track_name = f"sm_{sm_id}" if lane_idx == 0 else f"sm_{sm_id}.{lane_idx}"
        self.thread_names[(pid, tid)] = track_name
        self.thread_sort_indices[(pid, tid)] = int(sm_id) * 1000 + lane_idx
        return tid


def _chrome_ts(timestamp_ns):
    return timestamp_ns / 1000.0


def _chrome_flow_ts(ts_start, ts_end, prefer_end):
    if ts_end <= ts_start:
        return _chrome_ts(ts_start)
    if prefer_end:
        return _chrome_ts(ts_end - 1)
    return _chrome_ts(ts_start + 1)


def _add_chrome_metadata_events(trace_events, pid_names, track_allocator):
    for sort_index, pid in enumerate(sorted(pid_names)):
        trace_events.append({
            "name": "process_name",
            "ph": "M",
            "pid": pid,
            "tid": 0,
            "args": {
                "name": pid_names[pid],
            },
        })
        trace_events.append({
            "name": "process_sort_index",
            "ph": "M",
            "pid": pid,
            "tid": 0,
            "args": {
                "sort_index": sort_index,
            },
        })

    for (pid, tid), track_name in sorted(track_allocator.thread_names.items()):
        trace_events.append({
            "name": "thread_name",
            "ph": "M",
            "pid": pid,
            "tid": tid,
            "args": {
                "name": track_name,
            },
        })
        trace_events.append({
            "name": "thread_sort_index",
            "ph": "M",
            "pid": pid,
            "tid": tid,
            "args": {
                "sort_index": track_allocator.thread_sort_indices[(pid, tid)],
            },
        })


def _chrome_event_sort_key(event):
    if event.get("ph") == "M":
        return (-1, 0, int(event.get("pid", 0)), int(event.get("tid", 0)))
    phase_order = {
        "X": 0,
        "s": 1,
        "t": 2,
        "f": 3,
    }
    return (event.get("ts", 0), phase_order.get(event.get("ph"), 9), int(event.get("pid", 0)),
            int(event.get("tid", 0)))


def _add_chrome_flow_events(trace_events, event_track_map, event_key, ts_start, ts_end, event_name, open_flows,
                            close_flows, open_terminating_flows, close_terminating_flows):
    if event_key not in event_track_map:
        return

    pid, tid = event_track_map[event_key]
    flow_name = "dependency_flow"
    for flow_id in open_flows.get(event_key, []):
        trace_events.append({
            "name": flow_name,
            "cat": "dependency",
            "ph": "s",
            "ts": _chrome_flow_ts(ts_start, ts_end, prefer_end=False),
            "pid": pid,
            "tid": tid,
            "id": int(flow_id),
            "args": {
                "slice": event_name,
            },
        })
    for flow_id in close_flows.get(event_key, []):
        trace_events.append({
            "name": flow_name,
            "cat": "dependency",
            "ph": "s",
            "ts": _chrome_flow_ts(ts_start, ts_end, prefer_end=True),
            "pid": pid,
            "tid": tid,
            "id": int(flow_id),
            "args": {
                "slice": event_name,
            },
        })
    for flow_id in open_terminating_flows.get(event_key, []):
        trace_events.append({
            "name": flow_name,
            "cat": "dependency",
            "ph": "f",
            "ts": _chrome_flow_ts(ts_start, ts_end, prefer_end=False),
            "pid": pid,
            "tid": tid,
            "id": int(flow_id),
            "bp": "e",
            "args": {
                "slice": event_name,
            },
        })
    for flow_id in close_terminating_flows.get(event_key, []):
        trace_events.append({
            "name": flow_name,
            "cat": "dependency",
            "ph": "f",
            "ts": _chrome_flow_ts(ts_start, ts_end, prefer_end=True),
            "pid": pid,
            "tid": tid,
            "id": int(flow_id),
            "bp": "e",
            "args": {
                "slice": event_name,
            },
        })


def export_to_trace(profiler_buffer: torch.Tensor, task_names: List[str], file_name: str,
                    verbose: bool = False, dependency_metadata: Dict[str, Any] = None) -> None:
    if not (file_name.endswith(".json") or file_name.endswith(".trace")):
        file_name = file_name + ".json"
    assert profiler_buffer.dtype == torch.uint64
    profiler_buffer_host = profiler_buffer.cpu()
    num_blocks, num_groups = _read_profiler_header(profiler_buffer_host, verbose)

    pid_names = {}
    pid_map = {}
    track_allocator = _ChromeTraceTrackAllocator()

    block_idx_to_smid, block_meta_slots = _read_block_metadata(profiler_buffer_host, num_blocks, verbose)
    if num_groups == 1:
        pid_master = 0
        pid_names[pid_master] = "tracks of all SMs"

    for block_idx, sm_id in block_idx_to_smid.items():
        if num_groups > 1:
            pid = block_idx
            pid_map[block_idx] = pid
            pid_names[pid] = f"block_{block_idx}_sm_{sm_id}"
        else:
            pid_map[block_idx] = pid_master
    if verbose:
        print(f"block_idx_to_smid = {block_idx_to_smid}, {len(block_idx_to_smid)}")

    profiler_buffer_host = profiler_buffer_host[1 + block_meta_slots:].numpy()
    records = _verify_and_reorg_tracks(profiler_buffer_host, num_blocks, num_groups)
    event_map, event_key_by_record = _build_profile_event_map(records, task_names, dependency_metadata)
    (open_flows, close_flows, open_terminating_flows, close_terminating_flows,
     flow_count) = _build_dependency_flow_maps(event_map, dependency_metadata, target_wait_events=False)
    if verbose and dependency_metadata:
        print(f"dependency flow count = {flow_count}")

    trace_events = []
    event_track_map = {}
    event_time_map = {}
    for block_idx, group_idx, task_type, ts_start, ts_end in records:
        sm_id = _lookup_sm_id(block_idx_to_smid, block_idx, verbose)
        if verbose:
            print(
                f'block_idx = {block_idx}, group_idx: {group_idx}, task_type = {task_type}, range =[{ts_start}, {ts_end}]'
            )

        if num_groups > 1 and block_idx not in pid_map:
            pid = block_idx
            pid_map[block_idx] = pid
            pid_names[pid] = f"block_{block_idx}_sm_{sm_id}"
        pid = pid_map[block_idx] if num_groups > 1 else pid_master
        tid = track_allocator.choose_track(pid, sm_id, ts_start, ts_end)
        cur_task_name = _lookup_task_name(task_names, task_type)
        event_key = event_key_by_record.get((block_idx, group_idx, task_type, ts_start, ts_end))
        event = event_map.get(event_key)
        if _is_scoreboard_wait_event(event):
            continue
        args = _event_debug_annotations(event) if event is not None else {}
        if args is None:
            args = {}
        args.update({
            "block_idx": block_idx,
            "group_idx": group_idx,
            "sm_id": sm_id,
            "task_type": task_type,
        })
        event_name = f"{cur_task_name}:{block_idx}"
        trace_events.append({
            "name": event_name,
            "cat": "triton_dist",
            "ph": "X",
            "ts": _chrome_ts(ts_start),
            "dur": _chrome_ts(ts_end - ts_start),
            "pid": pid,
            "tid": tid,
            "args": args,
        })
        if event_key is not None:
            event_track_map[event_key] = (pid, tid)
            event_time_map[event_key] = (ts_start, ts_end, event_name)

    for event_key, (ts_start, ts_end, event_name) in event_time_map.items():
        _add_chrome_flow_events(trace_events, event_track_map, event_key, ts_start, ts_end, event_name, open_flows,
                                close_flows, open_terminating_flows, close_terminating_flows)

    _add_chrome_metadata_events(trace_events, pid_names, track_allocator)
    trace_events.sort(key=_chrome_event_sort_key)
    trace = {
        "traceEvents": trace_events,
        "displayTimeUnit": "ns",
        "otherData": {
            "source": "triton_dist profiler",
            "timestamp_unit": "us",
        },
    }
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2)


@dataclass
class Task:
    tag: int
    task_type: int
    start_time: int  # ns
    duration: int  # ns


def parse_to_tracks(profiler_buffer: torch.Tensor):
    assert profiler_buffer.dtype == torch.uint64
    profiler_buffer_host = profiler_buffer.cpu()
    num_blocks, num_groups = profiler_buffer_host[:1].view(dtype=torch.int32)
    num_blocks = int(num_blocks)
    num_groups = int(num_groups)

    begin_timestamp_map = {}

    block_idx_to_smid = {}
    profiler_buffer_host = profiler_buffer_host[1:]
    block_idx_to_tracks = {}
    for i in range(num_blocks):
        block_idx, sm_id = profiler_buffer_host[i:i + 1].view(dtype=torch.uint32)
        block_idx, sm_id = int(block_idx), int(sm_id)
        block_idx_to_smid[block_idx] = sm_id
        block_idx_to_tracks[block_idx] = []

    profiler_buffer_host = profiler_buffer_host[num_blocks:].numpy()

    empty_count = 0
    timestamp_offset = 0
    last_timestamp = None
    for i in range(len(profiler_buffer_host)):
        if is_empty_slot(profiler_buffer_host[i]):
            empty_count += 1
            if empty_count > num_blocks * num_groups:
                break
            continue
        empty_count = 0
        tag, timestamp = profiler_buffer_host[i:i + 1].view(np.uint32)
        tag = int(tag)
        timestamp = int(timestamp)
        if last_timestamp is not None and last_timestamp - timestamp > (1 << 31):
            timestamp_offset += 1 << 32
        last_timestamp = timestamp
        timestamp += timestamp_offset
        block_idx, group_idx, task_type, is_start = decode_tag(tag, num_groups)
        sm_id = block_idx_to_smid[block_idx]

        if is_start:
            begin_timestamp_map[(block_idx, group_idx, task_type)] = timestamp
        else:
            begin_timestamp = begin_timestamp_map[(block_idx, group_idx, task_type)]
            while timestamp < begin_timestamp:
                timestamp += 1 << 32
            assert begin_timestamp <= timestamp, f"timestamp order error, start = {begin_timestamp}, end = {timestamp}"
            track = Task(tag=tag, task_type=task_type, start_time=begin_timestamp, duration=timestamp - begin_timestamp)
            block_idx_to_tracks[block_idx].append(track)
    return block_idx_to_tracks

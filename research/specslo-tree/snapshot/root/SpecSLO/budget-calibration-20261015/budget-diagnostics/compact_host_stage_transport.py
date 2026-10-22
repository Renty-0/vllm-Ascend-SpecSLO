"""Uninstalled four-rank Gloo-only HOST stage transport prototype."""
import threading
from compact_host_stage_codec import WORDS, encode, decode_success


class CompactHostStageTransport:
    def __init__(self, torch, distributed, group):
        if distributed.get_backend(group) != 'gloo' or distributed.get_world_size(group) != 4:
            raise ValueError('Prototype requires the existing four-rank CPU Gloo group')
        self.torch, self.distributed, self.group = torch, distributed, group
        self.local = threading.local()
        self.compact_calls = self.object_fallbacks = 0

    def buffers(self):
        if not hasattr(self.local, 'buffers'):
            with self.torch.inference_mode(False):
                send = self.torch.empty(WORDS, dtype=self.torch.int64, device='cpu')
                receive = self.torch.empty(4*WORDS, dtype=self.torch.int64, device='cpu')
            self.local.buffers = send, receive
        return self.local.buffers

    def collect(self, original_row):
        rank, sequence = original_row['rank'], original_row['sequence']
        if (type(rank) is int and type(sequence) is int and 0 <= rank < 4
                and 0 <= sequence < 2**63):
            payload = encode(rank, sequence, original_row['stage'], original_row['error'])
        else:
            # Every rank takes the same first collective, even for a locally
            # unsupported value. It then collectively uses original objects.
            payload = (0, 0, 1, -1, *((0,)*(WORDS-4)))
        send, receive = self.buffers()
        source = self.torch.tensor(payload, dtype=self.torch.int64, device='cpu')
        send.copy_(source)
        self.distributed.all_gather_into_tensor(receive, send, group=self.group)
        rows = decode_success(receive.reshape(4, WORDS).tolist())
        if rows is not None:
            self.compact_calls += 1
            return rows
        self.object_fallbacks += 1
        rows = [None]*4
        self.distributed.all_gather_object(rows, original_row, group=self.group)
        return rows

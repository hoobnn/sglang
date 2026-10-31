"""Scheduler-thread agreement of HiCache storage operations.

The workers record local verdicts and progress on the operation; one MIN per
scheduler tick agrees them across ranks and publishes them in submission
order. A slow or dead backend on one rank must never strand the peers in a
worker-thread collective, every rank must publish the same operation with the
same agreed values in the same tick, and a deadline that expired on one rank
cuts the operation on every rank.
"""

import tempfile
import threading
import time
import unittest
from array import array
from collections import defaultdict, deque
from contextlib import nullcontext
from datetime import timedelta
from queue import Queue
from types import SimpleNamespace
from unittest import mock
from unittest.mock import Mock

import torch
import torch.distributed as dist

from sglang.srt.managers.cache_controller import StorageCallLanes, _drain_queue
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
    PrefetchOperation,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.storage_prefetch import StoragePrefetchRetries
from sglang.srt.mem_cache.unified_cache.storage_agreement import (
    QUERY_PENDING,
    OperationRowSource,
    StorageAgreementEngine,
    attention_group_min,
)
from sglang.srt.mem_cache.unified_radix_cache import (
    _RESERVE_DONE,
    _RESERVE_SHORT,
    UnifiedRadixCache,
    _OngoingPrefetch,
)
from sglang.srt.mem_cache.utils import get_storage_hash_str
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

PAGE = 2
TOKENS = list(range(8))  # four pages
CHAIN = ["h0", "h1", "h2", "h3"]
# Rank 1's backend holds its first call this long; the peer's scheduler must
# keep ticking meanwhile, each tick far quicker than the hold.
HOLD_S = 1.0
# A tick may wait for a slow peer's count exchange, never for the hold itself.
TICK_BUDGET_S = HOLD_S * 0.9


def _operation(rid: str, pool_transfers=None) -> PrefetchOperation:
    operation = PrefetchOperation(
        CacheRequestHandle(rid, 0), list(TOKENS), pool_transfers=pool_transfers
    )
    operation.all_hash_values = list(CHAIN)
    return operation


def _trailing_operation(rid: str) -> PrefetchOperation:
    return _operation(
        rid,
        pool_transfers=[
            PoolTransfer(
                name=PoolName.SWA,
                keys=list(CHAIN),
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            )
        ],
    )


def _sidecar_operation(rid: str) -> PrefetchOperation:
    return _operation(
        rid,
        pool_transfers=[PoolTransfer(name=PoolName.SWA, keys=list(CHAIN))],
    )


class _Rank:
    """One rank's engine plus the operations it would put in its rows; peers
    never tick, they only contribute rows to the rank under test."""

    def __init__(self, alive=True, stopped=False, source=None):
        self.alive = alive
        self.stopped = stopped
        self.peers: list[_Rank] = []
        self.engine = StorageAgreementEngine(
            PAGE,
            self._agree,
            lambda: self.alive,
            source=source,
            backend_stopped=lambda: self.stopped,
        )
        self.published: list = []
        self.acks: list = []
        self.cuts: list = []

    def _rows(self):
        stopped = not self.alive and self.stopped
        return [
            self.engine._query_row(op, self.alive, stopped)
            for op in self.engine.queries
        ] + [
            self.engine._transfer_row(op, self.alive, stopped)
            for op in self.engine.transfers
        ]

    def _agree(self, rows: list[list[int]]) -> list[list[int]]:
        agreed = [list(row) for row in rows]
        for peer in self.peers:
            peer_rows = peer._rows()
            assert [len(r) for r in peer_rows] == [len(r) for r in rows]
            agreed = [
                [min(a, b) for a, b in zip(mine, theirs)]
                for mine, theirs in zip(agreed, peer_rows)
            ]
        return agreed

    def tick(self) -> tuple[int, int]:
        return self.engine.tick(
            self.published.append,
            lambda op, completed, hits: self.acks.append((op, completed, hits)),
            self.cuts.append,
        )


def _ranks(
    n=2, alive=(True, True), stopped=(False, False), sources=(None, None)
) -> list[_Rank]:
    ranks = [
        _Rank(alive=alive[i], stopped=stopped[i], source=sources[i]) for i in range(n)
    ]
    for rank in ranks:
        rank.peers = [other for other in ranks if other is not rank]
    return ranks


class TestStorageAgreementQueries(unittest.TestCase):
    def test_publication_is_fifo_behind_a_pending_head(self):
        """An idle engine runs no collective. A rank still querying the head
        blocks later queries on every rank, even when those are done
        everywhere; the order is submission order."""
        local, peer = _ranks()
        local.engine._agree = Mock(wraps=local._agree)
        self.assertEqual(local.tick(), (0, 0))
        local.engine._agree.assert_not_called()

        ops = {
            rank: [_operation("a"), _operation("b"), _operation("c")]
            for rank in (local, peer)
        }
        for rank, (a, b, c) in ops.items():
            for operation in (a, b, c):
                rank.engine.register_query(operation)
            a.publish_local_query(4, {PoolName.KV: 4}, None)
            c.publish_local_query(4, {PoolName.KV: 4}, None)
        ops[local][1].publish_local_query(
            3, {PoolName.KV: 3}, None
        )  # the peer's b is pending

        self.assertEqual(local.tick(), (1, 0))
        self.assertEqual([op.request_id for op in local.published], ["a"])
        self.assertEqual(local.engine.pending_queries(), 2)

        ops[peer][1].publish_local_query(2, {PoolName.KV: 2}, None)
        peer.engine.queries.popleft()  # the peer published a in the same tick
        self.assertEqual(local.tick(), (2, 0))
        self.assertEqual([op.request_id for op in local.published], ["a", "b", "c"])
        b = ops[local][1]
        self.assertEqual(b.storage_hit_count, 2 * PAGE)
        self.assertEqual(b.hash_value, CHAIN[:2])
        self.assertEqual(local.engine.pending_queries(), 0)

    def test_agreed_hit_is_the_shortest_prefix_every_rank_can_restore(self):
        """MIN of the hit and of every pool verdict; a pool one rank gave no
        verdict for heals from the folded cut only. For a trailing-state pool
        the agreed hit is the longest prefix every rank can restore (state at
        2 and 4 pages here, at 1-3 there), not MIN of the hits."""
        local, peer = _ranks()
        mine, theirs = _operation("a"), _operation("a")
        local.engine.register_query(mine)
        peer.engine.register_query(theirs)
        mine.publish_local_query(4, {PoolName.KV: 4, PoolName.SWA: 4}, None)
        theirs.publish_local_query(2, {PoolName.KV: 2}, None)  # no SWA verdict there
        self.assertEqual(local.tick(), (1, 0))
        self.assertEqual(mine.storage_hit_count, 2 * PAGE)
        self.assertEqual(mine.hash_value, CHAIN[:2])
        self.assertEqual(mine.query_pool_hit_pages, {PoolName.KV: 2})

        local, peer = _ranks()
        mine, theirs = _trailing_operation("b"), _trailing_operation("b")
        local.engine.register_query(mine)
        peer.engine.register_query(theirs)
        mine.publish_local_query(4, {PoolName.KV: 4, PoolName.SWA: 4}, (2, 4))
        theirs.publish_local_query(3, {PoolName.KV: 3, PoolName.SWA: 3}, (1, 2, 3))
        self.assertEqual(local.tick(), (1, 0))
        self.assertEqual(mine.storage_hit_count, 2 * PAGE)
        self.assertEqual(mine.hash_value, CHAIN[:2])
        self.assertEqual(mine.query_pool_hit_pages, {PoolName.KV: 3, PoolName.SWA: 3})

    def test_misses_win_and_wait_for_their_single_writer(self):
        """A query that raised on one rank votes 0 with no pool verdict, so no
        belief learns absence from a backend failure. A cancelled query is not
        published until its worker recorded a local outcome (the worker may
        still be inside the backend call and writing the operation), then the
        rank that skipped it votes the winning miss. Storage stopped on purpose
        on one rank (a detach, finished or not) answers its pending queries
        with misses instead of killing every scheduler."""
        local, peer = _ranks()
        mine, theirs = _operation("a"), _operation("a")
        local.engine.register_query(mine)
        peer.engine.register_query(theirs)
        mine.publish_local_query(4, {PoolName.KV: 4}, None)
        theirs.publish_local_query(0, {}, None)
        self.assertEqual(local.tick(), (1, 0))
        self.assertEqual(mine.storage_hit_count, 0)
        self.assertEqual(mine.hash_value, [])
        self.assertEqual(mine.query_pool_hit_pages, {})

        local, peer = _ranks()
        mine, theirs = _operation("b"), _operation("b")
        local.engine.register_query(mine)
        peer.engine.register_query(theirs)
        mine.mark_terminate()
        self.assertEqual(mine.query_status(), (QUERY_PENDING, None))
        self.assertEqual(local.tick(), (0, 0))
        mine.publish_local_query(0, {}, None)  # the worker skipped it
        theirs.publish_local_query(4, {PoolName.KV: 4}, None)  # answered late
        self.assertEqual(local.tick(), (1, 0))
        self.assertEqual(mine.storage_hit_count, 0)
        self.assertEqual(mine.hash_value, [])

        # Peers never tick in this fixture, so each rank's view gets its own pair.
        for ticking in (0, 1):
            ranks = _ranks(alive=(False, True), stopped=(True, False))
            for rank in ranks:
                rank.engine.register_query(_operation("c"))
            list(ranks[1].engine.queries)[0].publish_local_query(
                4, {PoolName.KV: 4}, None
            )
            self.assertEqual(ranks[ticking].tick(), (1, 0))
            (operation,) = ranks[ticking].published
            self.assertEqual(operation.storage_hit_count, 0)
            self.assertEqual(operation.query_pool_hit_pages, {})

    def test_expired_query_is_cut_on_every_rank(self):
        """The deadline expired on the peer only; MAX of the bit makes this rank
        cut too, in the same tick, while the query itself stays pending."""
        local, peer = _ranks(
            sources=(None, OperationRowSource(PAGE, query_expired=lambda op: True))
        )
        mine, theirs = _operation("a"), _operation("a")
        local.engine.register_query(mine)
        peer.engine.register_query(theirs)

        self.assertEqual(local.tick(), (0, 0))
        self.assertEqual(local.cuts, [mine])
        self.assertEqual(local.engine.pending_queries(), 1)


class TestStorageAgreementTransfers(unittest.TestCase):
    def test_ack_is_agreed_once_every_rank_finished_with_its_rows(self):
        """Both FIFOs ride one collective, so a pending query holds back no
        finished transfer. Progress on one rank means nothing until the other
        rank's worker is done with its host rows; the agreed prefix is the MIN
        of final progress and sidecar hits MIN-reduce per pool."""
        local, peer = _ranks()
        for rank in (local, peer):
            rank.engine.register_query(_operation("q"))  # pending everywhere
        mine, theirs = _sidecar_operation("t"), _sidecar_operation("t")
        local.engine.register_transfer(mine)
        peer.engine.register_transfer(theirs)
        mine.publish_local_transfer_progress(8)
        mine.publish_local_sidecar_hits({PoolName.SWA.value: 4})
        mine.mark_transfer_finished()
        theirs.publish_local_transfer_progress(4)

        self.assertEqual(local.tick(), (0, 0))
        self.assertEqual(local.engine.pending_transfers(), 1)

        theirs.publish_local_transfer_progress(6)
        theirs.publish_local_sidecar_hits({PoolName.SWA.value: 2})
        theirs.mark_transfer_finished()
        self.assertEqual(local.tick(), (0, 1))
        ((operation, completed, pool_hits),) = local.acks
        self.assertIs(operation, mine)
        self.assertEqual(completed, 6)
        self.assertEqual(pool_hits[PoolName.SWA.value], 2)
        self.assertEqual(pool_hits[PoolName.KV.value], 0)
        self.assertEqual(local.engine.pending_transfers(), 0)
        self.assertEqual(local.engine.pending_queries(), 1)

    def test_cache_mode_cut_agrees_the_prefix_held_on_every_rank(self):
        """A cache-mode cut on the admission path agrees the transfer state
        held right now in one MIN over the attention group: completed tokens,
        the bit saying the sidecar read ran, and every pool's hits are each
        the MIN across ranks, and the agreed hits land on the operation per
        pool in PoolName order. A rank whose sidecar read has not run yet
        leaves the sidecar hits unclaimed everywhere. An operation without
        sidecar pools agrees its prefix and keeps its pool state untouched."""
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        pools = [pool.value for pool in PoolName]
        peer_rows: list[list[int]] = []
        reduces: list = []

        def reduce_min(tensor, op):
            reduces.append(op)
            peer = torch.tensor(peer_rows[len(reduces) - 1], dtype=tensor.dtype)
            torch.minimum(tensor, peer, out=tensor)

        cache._all_reduce_attn_groups = reduce_min

        # This rank read the whole span; the peer read less of it and fewer
        # SWA pages but more KV pages, so MIN picks a different rank per column.
        operation = _sidecar_operation("cut")
        operation.publish_local_transfer_progress(8)
        operation.publish_local_sidecar_hits(
            {PoolName.KV.value: 3, PoolName.SWA.value: 4}
        )
        peer_hits = {PoolName.KV.value: 4, PoolName.SWA.value: 2}
        peer_rows.append([6, 1, *(peer_hits.get(pool, 0) for pool in pools)])
        cache._agree_cache_mode_progress(operation)
        self.assertEqual(reduces, [dist.ReduceOp.MIN])
        self.assertEqual(operation.completed_tokens, 6)
        self.assertTrue(operation.pool_transfers_done)
        expected = dict.fromkeys(pools, 0)
        expected.update({PoolName.KV.value: 3, PoolName.SWA.value: 2})
        self.assertEqual(operation.pool_storage_result.extra_pool_hit_pages, expected)

        # Here the sidecar read has not run (no hits recorded): the bit is 0
        # on this rank, so the peer's finished sidecar read claims nothing.
        later = _sidecar_operation("later")
        later.publish_local_transfer_progress(4)
        peer_rows.append([6, 1, *(peer_hits.get(pool, 0) for pool in pools)])
        cache._agree_cache_mode_progress(later)
        self.assertEqual(len(reduces), 2)
        self.assertEqual(later.completed_tokens, 4)
        self.assertFalse(later.pool_transfers_done)
        self.assertEqual(
            later.pool_storage_result.extra_pool_hit_pages, dict.fromkeys(pools, 0)
        )

        # An operation with no sidecar pools (KV only) still agrees its prefix;
        # its pool state was done at construction and the agreement leaves it.
        plain = _operation("plain")
        plain.publish_local_transfer_progress(8)
        peer_rows.append([2, 0, *([0] * len(pools))])
        cache._agree_cache_mode_progress(plain)
        self.assertEqual(len(reduces), 3)
        self.assertEqual(plain.completed_tokens, 2)
        self.assertTrue(plain.pool_transfers_done)
        self.assertEqual(plain.pool_storage_result.extra_pool_hit_pages, {})

    def test_finished_transfer_publishes_past_a_pending_one(self):
        """A transfer whose rows a slow (orphaned) call still holds does not
        hold back later transfers: the agreed state names the operation."""
        local, peer = _ranks()
        for rank in (local, peer):
            slow, fast = _operation("slow"), _operation("fast")
            rank.engine.register_transfer(slow)
            rank.engine.register_transfer(fast)
            fast.publish_local_transfer_progress(8)
            fast.mark_transfer_finished()
        self.assertEqual(local.tick(), (0, 1))
        self.assertEqual(local.acks[0][0].request_id, "fast")
        self.assertEqual([op.request_id for op in local.engine.transfers], ["slow"])


class TestAgreedHitReservations(unittest.TestCase):
    """The hit drain reserves locally, agrees the outcomes in one MIN, and only
    then dispatches: a peer's shortfall parks the operation here too, and a
    parked hit keeps its turn."""

    def _cache(self, peer_outcomes):
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.host_memory_mode = "buffer_only"
        cache.prefetch_threshold = PAGE
        cache.enable_storage_metrics = False
        cache.storage_metrics_collector = None
        cache.storage_prefetch_retries = StoragePrefetchRetries()
        cache._storage_prefetch_wait = {}
        cache._prefetch_outcome_stats = defaultdict(int)
        cache._storage_prefetch_hit_remaining_by_reqid = {}
        cache._record_storage_prefetch_hit = Mock()
        cache.revoke_pending_prefetch = Mock()
        cache._invalidate_absent_from_hit_query = Mock()
        cache._account_prefetch_outcome = Mock()
        cache._trim_buffer_prefetch_full_head = (
            lambda request, info, operation, full, hit: (info, hit, hit)
        )
        cache.storage_engine = object()  # agreement on
        dispatched, rolled_back = [], []
        pipeline = SimpleNamespace(
            pending_hit_allocs=deque(),
            try_lock_anchor=lambda request, tokens: ("locked", 0),
            staged_span_covered=lambda request, tokens: False,
            reserve_prefetch=lambda operation, info, hit, aux: "reserved",
            dispatch_reserved_prefetch=dispatched.append,
            rollback_reserved_prefetch=rolled_back.append,
            release_anchor_lock=lambda request: None,
        )
        cache.buffer_pipeline = pipeline
        cache.cache_controller = SimpleNamespace(
            prefetch_hit_queue=Queue(),
            ack_prefetch_queue=Queue(),
            ack_backup_queue=Queue(),
            host_mem_release_queue=Queue(),
            extra_host_mem_release_queues={},
            prefetch_rate_limited=lambda: False,
        )
        cache.ongoing_prefetch = {}

        def reduce_min(tensor, op):
            # One row per outcome agreed in this drain; tests edit the list.
            peer = torch.tensor(peer_outcomes, dtype=torch.int64)
            torch.minimum(tensor, peer, out=tensor)

        cache._all_reduce_attn_groups = reduce_min
        return cache, pipeline, dispatched, rolled_back

    def _hit(self, cache, rid):
        operation = _operation(rid)
        operation.storage_hit_count = len(TOKENS)
        operation.hash_value = list(CHAIN)
        cache.ongoing_prefetch[operation.handle] = _OngoingPrefetch(
            0, RadixKey(array("q", TOKENS)), None, operation, None, {}
        )
        cache.cache_controller.prefetch_hit_queue.put(operation)
        return operation

    def _drain(self, cache, hits):
        cache._drain_storage_control_queues_impl(
            n_storage_hit=hits,
            n_ack_prefetch=0,
            n_backup=0,
            n_release=0,
            extra_release_counts=None,
            log_metrics=False,
        )

    def test_shortfall_parks_in_order_and_parked_hits_dispatch_once_agreed(self):
        reserved = [1, len(TOKENS)]
        # The peer is short on a and reserved b: both park here, in order.
        peer = [[0, 0], list(reserved)]
        cache, pipeline, dispatched, rolled_back = self._cache(peer)
        a, b = self._hit(cache, "a"), self._hit(cache, "b")
        self._drain(cache, 2)
        self.assertEqual(rolled_back, [a, b])
        self.assertEqual(dispatched, [])
        self.assertEqual(list(pipeline.pending_hit_allocs), [a, b])
        self.assertEqual(cache._prefetch_outcome_stats["declined_rate_limited"], 2)

        # Room on the peer: the parked hits are retried first and dispatch in order.
        peer[:] = [list(reserved), list(reserved)]
        self._drain(cache, 0)
        self.assertEqual(dispatched, [a, b])
        self.assertFalse(pipeline.pending_hit_allocs)

        # A local shortfall parks without a rollback and keeps its turn ahead
        # of the next hit; once reserved everywhere both dispatch in order.
        pipeline.reserve_prefetch = lambda operation, info, hit, aux: "short"
        peer[:] = [list(reserved)]
        c = self._hit(cache, "c")
        self._drain(cache, 1)
        self.assertEqual((rolled_back, dispatched), ([a, b], [a, b]))
        self.assertEqual(list(pipeline.pending_hit_allocs), [c])
        pipeline.reserve_prefetch = lambda operation, info, hit, aux: "reserved"
        peer[:] = [list(reserved), list(reserved)]
        d = self._hit(cache, "d")
        self._drain(cache, 1)
        self.assertEqual(dispatched, [a, b, c, d])
        self.assertFalse(pipeline.pending_hit_allocs)
        self.assertEqual(cache._prefetch_outcome_stats["declined_rate_limited"], 3)

    def _cache_mode(self, peer_outcomes):
        """host_memory_mode "cache": the hit is staged in the resident host
        pool, with no pipeline and no parking. A peer's shorter reservation
        trims this rank's allocation; an agreed shortfall revokes the prefetch.
        The whole hit fits locally, so the peer decides how much launches."""
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.host_memory_mode = "cache"
        cache.buffer_pipeline = None
        cache.tree_core = SimpleNamespace(page_size=PAGE)
        cache.prefetch_threshold = PAGE
        cache.enable_storage_metrics = False
        cache.storage_metrics_collector = None
        cache.evict_host = Mock()
        cache.storage_prefetch_retries = Mock(spec=StoragePrefetchRetries)
        cache._storage_prefetch_wait = {}
        cache._prefetch_outcome_stats = defaultdict(int)
        cache._storage_prefetch_hit_remaining_by_reqid = {}
        cache._record_storage_prefetch_hit = Mock()
        cache._resolve_storage_prefetch_tokens = Mock()
        cache._finish_storage_prefetch = Mock()
        cache._log_storage_prefetch_deferred = Mock()
        cache.revoke_pending_prefetch = Mock()
        cache._invalidate_absent_from_hit_query = Mock()
        cache._account_prefetch_outcome = Mock()
        cache.storage_engine = object()  # agreement on
        dispatched, freed_kv, rolled_back = [], [], []
        cache.cache_controller = SimpleNamespace(
            prefetch_hit_queue=Queue(),
            ack_prefetch_queue=Queue(),
            ack_backup_queue=Queue(),
            host_mem_release_queue=Queue(),
            extra_host_mem_release_queues={},
            allocate_storage_hit=lambda operation, hit_tokens, **kwargs: (
                torch.arange(hit_tokens),
                hit_tokens,
            ),
            mem_pool_host=SimpleNamespace(free=freed_kv.append),
            free_prefetch_host_buffers=lambda operation, host_indices: (
                rolled_back.append((operation, host_indices))
            ),
            dispatch_prefetch_transfer=dispatched.append,
        )
        cache.ongoing_prefetch = {}

        def reduce_min(tensor, op):
            peer = torch.tensor(peer_outcomes, dtype=torch.int64)
            torch.minimum(tensor, peer, out=tensor)

        cache._all_reduce_attn_groups = reduce_min
        return cache, dispatched, freed_kv, rolled_back

    def test_cache_mode_trims_the_allocation_to_the_agreed_prefix(self):
        # The peer could stage only the first two pages of the four-page hit.
        agreed = 2 * PAGE
        cache, dispatched, freed_kv, rolled_back = self._cache_mode(
            [[_RESERVE_DONE, agreed]]
        )
        a = self._hit(cache, "a")
        self._drain(cache, 1)

        # The excess KV rows go back to the pool before the transfer launches;
        # the reservation itself is not rolled back.
        self.assertEqual(len(freed_kv), 1)
        self.assertTrue(torch.equal(freed_kv[0], torch.arange(agreed, len(TOKENS))))
        self.assertEqual(rolled_back, [])
        # The dispatched operation and its tracking entry describe the agreed
        # prefix only.
        self.assertEqual(dispatched, [a])
        self.assertEqual(a.storage_hit_count, agreed)
        self.assertEqual(a.hash_value, CHAIN[: agreed // PAGE])
        self.assertTrue(torch.equal(a.host_indices, torch.arange(agreed)))
        self.assertTrue(
            torch.equal(
                cache.ongoing_prefetch[a.handle].host_indices, torch.arange(agreed)
            )
        )
        # The trimmed tokens are unfulfilled for want of host capacity; the
        # prefetch stays alive, so nothing is revoked or retried.
        cache._resolve_storage_prefetch_tokens.assert_called_once_with(
            a.handle, len(TOKENS) - agreed, reason="host_capacity"
        )
        cache._finish_storage_prefetch.assert_not_called()
        cache.revoke_pending_prefetch.assert_not_called()
        cache.storage_prefetch_retries.poll_miss.assert_not_called()

    def test_cache_mode_agreed_shortfall_frees_everything_and_revokes(self):
        cache, dispatched, freed_kv, rolled_back = self._cache_mode(
            [[_RESERVE_SHORT, 0]]
        )
        a = self._hit(cache, "a")
        self._drain(cache, 1)

        # This rank reserved the whole hit; the peer's shortfall gives all of
        # it back through the rollback path and launches nothing.
        self.assertEqual(dispatched, [])
        self.assertEqual(freed_kv, [])
        ((operation, host_indices),) = rolled_back
        self.assertIs(operation, a)
        self.assertTrue(torch.equal(host_indices, torch.arange(len(TOKENS))))
        self.assertIsNone(a.host_indices)
        self.assertEqual(a.storage_hit_count, len(TOKENS))
        self.assertEqual(a.hash_value, CHAIN)
        # Cache mode does not park: the whole hit is unfulfilled for host
        # capacity, the prefetch is revoked and the request polls the miss at
        # the hit's end.
        cache._finish_storage_prefetch.assert_called_once_with(
            a.handle, fulfilled_tokens=0, reason="host_capacity"
        )
        cache._log_storage_prefetch_deferred.assert_called_once_with(
            len(TOKENS), "host_capacity"
        )
        cache.revoke_pending_prefetch.assert_called_once_with(a.handle)
        cache.storage_prefetch_retries.poll_miss.assert_called_once_with(
            "a", a.storage_start + len(TOKENS)
        )
        cache._resolve_storage_prefetch_tokens.assert_not_called()

    def test_settled_outcome_must_match_the_peers(self):
        # This rank declines (span device-covered) while the peer reserved:
        # the tree state is no longer lockstep, which is fatal, not a park.
        cache, pipeline, *_ = self._cache([[1, len(TOKENS)]])
        pipeline.staged_span_covered = lambda request, tokens: True
        self._hit(cache, "a")
        with self.assertRaisesRegex(RuntimeError, "diverged"):
            self._drain(cache, 1)


class TestStorageCallLanes(unittest.TestCase):
    def test_bound_must_be_positive_and_finite(self):
        for bad in (0, -1.0, float("inf"), float("nan"), None):
            with self.assertRaises(ValueError):
                StorageCallLanes(object(), bad)

    def test_backend_timeout_error_is_a_failed_call_not_an_orphan(self):
        class _Raises:
            def batch_exists(self, hashes, extra_info=None):
                raise TimeoutError("client timeout")

        lanes = StorageCallLanes(_Raises(), 5.0)
        with self.assertRaisesRegex(TimeoutError, "client timeout"):
            lanes.batch_exists(CHAIN)
        self.assertEqual(lanes.orphaned_calls, 0)
        self.assertEqual(lanes.wait_idle(0), 0)


# ---------------------------------------------------------------- two ranks


class _HoldingBackend:
    """Every call reports all pages present; the first query, the first read
    and the first write each block on an event when asked to."""

    def __init__(self, hold_query: bool, hold_get: bool):
        self.query_hold = threading.Event()
        self.get_hold = threading.Event()
        self.set_hold = threading.Event()
        self.hold_query, self.hold_get, self.hold_set = hold_query, hold_get, False
        self.queries = 0
        self.gets = 0
        self.sets = 0
        self.closed = False

    def batch_exists(self, hashes, extra_info=None):
        self.queries += 1
        if self.hold_query and self.queries == 1:
            self.query_hold.wait()
        return len(hashes)

    def batch_get_v1(self, hashes, host_indices, extra_info=None):
        self.gets += 1
        if self.hold_get and self.gets == 1:
            self.get_hold.wait()
        return [True] * len(hashes)

    def batch_set_v1(self, hashes, host_indices, extra_info=None):
        self.sets += 1
        if self.hold_set and self.sets == 1:
            self.set_hold.wait()
        return [True] * len(hashes)

    def close(self):
        self.closed = True


def _controller(backend) -> HybridCacheController:
    """The real query and transfer workers over a scripted backend; agreement
    is the scheduler's, so the workers only record local outcomes."""
    cc = HybridCacheController.__new__(HybridCacheController)
    cc.page_size = PAGE
    cc.enable_storage = True
    cc.storage_backend = backend
    cc.storage_backend_type = "scripted"
    cc.storage_config = SimpleNamespace(is_storage_writer=True)
    cc.storage_calls = None
    cc.scheduler_agrees_storage = True
    cc.on_prefetch_dispatched = None
    cc.storage_stop_event = threading.Event()
    cc._inflight_lock = threading.Lock()
    cc._worker_transfers, cc._worker_backups = set(), {}
    cc.prefetch_queue = Queue()
    cc.backup_queue = Queue()
    cc.prefetch_buffer = Queue()
    cc.prefetch_sync_queue = Queue()
    cc.prefetch_hit_queue = Queue()
    cc.ack_prefetch_queue = Queue()
    cc.ack_backup_queue = Queue()
    cc.host_mem_release_queue = Queue()
    cc.extra_host_mem_release_queues = {}
    cc.write_queue, cc.load_queue = [], []
    cc.ack_write_queue, cc.ack_load_queue = [], []
    cc.prefetch_hits_sync_groups = cc.prefetch_completion_sync_groups = []
    cc.prefetch_tokens_occupied = 0
    # The hybrid reset/detach path also parks the PP ticket machinery.
    cc.pp_prefetch_command_thread = cc.pp_prefetch_command_group = None
    cc.pp_prefetch_state_lock = threading.Lock()
    cc.pp_prefetch_states, cc.pp_prefetch_decisions = {}, {}
    cc.start_writing = Mock()
    cc.page_get_func = cc._page_get_zero_copy
    cc.page_set_func = cc._page_set_zero_copy
    cc.mem_pool_host = SimpleNamespace(
        layout_lease=lambda fence=None: nullcontext(), page_size=PAGE
    )
    cc.prefetch_sync_thread = None
    cc.prefetch_thread = cc.prefetch_io_aux_thread = cc.backup_thread = None
    cc._spawn_storage_threads()
    return cc


def _scheduler_cache(cc) -> UnifiedRadixCache:
    """The scheduler-thread side: the real check_hicache_events round with the
    real count exchange and agreement tick; the drain itself is observed."""
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.cache_controller = cc
    cache.tree_core = SimpleNamespace(
        enable_storage=True, page_size=PAGE, write_back_duplicate_reclaim_digest=0
    )
    cache.host_memory_mode = "cache"
    cache.buffer_pipeline = None
    cache.enable_storage_metrics = False
    cache.storage_metrics_collector = None
    cache.linker = None
    cache.pp_rank, cache.pp_size, cache.pp_group = 0, 1, None
    cache.tp_group = dist.group.WORLD
    cache.tp_world_size = dist.get_world_size()
    cache.attn_cp_group = cache.attn_tp_group = None
    cache.work_list = []
    cache.writing_check = Mock()
    cache.loading_check = Mock()
    cache._drain_storage_control_queues_impl = Mock()
    cache.prefetch_stop_policy = "wait_complete"
    cache.storage_prefetch_retries = StoragePrefetchRetries()
    cache._storage_prefetch_wait = {}
    cache._prefetch_outcome_stats = defaultdict(int)
    cache.ongoing_prefetch = {}
    cache.storage_engine = StorageAgreementEngine(
        PAGE,
        attention_group_min(cache._reduce_min_attn_groups),
        cache._storage_workers_alive,
        OperationRowSource(
            PAGE,
            query_expired=cache._query_expired,
            transfer_expired=cache._transfer_expired,
        ),
        backend_stopped=cache._storage_workers_stopped,
    )
    cc.on_prefetch_dispatched = cache.storage_engine.register_transfer
    return cache


def _submit(cache, rid: str, enqueue: bool = True) -> PrefetchOperation:
    """What prefetch_from_storage does for the query: track, register, enqueue."""
    operation = PrefetchOperation(CacheRequestHandle(rid, 0), list(TOKENS))
    cache.ongoing_prefetch[operation.handle] = _OngoingPrefetch(
        0, RadixKey(array("q", TOKENS)), None, operation, None, {}
    )
    cache.storage_engine.register_query(operation)
    if enqueue:
        cache.cache_controller.prefetch_queue.put(operation)
    return operation


def _dispatch(cache, operation) -> None:
    """What the hit drain does after the bounce alloc."""
    operation.host_indices = torch.arange(len(TOKENS))
    cache.ongoing_prefetch[operation.handle] = cache.ongoing_prefetch[
        operation.handle
    ]._replace(host_indices=operation.host_indices)
    cache.cache_controller.dispatch_prefetch_transfer(operation)


def _same_on_every_rank(snapshot):
    snapshots = [None] * dist.get_world_size()
    dist.all_gather_object(snapshots, snapshot)
    assert all(other == snapshots[0] for other in snapshots), snapshots
    return snapshots[0]


def _gathered(snapshot) -> list:
    snapshots = [None] * dist.get_world_size()
    dist.all_gather_object(snapshots, snapshot)
    return snapshots


def _drain_kw(cache) -> dict:
    return cache._drain_storage_control_queues_impl.call_args.kwargs


def _drain_counts(cache) -> tuple[int, int]:
    kwargs = _drain_kw(cache)
    return kwargs["n_storage_hit"], kwargs["n_ack_prefetch"]


def _tick_until(cache, done, budget_s=20.0):
    """Tick until ``done(hits, acks)`` holds; returns (tick index, max tick time)."""
    start = time.monotonic()
    worst = 0.0
    for tick in range(int(budget_s / 0.02)):
        t0 = time.monotonic()
        cache.check_hicache_events()
        worst = max(worst, time.monotonic() - t0)
        if done(*_drain_counts(cache)):
            return tick, worst, time.monotonic() - start
        time.sleep(0.02)
    raise AssertionError("the agreed outcome was never published")


def _slow_backend_body(rank: int) -> None:
    """Rank 1's backend blocks its first query, then its first read. Rank 0's
    scheduler keeps ticking (nothing waits on a worker), nothing is published
    until each hold lifts, and then both ranks publish the same agreed outcome
    in the same tick."""
    backend = _HoldingBackend(hold_query=(rank == 1), hold_get=(rank == 1))
    cc = _controller(backend)
    cache = _scheduler_cache(cc)
    get_storage_hash_str(TOKENS, None, page_size=PAGE)  # JIT warm-up off the clock

    # Query phase.
    operation = _submit(cache, "slow")
    if rank == 1:
        threading.Timer(HOLD_S, backend.query_hold.set).start()
    tick, worst, elapsed = _tick_until(cache, lambda hits, acks: hits > 0)
    assert elapsed >= HOLD_S * 0.5, elapsed
    assert worst < TICK_BUDGET_S, worst
    assert tick >= 3, tick
    assert cc.prefetch_hit_queue.get_nowait() is operation
    _same_on_every_rank((tick, operation.storage_hit_count, operation.hash_value))
    assert operation.storage_hit_count == len(TOKENS)

    # Transfer phase: the read is held on rank 1 the same way.
    _dispatch(cache, operation)
    if rank == 1:
        threading.Timer(HOLD_S, backend.get_hold.set).start()
    tick, worst, elapsed = _tick_until(cache, lambda hits, acks: acks > 0)
    assert elapsed >= HOLD_S * 0.5, elapsed
    assert worst < TICK_BUDGET_S, worst
    ack = cc.ack_prefetch_queue.get_nowait()
    assert ack.operation is operation and ack.completed_req
    assert ack.pool_hits is None
    _same_on_every_rank((tick, ack.completed_tokens))
    assert ack.completed_tokens == len(TOKENS)
    assert operation.local_transfer_progress().finished
    assert cache.storage_engine.pending_transfers() == 0
    assert cc.storage_workers_alive()

    # Cut phases: the clock expired (timeout) or the request reached its
    # admission poll (best_effort) on rank 0 only; both ranks cut the transfer
    # in the same tick and still agree the outcome afterwards.
    def _expire(operation):
        cache._storage_prefetch_wait[operation.handle] = SimpleNamespace(expired=True)

    def _poll(operation):
        cache.storage_engine.admission_polled.add(operation.handle)

    for policy, cut_here in (("timeout", _expire), ("best_effort", _poll)):
        cache.prefetch_stop_policy = policy
        second = _submit(cache, f"cut-{policy}")
        _tick_until(cache, lambda hits, acks: hits > 0)
        cc.prefetch_hit_queue.get_nowait()
        backend.hold_get, backend.gets = True, 0
        backend.get_hold.clear()
        _dispatch(cache, second)
        if rank == 0:
            cut_here(second)
        cut_tick, _, _ = _tick_until(
            cache, lambda hits, acks: second.is_terminated() or acks > 0
        )
        assert second.is_terminated()
        assert second.deadline_expired == (policy == "timeout")
        _same_on_every_rank(cut_tick)
        backend.get_hold.set()
        _tick_until(cache, lambda hits, acks: acks > 0)
        ack = cc.ack_prefetch_queue.get_nowait()
        _same_on_every_rank(ack.completed_tokens)
    cache.prefetch_stop_policy = "wait_complete"

    def _bound(timeout_s):
        cc.storage_backend = cc.storage_calls = StorageCallLanes(backend, timeout_s)

    def _unbound():
        cc.storage_backend, cc.storage_calls = backend, None

    def _wait_until(predicate, budget_s=5.0):
        deadline = time.monotonic() + budget_s
        while not predicate() and time.monotonic() < deadline:
            time.sleep(0.005)
        assert predicate()

    # Orphan phase: calls are bounded on both ranks; rank 1 holds the read of
    # the first operation past the bound, so its worker orphans the call and
    # serves the next operation. That one publishes first on every rank; the
    # held one stays unfinished (its rows quarantined) until the orphan
    # returns, then publishes with the agreed prefix (0: it failed on rank 1).
    _bound(0.3)
    held = _submit(cache, "held")
    behind = _submit(cache, "behind")
    _tick_until(cache, lambda hits, acks: cc.prefetch_hit_queue.qsize() == 2)
    cc.prefetch_hit_queue.get_nowait()
    cc.prefetch_hit_queue.get_nowait()
    backend.hold_get, backend.gets = rank == 1, 0
    backend.get_hold.clear()
    _dispatch(cache, held)
    _dispatch(cache, behind)
    if rank == 1:
        threading.Timer(HOLD_S, backend.get_hold.set).start()
    tick, _, elapsed = _tick_until(cache, lambda hits, acks: acks > 0)
    ack = cc.ack_prefetch_queue.get_nowait()
    assert ack.operation is behind, ack.operation.request_id
    assert elapsed < HOLD_S * 0.9, elapsed  # did not wait behind the held call
    _same_on_every_rank((tick, ack.completed_tokens))
    assert cache.storage_engine.pending_transfers() == 1
    if rank == 1:
        assert not held.local_transfer_progress().finished
        assert not cc.storage_calls.fence(held).query()
    _tick_until(cache, lambda hits, acks: acks > 0)
    ack = cc.ack_prefetch_queue.get_nowait()
    assert ack.operation is held
    _same_on_every_rank(ack.completed_tokens)
    assert ack.completed_tokens == 0
    assert held not in cc._worker_transfers
    assert cc.storage_calls.fence(held).query()
    assert cc.storage_workers_alive()
    _unbound()

    # Backup phase: acks leave in submission order. Rank 1 holds the first
    # write past the bound: no rank acks anything while it runs (the count
    # MIN hides rank 0's ready acks), then both ack [a, b] in the same tick.
    _bound(0.3)
    backend.hold_set, backend.sets = rank == 1, 0
    backend.set_hold.clear()
    rows = torch.arange(len(TOKENS))
    a = cc.write_storage(rows, list(TOKENS), hash_value=list(CHAIN))
    b = cc.write_storage(rows, list(TOKENS), hash_value=list(CHAIN))
    if rank == 1:
        threading.Timer(HOLD_S, backend.set_hold.set).start()
    start = time.monotonic()
    while time.monotonic() - start < HOLD_S * 0.5:
        cache.check_hicache_events()
        assert _drain_kw(cache)["n_backup"] == 0
        time.sleep(0.02)
    _tick_until(cache, lambda hits, acks: _drain_kw(cache)["n_backup"] > 0)
    assert _drain_kw(cache)["n_backup"] == 2
    acked = _drain_queue(cc.ack_backup_queue)
    _same_on_every_rank([op.id for op in acked])
    assert [op.id for op in acked] == [a, b]
    assert acked[0].completed_tokens == (0 if rank == 1 else len(TOKENS))
    assert not cc._worker_backups
    _unbound()

    # Reset phase: a bounded read is held on rank 1 and a second transfer is
    # queued behind it. Reset drops the queued one, cuts the running one and
    # waits for the orphan; the workers are neither joined nor restarted.
    dist.barrier()
    _bound(0.6)  # wide enough that the drain always beats the worker to b
    a = _submit(cache, "reset-a")
    b = _submit(cache, "reset-b")
    _tick_until(cache, lambda hits, acks: cc.prefetch_hit_queue.qsize() == 2)
    cc.prefetch_hit_queue.get_nowait()
    cc.prefetch_hit_queue.get_nowait()
    backend.hold_get, backend.gets = rank == 1, 0
    backend.get_hold.clear()
    _dispatch(cache, a)
    if rank == 1:
        _wait_until(lambda: backend.gets == 1)
    _dispatch(cache, b)
    if rank == 1:
        threading.Timer(HOLD_S, backend.get_hold.set).start()
    t0 = time.monotonic()
    cache.ongoing_prefetch.clear()
    cache.storage_engine.reset()
    cc.reset()
    if rank == 1:
        assert time.monotonic() - t0 >= HOLD_S * 0.5
        assert backend.gets == 1  # b was dropped, never read
    assert b.local_transfer_progress().finished
    assert not cc._worker_transfers and not cc._worker_backups
    assert cc.prefetch_buffer.empty()
    assert cc.storage_workers_alive()
    assert not cc.storage_stop_event.is_set()
    assert cc.prefetch_tokens_occupied == 0
    _unbound()
    dist.barrier()
    fresh = _submit(cache, "fresh")
    _tick_until(cache, lambda hits, acks: hits > 0)
    assert cc.prefetch_hit_queue.get_nowait() is fresh
    _dispatch(cache, fresh)
    _tick_until(cache, lambda hits, acks: acks > 0)
    ack = cc.ack_prefetch_queue.get_nowait()
    assert ack.operation is fresh
    _same_on_every_rank(ack.completed_tokens)

    # Quiesce deadline: rank 1 holds a read with no timer. With the wait cut
    # to bound + 1 s, reset raises there (naming the orphan) and returns on
    # rank 0; once the call returns a second reset succeeds.
    _bound(0.3)
    a = _submit(cache, "outlive")
    _tick_until(cache, lambda hits, acks: hits > 0)
    cc.prefetch_hit_queue.get_nowait()
    backend.hold_get, backend.gets = rank == 1, 0
    backend.get_hold.clear()
    _dispatch(cache, a)
    if rank == 1:
        _wait_until(lambda: backend.gets == 1)
    raised = False
    with mock.patch("sglang.srt.managers.cache_controller.STORAGE_QUIESCE_WAIT_S", 0.2):
        try:
            cache.ongoing_prefetch.clear()
            cache.storage_engine.reset()
            cc.reset()
        except RuntimeError as e:
            raised = "1 behind orphaned calls" in str(e)
    assert _gathered(raised) == [False, True]
    backend.get_hold.set()
    _wait_until(lambda: a.local_transfer_progress().finished)
    cc.reset()
    assert not cc._worker_transfers
    _unbound()
    dist.barrier()

    # Divergent submission: only rank 0 registers a query. The count exchange
    # catches it on every rank before the agreement collective could hang.
    if rank == 0:
        _submit(cache, "only-here")
    raised = False
    try:
        cache.check_hicache_events()
    except RuntimeError as e:
        raised = "diverged" in str(e)
    assert raised, "FIFO divergence was not detected"
    if rank == 0:
        cache.storage_engine.queries.clear()
        cache.ongoing_prefetch.clear()

    # A dead storage worker is a fatal row: every rank raises in the same tick.
    dist.barrier()
    real_query = cc.prefetch_thread  # stopped for real before the workers respawn
    gone = threading.Thread(target=lambda: None)
    gone.start()
    gone.join()
    cc.prefetch_thread = gone
    third = _submit(cache, "third")
    third.publish_local_query(4, {PoolName.KV: 4}, None)
    raised = False
    try:
        cache.check_hicache_events()
    except RuntimeError as e:
        raised = "worker is gone" in str(e)
    assert raised, "dead worker was not reported"
    _same_on_every_rank(raised)

    # Storage stopped on purpose: the same gone worker now means "no backend".
    # A query that was answered keeps its verdict; one that no worker will
    # answer (never enqueued here) is published as an agreed miss, no raise.
    cc.storage_stop_event.set()
    fourth = _submit(cache, "fourth", enqueue=False)
    cache.check_hicache_events()
    assert _drain_counts(cache)[0] == 2
    assert cc.prefetch_hit_queue.get_nowait() is third
    assert cc.prefetch_hit_queue.get_nowait() is fourth
    _same_on_every_rank((third.storage_hit_count, fourth.storage_hit_count))
    assert third.storage_hit_count == len(TOKENS) and fourth.storage_hit_count == 0

    # Reset after a stop: the workers exited on the stop event; reset is not
    # in place then (nothing alive to keep): it joins them, clears the event,
    # resets the ledgers and respawns, as main did.
    cc.prefetch_thread = real_query
    cache.ongoing_prefetch.clear()
    cache.storage_engine.reset()
    cc.reset()
    assert cc.storage_workers_alive() and not cc.storage_stop_event.is_set()
    assert not any(t.is_alive() for t in (real_query, gone))
    dist.barrier()

    # Detach phase, first with an orphaned query (no rows held, so quiesce
    # passes and wait_idle refuses: the backend is closed afterwards), then
    # with an orphaned read on both ranks. Detach raises and changes nothing:
    # the backend stays attached and open, the workers alive, the held row
    # unpublished; once the calls return the retry detaches cleanly.
    _bound(0.3)
    backend.hold_query, backend.queries = True, 0
    backend.query_hold.clear()
    asked = _submit(cache, "detach-query")
    _wait_until(lambda: backend.queries == 1)
    _tick_until(cache, lambda hits, acks: hits > 0)  # the bound made it a miss
    assert cc.prefetch_hit_queue.get_nowait() is asked
    assert asked.storage_hit_count == 0
    raised = False
    with mock.patch("sglang.srt.managers.cache_controller.STORAGE_QUIESCE_WAIT_S", 0.2):
        try:
            cc.detach_storage_backend()
        except RuntimeError as e:
            raised = "detach aborted" in str(e)
    assert raised, "detach under an orphaned query did not raise"
    assert cc.enable_storage and not backend.closed and cc.storage_workers_alive()
    backend.query_hold.set()
    _wait_until(lambda: cc.storage_calls.wait_idle(0) == 0)
    backend.hold_query = False
    held = _submit(cache, "detach-held")
    _tick_until(cache, lambda hits, acks: hits > 0)
    cc.prefetch_hit_queue.get_nowait()
    backend.hold_get, backend.gets = True, 0
    backend.get_hold.clear()
    _dispatch(cache, held)
    _wait_until(lambda: backend.gets == 1)
    time.sleep(0.4)  # past the bound: the read runs orphaned on both ranks
    raised = False
    with mock.patch("sglang.srt.managers.cache_controller.STORAGE_QUIESCE_WAIT_S", 0.2):
        try:
            cc.detach_storage_backend()
        except RuntimeError as e:
            raised = "1 behind orphaned calls" in str(e)
    assert raised, "detach under an orphaned read did not raise"
    assert cc.enable_storage and not backend.closed
    assert cc.storage_workers_alive() and not cc.storage_stop_event.is_set()
    cache.check_hicache_events()
    assert _drain_counts(cache)[1] == 0
    assert held in cc._worker_transfers
    backend.get_hold.set()
    _wait_until(lambda: held not in cc._worker_transfers)
    cc.detach_storage_backend()
    assert backend.closed and cc.storage_calls is None and not cc.enable_storage
    assert not any(
        t is not None and t.is_alive()
        for t in (cc.prefetch_thread, cc.prefetch_io_aux_thread, cc.backup_thread)
    )
    assert cc.storage_calls is None  # nothing keeps the bounded proxy alive


def _rank_main(rank: int, rendezvous: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        _slow_backend_body(rank)
    finally:
        dist.destroy_process_group()


class TestStorageAgreementTwoRanks(unittest.TestCase):
    def test_slow_backend_on_one_rank_never_strands_the_peer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            torch.multiprocessing.spawn(
                _rank_main, args=(f"{tmpdir}/gloo",), nprocs=2, join=True
            )


if __name__ == "__main__":
    unittest.main()

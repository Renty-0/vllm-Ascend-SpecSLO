/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ALLREDUCE_ADD_RMSNORM_AIV_KERNEL_H
#define ALLREDUCE_ADD_RMSNORM_AIV_KERNEL_H

#include "kernel_operator.h"
#include "allreduce_add_rmsnorm_tiling.h"

using namespace AscendC;

// Keep the correctness-first rank-3 path comfortably below the 910B2 UB
// ceiling.  Larger tiles are admitted only after an eager/graph numerical
// sweep proves that every independently owned source buffer remains disjoint.
constexpr int32_t DIFUSION_ADD_LEN = 512;
constexpr int32_t TQUE_DEPTH = 1;
constexpr uint32_t TBUF_POOL_MAX_BUFID_SIZE = 8;
constexpr int32_t NUM_PER_REP_FP32 = 64;

// This standalone vector epilogue intentionally carries only the two
// constants it consumes.  Do not include the legacy mixed-core utility
// header here: it contains unrelated projection helpers and would make the
// packaged active source unnecessarily broad.
constexpr int32_t SWIZZL_MASK = 0b100000;
constexpr int64_t TP3_IPC_FLAG_OFFSET_BYTES =
    180LL * 1024 * 1024 + 4096;

// v126 deliberately does not use SyncAll.  Every AIV block owns a disjoint
// row chunk and performs a point-to-point rendezvous with the same logical
// block on the other TP ranks.  Give every control word its own 32-byte GM
// slot so concurrent remote writes cannot alias at the cache-line granularity.
constexpr int32_t TP3_AIV_ONLY_RANKS = 3;
constexpr int32_t TP3_AIV_ONLY_MAX_CORES = 64;
constexpr int32_t TP3_AIV_ONLY_SLOT_BYTES = 32;
constexpr int32_t TP3_CHAIN_STATE_WORDS = 4;
constexpr uint64_t TP3_CHAIN_STATE_SALT = 0xC4A17E5A5A17C4E1ULL;
enum Tp3AivOnlyPhase : int32_t {
    TP3_AIV_REQUEST = 0,
    TP3_AIV_RESPONSE,
    TP3_AIV_RECEIPT,
    TP3_AIV_READ_DONE,
    TP3_AIV_PHASES,
};

template <typename T>
__aicore__ void CopyUbufToGmAlignB16(__gm__ T *dst, __ubuf__ T *src, uint16_t nBurst, uint32_t lenBurst,
                                         uint16_t srcSTride, uint16_t dstStride)
{
    DataCopyExtParams dataCopyParams(nBurst,
                                     lenBurst,
                                     srcSTride,
                                     dstStride,
                                     0);
    LocalTensor<uint8_t> ubTensor;
    TBuffAddr ubAddr;
    ubAddr.logicPos = static_cast<uint8_t>(TPosition::VECIN);
    ubAddr.bufferAddr = reinterpret_cast<uint64_t>(src);
    ubTensor.SetAddr(ubAddr);
    GlobalTensor<uint8_t> gmTensor;
    gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(dst));
    DataCopyPad(gmTensor, ubTensor, dataCopyParams);
}

template <typename T>
__aicore__ void CopyGmToUbufAlignB16(__ubuf__ T *dst, __gm__ T *src, uint16_t nBurst, uint32_t lenBurst,
                                        uint16_t srcSTride, uint16_t dstStride)
{
    DataCopyExtParams dataCopyParams(nBurst,
                                     lenBurst,
                                     srcSTride,
                                     dstStride,
                                     0);
    LocalTensor<uint8_t> ubTensor;
    TBuffAddr ubAddr;
    ubAddr.logicPos = static_cast<uint8_t>(TPosition::VECIN);
    ubAddr.bufferAddr = reinterpret_cast<uint64_t>(dst);
    ubTensor.SetAddr(ubAddr);
    GlobalTensor<uint8_t> gmTensor;
    gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(src));
    DataCopyPadExtParams<uint8_t> padParams;
    DataCopyPad(ubTensor, gmTensor, dataCopyParams, padParams);
}

template <typename MmadDtype, typename OutDtype, bool GatherAddOut>
class AllreduceAddRmsnormAivKernel {

public:
    __aicore__ inline AllreduceAddRmsnormAivKernel<MmadDtype, OutDtype, GatherAddOut>() { }
    __aicore__ inline void Init(
        GM_ADDR local_projection, GM_ADDR residual, GM_ADDR gamma,
        GM_ADDR chain_state, GM_ADDR y, GM_ADDR add_out,
        GM_ADDR next_chain_state,
        GM_ADDR workspace, const AllreduceAddRmsnormTilingData *tilingData,
        Hccl<HCCL_SERVER_TYPE_AICPU> &hccl_)
    {
        this->hccl_ = hccl_;
        is_deterministic = false;
        auto ppTilingData = &tilingData->allreduceAddRmsnormInfo.ppTilingData;
        auto commTilingData = &tilingData->allreduceAddRmsnormInfo.commTilingData;
        auto quantInfo = &tilingData->allreduceAddRmsnormInfo.quantInfo;

        gm_local_projection = reinterpret_cast<__gm__ MmadDtype *>(local_projection);
        gm_out = reinterpret_cast<__gm__ MmadDtype *>(y);
        gm_add_input = reinterpret_cast<__gm__ MmadDtype *>(residual);
        gm_add_output = reinterpret_cast<__gm__ MmadDtype *>(add_out);
        gm_gamma = reinterpret_cast<__gm__ MmadDtype *>(gamma);
        gm_chain_state = reinterpret_cast<__gm__ uint64_t *>(chain_state);
        gm_next_chain_state = reinterpret_cast<__gm__ uint64_t *>(next_chain_state);

        batch_size = ppTilingData->opShape.batchSize;
        m = ppTilingData->opShape.m;
        k = ppTilingData->opShape.k;
        n = ppTilingData->opShape.n;
        avg_factor = (n != 0) ? (float)1.0 / n : 0;

        m0 = ppTilingData->m0;
        k0 = ppTilingData->k0;
        n0 = ppTilingData->n0;

        m_loop = ppTilingData->mLoop;
        k_loop = ppTilingData->kLoop;
        n_loop = ppTilingData->nLoop;

        core_loop = ppTilingData->coreLoop;
        swizzl_count = ppTilingData->swizzlCount;
        tiling_key = ppTilingData->tilingKey;
        rank = hccl_.GetRankId();
        rank_size = hccl_.GetRankDim();

        max_ub_single_dma_size = commTilingData->ubMoveNum;
        withSerialMode = false;
        tag = commTilingData->tag;
        comm_npu_split = commTilingData->commNpuSplit;
        comm_data_split = commTilingData->commDataSplit;
        comm_direct = commTilingData->commDirect;
        is_91093 = false;
        core_count = comm_npu_split * comm_data_split;
        dequant_granularity = static_cast<QuantGranularity>(quantInfo->dequantGranularity);
        dequant_group_size = quantInfo->dequantGroupSize;
        quant_granularity = static_cast<QuantGranularity>(quantInfo->quantGranularity);
        quant_group_size = quantInfo->quantGroupSize;
        epsilon = ppTilingData->epsilon;
        flush_chain = ppTilingData->flushChain != 0;
        swizzl_direct = (tiling_key & SWIZZL_MASK) ? true : false;
        trans_a = ppTilingData->isTransA;
        trans_b = ppTilingData->isTransB;
        projection_only = false;
        is_int8 = false;
        ag_dim = 0;
        rs_dim = 0;
        inner_dim_is_Ag = false;
        weight_nz = false;
        max_ub_ping_pong_size = max_ub_single_dma_size / 2; // 2 - double buffer

        core_idx = get_block_idx();
        core_num = get_block_num();
        // The vector-only entry has one logical vector core per block; there
        // is no paired sub-block dimension in this candidate.
        aiv_idx = 0;
        other_rank = (core_idx < rank_size) ? core_idx : -1;

        // init ub usage
        pipe.InitBuffer(ctrlBuf, AscendC::ONE_BLK_SIZE);
        ub_ctrl_flag = reinterpret_cast<__ubuf__ int32_t *>(ctrlBuf.Get<int32_t>().GetPhyAddr());

        pipe.InitBuffer(gammaBuf, n * sizeof(MmadDtype));

        uint32_t step1_ub_usage = AscendC::AlignUp(
            n * sizeof(MmadDtype) +
            2 * (rank_size * DIFUSION_ADD_LEN * sizeof(MmadDtype)) +
            (GatherAddOut ? n * sizeof(MmadDtype) : 0) +
            n * sizeof(MmadDtype) +
            n * sizeof(float) +
            n * sizeof(float) +
            n * sizeof(float),
            AscendC::ONE_BLK_SIZE);

        uint32_t step2_ub_usage = AscendC::AlignUp(
            max_ub_ping_pong_size * sizeof(MmadDtype),
            AscendC::ONE_BLK_SIZE) * 2;
        uint32_t max_step_ub_usage = max(step1_ub_usage, step2_ub_usage);

        pipe.InitBufPool(step1BufPool, max_step_ub_usage);
        pipe.InitBufPool(step2BufPool, max_step_ub_usage, step1BufPool);

        step1BufPool.InitBuffer(inQueueX, 1, n * sizeof(MmadDtype));
        step1BufPool.InitBuffer(inQueueY, 2, rank_size * DIFUSION_ADD_LEN * sizeof(MmadDtype));
        if constexpr (GatherAddOut) {
            step1BufPool.InitBuffer(addOutQueue, 1, n * sizeof(MmadDtype));
        }
        step1BufPool.InitBuffer(outQueue, 1, n * sizeof(MmadDtype));
        step1BufPool.InitBuffer(xFp32Buf, n * sizeof(float));
        step1BufPool.InitBuffer(sqxBuf, n * sizeof(float));
        step1BufPool.InitBuffer(reduceFp32Buf, n * sizeof(float));

        step2BufPool.InitBuffer(allgatherBuf[0], max_ub_ping_pong_size * sizeof(MmadDtype));
        step2BufPool.InitBuffer(allgatherBuf[1], max_ub_ping_pong_size * sizeof(MmadDtype));

        if (!projection_only) {
            CopyInGamma();
        }
    }

    __aicore__ inline void Process(const AllreduceAddRmsnormTilingData *tilingData)
    {
        // Native F.linear owns the local projection.  Each AIV copies only its
        // row chunk to this rank's symmetric HCCL window and then rendezvous
        // with the same logical AIV on the other TP ranks.  The per-chunk
        // protocol removes both the v124 cross-engine event and every
        // full-cohort barrier/schedule-mode dependency.
        // The preceding epilogue has already published every remote
        // READ_DONE record, but deliberately did not wait for this rank's
        // readers.  Wait here, before publishing a new request or overwriting
        // the symmetric payload window.  Native F.linear has therefore had a
        // complete projection window in which to hide this dependency.
        WaitPreviousConsumersDone();

        const uint64_t request_token = MakeInvocationToken(0x52ULL);
        const uint64_t producer_token = MakeInvocationToken(0x50ULL);

        PublishChunkRequests(request_token);
        CopyLocalProjectionToWindow();
        PipeBarrier<PIPE_ALL>();

        uint64_t peer_producer_tokens[TP3_AIV_ONLY_RANKS] = {0, 0, 0};
        ChunkReadyRendezvous(request_token, producer_token, peer_producer_tokens);

        int32_t core_offset_m = 0;
        int32_t m_cur_core = 0;
        GetBalancedRowShard(core_offset_m, m_cur_core);
        ParallelWithSplitStepOneAddNorm(core_offset_m * n, m_cur_core, true);

        PipeBarrier<PIPE_ALL>();
        PublishChunkReadDone(peer_producer_tokens);
        if (flush_chain) {
            WaitChunkConsumersDone(producer_token);
            ClearNextChainState();
        } else {
            StoreNextChainToken(producer_token);
            ClearInactiveNextChainState();
        }
    }


private:
    __aicore__ inline uint64_t MakeInvocationToken(uint64_t role_salt)
    {
        // GetSystemCycle is local and monotonic.  Tokens only need to be unique
        // for a fixed (rank, core, role) slot across successive invocations;
        // rank/core bits make diagnostics unambiguous and zero is reserved.
        uint64_t token = static_cast<uint64_t>(AscendC::GetSystemCycle());
        token ^= (static_cast<uint64_t>(rank + 1) << 56);
        token ^= (static_cast<uint64_t>(core_idx + 1) << 48);
        token ^= (role_salt << 32);
        return token == 0 ? 1 : token;
    }

    __aicore__ inline uint64_t ChainRecordSalt()
    {
        return TP3_CHAIN_STATE_SALT ^
            (static_cast<uint64_t>(rank + 1) << 40) ^
            (static_cast<uint64_t>(core_idx + 1) << 24);
    }

    __aicore__ inline __gm__ uint64_t *ChainStateRecord(
        __gm__ uint64_t *state, int32_t record_idx)
    {
        return state + static_cast<uint64_t>(record_idx) *
            TP3_CHAIN_STATE_WORDS;
    }

    __aicore__ uint64_t LoadPreviousChainToken()
    {
        auto ctrl = reinterpret_cast<__ubuf__ uint64_t *>(ub_ctrl_flag);
        auto record = ChainStateRecord(gm_chain_state, core_idx);
        const uint64_t salt = ChainRecordSalt();
        while (true) {
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            CopyGmToUbufAlignB16(
                ctrl, record, 1, TP3_AIV_ONLY_SLOT_BYTES, 0, 0);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID3);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID3);
            if (ctrl[0] == 0 && ctrl[1] == 0 &&
                ctrl[2] == 0 && ctrl[3] == 0) {
                return 0;
            }
            if (ctrl[0] != 0 && ctrl[1] == ~ctrl[0] &&
                ctrl[2] == (ctrl[0] ^ salt) && ctrl[3] == ~ctrl[2]) {
                return ctrl[0];
            }
            // A complemented 32-byte record may be observed while its DMA is
            // still becoming visible.  Never accept a torn token as a reason
            // to overwrite the previous producer window.
        }
    }

    __aicore__ void WaitPreviousConsumersDone()
    {
        const uint64_t previous_token = LoadPreviousChainToken();
        if (previous_token != 0) {
            WaitChunkConsumersDone(previous_token);
        }
    }

    __aicore__ void StoreNextChainToken(uint64_t producer_token)
    {
        auto ctrl = reinterpret_cast<__ubuf__ uint64_t *>(ub_ctrl_flag);
        const uint64_t salt = ChainRecordSalt();
        ctrl[0] = producer_token;
        ctrl[1] = ~producer_token;
        ctrl[2] = producer_token ^ salt;
        ctrl[3] = ~ctrl[2];
        SetFlag<HardEvent::S_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID2);
        CopyUbufToGmAlignB16(
            ChainStateRecord(gm_next_chain_state, core_idx),
            ctrl,
            1,
            TP3_AIV_ONLY_SLOT_BYTES,
            0,
            0);
        PipeSync<HardEvent::MTE3_S>();
    }

    __aicore__ void ClearNextChainState()
    {
        auto ctrl = reinterpret_cast<__ubuf__ uint64_t *>(ub_ctrl_flag);
        ctrl[0] = 0;
        ctrl[1] = 0;
        ctrl[2] = 0;
        ctrl[3] = 0;
        SetFlag<HardEvent::S_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID2);
        // Clear all 64 records, including rows not launched for this M, so a
        // completed graph exposes an unambiguous all-zero end state.
        for (int32_t record_idx = core_idx;
             record_idx < TP3_AIV_ONLY_MAX_CORES;
             record_idx += core_num) {
            CopyUbufToGmAlignB16(
                ChainStateRecord(gm_next_chain_state, record_idx),
                ctrl,
                1,
                TP3_AIV_ONLY_SLOT_BYTES,
                0,
                0);
        }
        PipeSync<HardEvent::MTE3_S>();
    }

    __aicore__ void ClearInactiveNextChainState()
    {
        // next_chain_state is an ACLGraph-visible output allocated with
        // empty_like.  Production launches at most 24 AIVs while the ABI
        // deliberately reserves 64 mailbox records, so the tail must be
        // definitionally written on every non-flush invocation.  The active
        // cohort stripes the disjoint inactive range to keep this off block
        // zero's critical path; active records were already written by their
        // corresponding blocks in StoreNextChainToken above.
        if (core_num >= TP3_AIV_ONLY_MAX_CORES) {
            return;
        }
        auto ctrl = reinterpret_cast<__ubuf__ uint64_t *>(ub_ctrl_flag);
        ctrl[0] = 0;
        ctrl[1] = 0;
        ctrl[2] = 0;
        ctrl[3] = 0;
        SetFlag<HardEvent::S_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID2);
        for (int32_t record_idx = core_num + core_idx;
             record_idx < TP3_AIV_ONLY_MAX_CORES;
             record_idx += core_num) {
            CopyUbufToGmAlignB16(
                ChainStateRecord(gm_next_chain_state, record_idx),
                ctrl,
                1,
                TP3_AIV_ONLY_SLOT_BYTES,
                0,
                0);
        }
        PipeSync<HardEvent::MTE3_S>();
    }

    __aicore__ inline __gm__ uint64_t *Tp3ChunkSlot(
        int32_t target_rank, int32_t phase, int32_t peer_rank)
    {
        uint64_t slot =
            (static_cast<uint64_t>(phase) * TP3_AIV_ONLY_MAX_CORES +
             static_cast<uint64_t>(core_idx)) * TP3_AIV_ONLY_RANKS +
            static_cast<uint64_t>(peer_rank);
        auto base = reinterpret_cast<__gm__ uint8_t *>(
            hccl_.GetWindowsInAddr(target_rank)) + TP3_IPC_FLAG_OFFSET_BYTES;
        return reinterpret_cast<__gm__ uint64_t *>(
            base + slot * TP3_AIV_ONLY_SLOT_BYTES);
    }

    __aicore__ inline bool IsRemotePeer(int32_t peer_rank)
    {
        // The local projection DMA and the following PIPE_ALL barrier already
        // order this block's read of its own symmetric-window shard.  Running
        // the four-phase remote visibility protocol against the same rank
        // therefore adds control DMAs and polling without protecting data.
        return peer_rank != rank;
    }

    __aicore__ inline void SetChunkToken(__gm__ uint64_t *dst, uint64_t value)
    {
        auto ctrl = reinterpret_cast<__ubuf__ uint64_t *>(ub_ctrl_flag);
        ctrl[0] = value;
        ctrl[1] = ~value;
        SetFlag<HardEvent::S_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID2);
        CopyUbufToGmAlignB16(dst, ctrl, 1, 2 * sizeof(uint64_t), 0, 0);
        PipeSync<HardEvent::MTE3_S>();
    }

    __aicore__ inline uint64_t GetChunkToken(__gm__ uint64_t *src)
    {
        auto ctrl = reinterpret_cast<__ubuf__ uint64_t *>(ub_ctrl_flag);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        CopyGmToUbufAlignB16(ctrl, src, 1, 2 * sizeof(uint64_t), 0, 0);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID3);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID3);
        // Remote 8/16-byte stores are not assumed atomic.  A torn cache-line
        // observation is treated as "not ready" and simply polled again.
        return ctrl[1] == ~ctrl[0] ? ctrl[0] : 0;
    }

    __aicore__ inline void SetChunkResponse(
        __gm__ uint64_t *dst, uint64_t producer_token, uint64_t request_token)
    {
        auto ctrl = reinterpret_cast<__ubuf__ uint64_t *>(ub_ctrl_flag);
        ctrl[0] = producer_token;
        ctrl[1] = request_token;
        ctrl[2] = ~producer_token;
        ctrl[3] = ~(producer_token ^ request_token);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID2);
        // One 32-byte response record binds the producer token to the request
        // it answers.  This avoids relying on visibility ordering between two
        // independent remote slots.
        CopyUbufToGmAlignB16(dst, ctrl, 1, TP3_AIV_ONLY_SLOT_BYTES, 0, 0);
        PipeSync<HardEvent::MTE3_S>();
    }

    __aicore__ inline bool GetChunkResponse(
        __gm__ uint64_t *src, uint64_t &producer_token, uint64_t &request_token)
    {
        auto ctrl = reinterpret_cast<__ubuf__ uint64_t *>(ub_ctrl_flag);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        CopyGmToUbufAlignB16(ctrl, src, 1, TP3_AIV_ONLY_SLOT_BYTES, 0, 0);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID3);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID3);
        producer_token = ctrl[0];
        request_token = ctrl[1];
        return producer_token != 0 &&
            ctrl[2] == ~producer_token &&
            ctrl[3] == ~(producer_token ^ request_token);
    }

    __aicore__ void PublishChunkRequests(uint64_t request_token)
    {
        for (int32_t source = 0; source < rank_size; ++source) {
            if (!IsRemotePeer(source)) {
                continue;
            }
            SetChunkToken(
                Tp3ChunkSlot(source, TP3_AIV_REQUEST, rank),
                request_token);
        }
    }

    __aicore__ void ChunkReadyRendezvous(
        uint64_t request_token,
        uint64_t producer_token,
        uint64_t (&peer_producer_tokens)[TP3_AIV_ONLY_RANKS])
    {
        uint64_t last_request[TP3_AIV_ONLY_RANKS] = {0, 0, 0};
        bool source_ready[TP3_AIV_ONLY_RANKS] = {false, false, false};
        bool consumer_received[TP3_AIV_ONLY_RANKS] = {false, false, false};
        // Pre-credit the local producer/consumer edge.  Its payload is
        // already ordered by CopyLocalProjectionToWindow + PIPE_ALL, so only
        // the two remote ranks participate in the mailbox protocol.
        peer_producer_tokens[rank] = producer_token;
        source_ready[rank] = true;
        consumer_received[rank] = true;
        int32_t ready_count = 1;
        int32_t received_count = 1;

        // Cooperative state machine: every block keeps servicing requests
        // while it polls its own responses.  This prevents the symmetric TP3
        // producer/consumer roles from deadlocking without a device-wide
        // barrier.  The producer token and echoed request are published in one
        // complemented 32-byte record.  The four-word invariant binds both
        // values in the same DMA and rejects a torn observation, so a second
        // identical remote read adds latency without strengthening the
        // publication contract.
        while (ready_count < rank_size || received_count < rank_size) {
            for (int32_t consumer = 0; consumer < rank_size; ++consumer) {
                if (!IsRemotePeer(consumer)) {
                    continue;
                }
                uint64_t observed_request = GetChunkToken(
                    Tp3ChunkSlot(rank, TP3_AIV_REQUEST, consumer));
                if (observed_request != last_request[consumer]) {
                    SetChunkResponse(
                        Tp3ChunkSlot(consumer, TP3_AIV_RESPONSE, rank),
                        producer_token,
                        observed_request);
                    last_request[consumer] = observed_request;
                }
                if (!consumer_received[consumer] &&
                    GetChunkToken(Tp3ChunkSlot(rank, TP3_AIV_RECEIPT, consumer)) ==
                        producer_token) {
                    consumer_received[consumer] = true;
                    ++received_count;
                }
            }

            for (int32_t source = 0; source < rank_size; ++source) {
                if (!IsRemotePeer(source) || source_ready[source]) {
                    continue;
                }
                uint64_t source_token = 0;
                uint64_t echoed_request = 0;
                auto response_slot = Tp3ChunkSlot(rank, TP3_AIV_RESPONSE, source);
                if (!GetChunkResponse(
                        response_slot, source_token, echoed_request) ||
                    echoed_request != request_token) {
                    continue;
                }
                peer_producer_tokens[source] = source_token;
                SetChunkToken(
                    Tp3ChunkSlot(source, TP3_AIV_RECEIPT, rank),
                    source_token);
                source_ready[source] = true;
                ++ready_count;
            }
        }
    }

    __aicore__ void PublishChunkReadDone(
        const uint64_t (&peer_producer_tokens)[TP3_AIV_ONLY_RANKS])
    {
        for (int32_t source = 0; source < rank_size; ++source) {
            if (!IsRemotePeer(source)) {
                continue;
            }
            SetChunkToken(
                Tp3ChunkSlot(source, TP3_AIV_READ_DONE, rank),
                peer_producer_tokens[source]);
        }
    }

    __aicore__ void WaitChunkConsumersDone(uint64_t producer_token)
    {
        for (int32_t consumer = 0; consumer < rank_size; ++consumer) {
            if (!IsRemotePeer(consumer)) {
                continue;
            }
            while (GetChunkToken(
                       Tp3ChunkSlot(rank, TP3_AIV_READ_DONE, consumer)) !=
                   producer_token) {
            }
        }
    }

    __aicore__ void CopyLocalProjectionToWindow()
    {
        int32_t row_offset = 0;
        int32_t rows = 0;
        GetBalancedRowShard(row_offset, rows);
        if (rows <= 0) {
            return;
        }
        uint32_t offset = static_cast<uint32_t>(row_offset * n);
        uint32_t count = static_cast<uint32_t>(rows * n);
        auto local_window = reinterpret_cast<__gm__ MmadDtype *>(hccl_.GetWindowsInAddr(rank));
        ParallelAllGather(local_window, gm_local_projection, offset, count);
    }

    __aicore__ inline void GetBalancedRowShard(int32_t &row_offset, int32_t &rows)
    {
        int32_t base = m / core_num;
        int32_t extra = m % core_num;
        rows = base + (core_idx < extra ? 1 : 0);
        row_offset = core_idx * base + min(core_idx, extra);
    }

    template <typename T>
    __aicore__ inline T min(const T& a, const T& b) {
        return (a < b) ? a : b;
    }

    template <typename T>
    __aicore__ inline T max(const T& a, const T& b) {
        return (a > b) ? a : b;
    }

    template <typename T>
    __aicore__ inline T LimitRange(const T& val, const T& low, const T& high) {
        return min(max(val, low), high);
    }

    template <AscendC::HardEvent EVENT>
    __aicore__ inline void PipeSync()
    {
        AscendC::TEventID event_id = static_cast<event_t>(GetTPipePtr()->FetchEventID(EVENT));
        AscendC::SetFlag<EVENT>(event_id);
        AscendC::WaitFlag<EVENT>(event_id);
    }

    // Reproduce the FP32 reduction tree used by CANN AddRmsNorm key 30.
    // AscendC's generic ReduceSum uses a different 910B implementation and
    // can perturb rstd enough to change the final BF16 result by one ULP.
    // This helper is selected only by the qualified TP3 BF16 direct path;
    // all other paths retain the generic reduction below.
    __aicore__ inline void Tp3NativeReduceSumFp32(
        const LocalTensor<float>& dst_local,
        const LocalTensor<float>& src_local,
        const LocalTensor<float>& work_local,
        int32_t count)
    {
        uint64_t mask = NUM_PER_REP_FP32;
        int32_t repeat_times = count / NUM_PER_REP_FP32;
        int32_t tail_count = count % NUM_PER_REP_FP32;
        int32_t body_count = repeat_times * NUM_PER_REP_FP32;
        BinaryRepeatParams repeat_params;
        repeat_params.src0RepStride = ONE_REPEAT_BYTE_SIZE / ONE_BLK_SIZE;
        repeat_params.src0BlkStride = 1;
        repeat_params.src1RepStride = 0;
        repeat_params.src1BlkStride = 1;
        repeat_params.dstRepStride = 0;
        repeat_params.dstBlkStride = 1;

        Duplicate(work_local, (float)0.0, NUM_PER_REP_FP32);
        PipeBarrier<PIPE_V>();
        if (likely(repeat_times > 0)) {
            Add(
                work_local,
                src_local,
                work_local,
                mask,
                repeat_times,
                repeat_params);
            PipeBarrier<PIPE_V>();
        }
        if (unlikely(tail_count != 0)) {
            Add(
                work_local,
                src_local[body_count],
                work_local,
                tail_count,
                1,
                repeat_params);
            PipeBarrier<PIPE_V>();
        }

        AscendCUtils::SetMask<float>(NUM_PER_REP_FP32);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
        if (g_coreType == AIV) {
            WholeReduceSum<float, false>(
                dst_local,
                work_local,
                MASK_PLACEHOLDER,
                1,
                0,
                1,
                0);
        }
#elif !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
        WholeReduceSum<float, false>(
            dst_local,
            work_local,
            MASK_PLACEHOLDER,
            1,
            1,
            1,
            DEFAULT_REPEAT_STRIDE);
#endif
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyInGamma()
    {
        GlobalTensor<MmadDtype> gamma_global;
        gamma_global.SetGlobalBuffer((__gm__ MmadDtype *)gm_gamma, n);
        DataCopy(gammaBuf.Get<MmadDtype>(), gamma_global, n);
        PipeSync<HardEvent::MTE2_V>();
    }

    __aicore__ void ParallelWithSplitStepOneAddNorm(
        uint32_t core_buf_offset, uint32_t m_cur_core, bool rank_ordered_direct_output)
    {
        if (m_cur_core <= 0) {
            return;
        }

        auto buff = (__gm__ MmadDtype *)hccl_.GetWindowsInAddr(
            rank_ordered_direct_output ? 0 : rank);

        GlobalTensor<MmadDtype> x_global;
        GlobalTensor<MmadDtype> y_global;
        GlobalTensor<MmadDtype> out_global;
        GlobalTensor<MmadDtype> add_out_global;

        x_global.SetGlobalBuffer(buff + core_buf_offset);
        out_global.SetGlobalBuffer(
            rank_ordered_direct_output ? gm_out + core_buf_offset : buff + core_buf_offset);
        add_out_global.SetGlobalBuffer(gm_add_output + core_buf_offset);

        uint32_t add_count = DivCeil(n, DIFUSION_ADD_LEN);
        // Keep the original ownership-based reduction as the correctness
        // reference.  The replicated TP3 path is enabled only after its
        // independent eager/graph qualification succeeds.
        bool dedicated_tp3_reduce = rank_ordered_direct_output;

        LocalTensor<MmadDtype> x_local;
        LocalTensor<MmadDtype> y_local;

        for (uint32_t i = 0; i < m_cur_core; i++) {
            LocalTensor<float> x_fp32 = xFp32Buf.Get<float>();
            LocalTensor<float> sqx = sqxBuf.Get<float>();
            LocalTensor<float> reduce_fp32 = reduceFp32Buf.Get<float>();

            // Keep every direct-reduce temporary inside the original seven
            // TBufPool IDs.  The pool has an eight-ID hardware limit; adding
            // one independent TBuf per peer aliases later buffers and can
            // silently turn rank0+rank1+rank2 into rank0+2*rank2.
            x_local = inQueueX.AllocTensor<MmadDtype>();
            for (uint32_t j = 0; j < add_count; j++) {
                uint32_t add_offset = j * DIFUSION_ADD_LEN;
                uint32_t add_len = min<uint32_t>(n - add_offset, DIFUSION_ADD_LEN);

                if (dedicated_tp3_reduce) {
                    uint32_t peer1_rank = rank_ordered_direct_output ? 1 : (rank + 1) % rank_size;
                    uint32_t peer2_rank = rank_ordered_direct_output ? 2 : (rank + 2) % rank_size;
                    auto peer1_buff = (__gm__ MmadDtype *)hccl_.GetWindowsInAddr(peer1_rank);
                    auto peer2_buff = (__gm__ MmadDtype *)hccl_.GetWindowsInAddr(peer2_rank);
                    GlobalTensor<MmadDtype> peer1_global;
                    GlobalTensor<MmadDtype> peer2_global;
                    peer1_global.SetGlobalBuffer(peer1_buff + core_buf_offset);
                    peer2_global.SetGlobalBuffer(peer2_buff + core_buf_offset);
                    DataCopy(x_local[add_offset], x_global[i * n + add_offset], add_len);
                    inQueueX.EnQue(x_local);
                    y_local = inQueueY.AllocTensor<MmadDtype>();
                    DataCopy(y_local, peer1_global[i * n + add_offset], add_len);
                    DataCopy(y_local[add_len], peer2_global[i * n + add_offset], add_len);
                    if (!projection_only) {
                        GlobalTensor<MmadDtype> residual_global;
                        residual_global.SetGlobalBuffer(gm_add_input + core_buf_offset);
                        DataCopy(y_local[2 * add_len], residual_global[i * n + add_offset], add_len);
                    }
                    inQueueY.EnQue(y_local);
                } else {
                    DataCopy(x_local[add_offset], x_global[i * n + add_offset], add_len);
                    inQueueX.EnQue(x_local);
                    y_local = inQueueY.AllocTensor<MmadDtype>();
                    for (uint32_t k = 0; k < rank_size; ++k) {
                        uint32_t iterate_idx = (rank + 1 + k) % rank_size;
                        if (iterate_idx == rank) {
                            if (projection_only) {
                                Duplicate(
                                    y_local[k * add_len],
                                    static_cast<MmadDtype>(0),
                                    add_len);
                                PipeBarrier<PIPE_V>();
                                continue;
                            }
                            y_global.SetGlobalBuffer(gm_add_input + core_buf_offset);
                        } else {
                            auto other_buff = (__gm__ MmadDtype *)hccl_.GetWindowsInAddr(iterate_idx);
                            y_global.SetGlobalBuffer(other_buff + core_buf_offset);
                        }
                        DataCopy(y_local[k * add_len], y_global[i * n + add_offset], add_len);
                    }
                    inQueueY.EnQue(y_local);
                }
                x_local = inQueueX.DeQue<MmadDtype>();
                y_local = inQueueY.DeQue<MmadDtype>();

                if (dedicated_tp3_reduce) {
                    // HCCL_OP_EXPANSION_MODE=AIV uses a deterministic global
                    // 01->2 BF16 tree for the qualified TP3 deployment.  The
                    // first pair must be materialized to BF16 before rank2;
                    // summing all three partials in FP32 and casting once can
                    // change a low-margin greedy token even though the local
                    // projection differs by only one BF16 ULP.
                    Cast(x_fp32[add_offset], x_local[add_offset], RoundMode::CAST_NONE, add_len);
                    Cast(sqx, y_local, RoundMode::CAST_NONE, add_len);
                    PipeBarrier<PIPE_V>();
                    Add(reduce_fp32[add_offset], x_fp32[add_offset], sqx, add_len);
                    PipeBarrier<PIPE_V>();
                    if constexpr (std::is_same<MmadDtype, bfloat16_t>::value) {
                        Cast(
                            x_local[add_offset],
                            reduce_fp32[add_offset],
                            RoundMode::CAST_RINT,
                            add_len);
                        PipeBarrier<PIPE_V>();
                        Cast(
                            reduce_fp32[add_offset],
                            x_local[add_offset],
                            RoundMode::CAST_NONE,
                            add_len);
                    }
                    Cast(sqx, y_local[add_len], RoundMode::CAST_NONE, add_len);
                    PipeBarrier<PIPE_V>();
                    Add(x_fp32[add_offset], reduce_fp32[add_offset], sqx, add_len);
                    PipeBarrier<PIPE_V>();
                    if (!projection_only) {
                        // Match the split HCCL + AddRMSNorm contract: the
                        // all-reduce result is materialized as BF16 before
                        // residual addition. Keeping both operations in FP32
                        // until the final cast changes add_out by up to one
                        // large BF16 ULP on real Qwen3 activations.
                        Cast(
                            x_local[add_offset],
                            x_fp32[add_offset],
                            RoundMode::CAST_RINT,
                            add_len);
                        PipeBarrier<PIPE_V>();
                        Cast(
                            reduce_fp32[add_offset],
                            x_local[add_offset],
                            RoundMode::CAST_NONE,
                            add_len);
                        Cast(
                            sqx,
                            y_local[2 * add_len],
                            RoundMode::CAST_NONE,
                            add_len);
                        PipeBarrier<PIPE_V>();
                        Add(
                            reduce_fp32[add_offset],
                            reduce_fp32[add_offset],
                            sqx,
                            add_len);
                        PipeBarrier<PIPE_V>();
                    }
                } else {
                    Cast(x_fp32[add_offset], x_local[add_offset], RoundMode::CAST_NONE, add_len);
                    PipeBarrier<PIPE_V>();
                    for (uint32_t k = 0; k < rank_size; ++k) {
                        // use sqx as shared buf, required n >= add_len
                        Cast(sqx, y_local[k * add_len], RoundMode::CAST_NONE, add_len);
                        PipeBarrier<PIPE_V>();
                        Add(x_fp32[add_offset], x_fp32[add_offset], sqx, add_len);
                        PipeBarrier<PIPE_V>();
                    }
                }

                inQueueY.FreeTensor(y_local);
            }
            inQueueX.FreeTensor(x_local);

            // The generic owner-reduce path accumulates in x_fp32, whereas
            // the replicated TP3 tree writes result_fp32.  Selecting the
            // wrong buffer here silently normalizes stale UB contents and
            // can look like a missing rank contribution.
            LocalTensor<float> reduced_fp32 =
                dedicated_tp3_reduce && !projection_only ? reduce_fp32 : x_fp32;

            if (projection_only) {
                LocalTensor<MmadDtype> out_local = outQueue.AllocTensor<MmadDtype>();
                Cast(out_local, reduced_fp32, RoundMode::CAST_RINT, n);
                PipeBarrier<PIPE_V>();
                outQueue.EnQue(out_local);
                out_local = outQueue.DeQue<MmadDtype>();
                DataCopy(out_global[i * n], out_local, n);
                outQueue.FreeTensor(out_local);
                continue;
            }

            if constexpr (GatherAddOut) {
                // copy add result out
                LocalTensor<MmadDtype> add_out = addOutQueue.AllocTensor<MmadDtype>();
                Cast(add_out, reduced_fp32, RoundMode::CAST_RINT, n);
                addOutQueue.EnQue(add_out);
                add_out = addOutQueue.DeQue<MmadDtype>();
                DataCopy(add_out_global[i * n], add_out, n);
                addOutQueue.FreeTensor(add_out);
            }

            LocalTensor<MmadDtype> gamma_local = gammaBuf.Get<MmadDtype>();
            LocalTensor<MmadDtype> out_local = outQueue.AllocTensor<MmadDtype>();
            // The dedicated path already owns the native pre-round residual
            // sum in reduce_fp32.  Normalize it in place and reuse x_fp32 as
            // the 64-lane reduction workspace.  Besides avoiding a full-row
            // copy, this makes it impossible for the normalization input to
            // silently acquire the independently materialized BF16 add_out
            // value through a queue/buffer reuse.
            LocalTensor<float> norm_fp32 =
                dedicated_tp3_reduce ? reduced_fp32 : x_fp32;
            LocalTensor<float> reduce_buf_local = dedicated_tp3_reduce ?
                x_fp32 : reduceFp32Buf.Get<float>();

            if (!dedicated_tp3_reduce) {
                // Keep the original generic-path BF16 materialization.
                Cast(out_local, reduced_fp32, RoundMode::CAST_RINT, n);
                PipeBarrier<PIPE_V>();

                Cast(norm_fp32, out_local, RoundMode::CAST_NONE, n);
                PipeBarrier<PIPE_V>();
            }

            Mul(sqx, norm_fp32, norm_fp32, n);
            PipeBarrier<PIPE_V>();

            Muls(sqx, sqx, avg_factor, n);
            PipeBarrier<PIPE_V>();

            if constexpr (std::is_same<MmadDtype, bfloat16_t>::value) {
                if (dedicated_tp3_reduce) {
                    Tp3NativeReduceSumFp32(
                        sqx,
                        sqx,
                        reduce_buf_local,
                        n);
                } else {
                    ReduceSum(sqx, sqx, reduce_buf_local, n);
                }
            } else {
                ReduceSum(sqx, sqx, reduce_buf_local, n);
            }
            PipeBarrier<PIPE_V>();

            Adds(sqx, sqx, epsilon, 1);
            PipeBarrier<PIPE_V>();

            Sqrt(sqx, sqx, 1);
            PipeBarrier<PIPE_V>();

            if (dedicated_tp3_reduce) {
                // Match CANN AddRmsNorm's FP32 rounding order exactly:
                // compute one scalar reciprocal, transfer it from V to S,
                // then scale the row with Muls.  Dividing every element by
                // sqrt(mean + eps) is algebraically equivalent but can differ
                // by one BF16 ULP after the following materialization.
                Duplicate(reduce_buf_local, (float)1.0, 1);
                PipeBarrier<PIPE_V>();

                Div(sqx, reduce_buf_local, sqx, 1);
                PipeBarrier<PIPE_V>();

                PipeSync<HardEvent::V_S>();
                float rstd_value = sqx.GetValue(0);
                PipeSync<HardEvent::S_V>();
                PipeBarrier<PIPE_V>();

                Muls(norm_fp32, norm_fp32, rstd_value, n);
                PipeBarrier<PIPE_V>();
            } else {
                Duplicate(reduce_buf_local, (float)1.0, 1);
                PipeBarrier<PIPE_V>();

                Div(sqx, reduce_buf_local, sqx, 1);
                PipeBarrier<PIPE_V>();

                PipeSync<HardEvent::V_S>();
                float rstd_value = sqx.GetValue(0);
                PipeSync<HardEvent::S_V>();
                PipeBarrier<PIPE_V>();

                Muls(norm_fp32, norm_fp32, rstd_value, n);
                PipeBarrier<PIPE_V>();
            }

            if constexpr (std::is_same<MmadDtype, half>::value) {
                Cast(out_local, norm_fp32, RoundMode::CAST_NONE, n);
                PipeBarrier<PIPE_V>();
                Mul(out_local, gamma_local, out_local, n);
                PipeBarrier<PIPE_V>();
            } else if constexpr (std::is_same<MmadDtype, bfloat16_t>::value) {
                Cast(out_local, norm_fp32, RoundMode::CAST_RINT, n);
                PipeBarrier<PIPE_V>();
                Cast(norm_fp32, out_local, RoundMode::CAST_NONE, n);
                PipeBarrier<PIPE_V>();
                Cast(sqx, gamma_local, RoundMode::CAST_NONE, n);
                PipeBarrier<PIPE_V>();

                Mul(norm_fp32, norm_fp32, sqx, n);
                PipeBarrier<PIPE_V>();
                Cast(out_local, norm_fp32, RoundMode::CAST_RINT, n);
                PipeBarrier<PIPE_V>();
                PipeSync<HardEvent::V_MTE2>();
            }

            outQueue.EnQue(out_local);
            out_local = outQueue.DeQue<MmadDtype>();
            DataCopy(out_global[i * n], out_local, n);
            outQueue.FreeTensor(out_local);
        }
    }

    __aicore__ void ParallelAllGather(__gm__ MmadDtype *gm_dst, __gm__ MmadDtype *gm_src,
        uint32_t core_buf_offset, uint32_t data_len)
    {
        GlobalTensor<MmadDtype> src_global;
        GlobalTensor<MmadDtype> dst_global;
        src_global.SetGlobalBuffer(gm_src);
        dst_global.SetGlobalBuffer(gm_dst);

        constexpr uint32_t PIPELINE_COPY_NUM = sizeof(allgatherBuf) / sizeof(allgatherBuf[0]);
        TEventID ev_mte3_mte2[PIPELINE_COPY_NUM];
        TEventID ev_mte2_mte3[PIPELINE_COPY_NUM];
        LocalTensor<MmadDtype> local_tensors[PIPELINE_COPY_NUM];

        for (uint32_t i = 0; i < PIPELINE_COPY_NUM; i++) {
            ev_mte3_mte2[i] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
            ev_mte2_mte3[i] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>();
            SetFlag<HardEvent::MTE3_MTE2>(ev_mte3_mte2[i]);
            local_tensors[i] = allgatherBuf[i].Get<MmadDtype>();
        }

        uint32_t offset = core_buf_offset;
        uint32_t copy_len = max_ub_ping_pong_size; // num of MmadDtype, not the byte length
        uint32_t copy_count = DivCeil(data_len, copy_len);
        uint32_t pipe_id = 0;

        for (uint32_t i = 0; i < copy_count; i++) {
            uint32_t actual_copy_len =
                (i == copy_count - 1) ? (data_len - i * copy_len) : copy_len;

            auto &local_tensor = local_tensors[pipe_id];

            WaitFlag<HardEvent::MTE3_MTE2>(ev_mte3_mte2[pipe_id]);
            DataCopy(local_tensor, src_global[offset], actual_copy_len);
            SetFlag<HardEvent::MTE2_MTE3>(ev_mte2_mte3[pipe_id]);
            WaitFlag<HardEvent::MTE2_MTE3>(ev_mte2_mte3[pipe_id]);
            DataCopy(dst_global[offset], local_tensor, actual_copy_len);
            SetFlag<HardEvent::MTE3_MTE2>(ev_mte3_mte2[pipe_id]);

            offset += actual_copy_len;
            pipe_id = (pipe_id + 1) % PIPELINE_COPY_NUM;
        }

        for (uint32_t i = 0; i < PIPELINE_COPY_NUM; i++) {
            WaitFlag<HardEvent::MTE3_MTE2>(ev_mte3_mte2[i]);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(ev_mte3_mte2[i]);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE3>(ev_mte2_mte3[i]);
        }

        PipeBarrier<PIPE_ALL>();
    }

    __gm__ MmadDtype *gm_local_projection;
    __gm__ MmadDtype *gm_out;
    __gm__ MmadDtype *gm_add_input;
    __gm__ MmadDtype *gm_add_output;
    __gm__ MmadDtype *gm_gamma;
    __gm__ uint64_t *gm_chain_state;
    __gm__ uint64_t *gm_next_chain_state;
    __ubuf__ int32_t *ub_ctrl_flag;

    int32_t batch_size;
    int32_t m;
    int32_t k;
    int32_t n;
    int32_t m0;
    int32_t k0;
    int32_t n0;

    int32_t m_loop;
    int32_t n_loop;
    int32_t k_loop;
    int32_t core_loop;
    int32_t core_idx;

    int32_t rank;
    int32_t rank_size;
    int32_t tiling_key;
    int32_t swizzl_count;
    bool swizzl_direct;

    bool trans_a;
    bool trans_b;
    bool projection_only;
    bool is_int8;
    bool is_91093;

    int32_t aiv_idx;
    int32_t other_rank;
    int32_t core_num;
    int32_t max_ub_single_dma_size;
    int32_t max_ub_ping_pong_size;

    int32_t gm_c_pingpong_size;
    int32_t withSerialMode;
    int32_t tag;
    int32_t comm_npu_split;
    int32_t comm_data_split;
    int32_t comm_direct;

    int32_t core_count;
    bool is_deterministic;

    QuantGranularity dequant_granularity;
    int32_t dequant_group_size;
    QuantGranularity quant_granularity;
    int32_t quant_group_size;

    int32_t ag_dim;
    int32_t rs_dim;
    bool inner_dim_is_Ag;
    bool weight_nz{false};
    bool flush_chain{true};

    float epsilon;
    float avg_factor;

    TPipe pipe;
    AscendC::TBufPool<TPosition::VECCALC, TBUF_POOL_MAX_BUFID_SIZE> step1BufPool;
    AscendC::TBufPool<TPosition::VECCALC, TBUF_POOL_MAX_BUFID_SIZE> step2BufPool;

    AscendC::TQue<AscendC::QuePosition::VECIN, TQUE_DEPTH> inQueueX, inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, TQUE_DEPTH> outQueueZ;
    AscendC::TQue<AscendC::QuePosition::VECOUT, TQUE_DEPTH> addOutQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, TQUE_DEPTH> outQueue;

    AscendC::TBuf<TPosition::VECCALC> ctrlBuf;
    AscendC::TBuf<TPosition::VECCALC> gammaBuf;
    AscendC::TBuf<TPosition::VECCALC> xFp32Buf;
    AscendC::TBuf<TPosition::VECCALC> sqxBuf;
    AscendC::TBuf<TPosition::VECCALC> reduceFp32Buf;
    AscendC::TBuf<TPosition::VECCALC> allgatherBuf[2];

    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
};
#endif

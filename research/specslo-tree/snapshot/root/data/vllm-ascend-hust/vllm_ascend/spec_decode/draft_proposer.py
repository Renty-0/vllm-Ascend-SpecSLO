import torch
from vllm.config import VllmConfig
from vllm.tokenizers.registry import get_tokenizer
from vllm.v1.spec_decode.draft_model import DraftModelProposer
from vllm.v1.spec_decode.vocab_mapping import VocabMapping

from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer
from vllm_ascend.spec_decode.pearl.vocab import PearlVocabProjection
from vllm_ascend.utils import lmhead_tp_enable


class AscendDraftModelProposer(DraftModelProposer, AscendSpecDecodeBaseProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        AscendSpecDecodeBaseProposer.__init__(self, vllm_config, device, False, runner=runner)
        speculative_config = self.speculative_config
        self.use_heterogeneous_vocab = speculative_config.use_heterogeneous_vocab
        self.pearl_vocab_projection: PearlVocabProjection | None = None
        if self.use_heterogeneous_vocab:
            if lmhead_tp_enable():
                raise ValueError(
                    "PEARL heterogeneous-vocabulary decoding does not support "
                    "finegrained_tp_config.lmhead_tensor_parallel_size. Disable "
                    "LM-head TP so PEARL can crop full target logits."
                )
            draft_model_config = speculative_config.draft_model_config
            target_model_config = speculative_config.target_model_config
            target_tokenizer = get_tokenizer(
                target_model_config.tokenizer,
                trust_remote_code=target_model_config.trust_remote_code,
            )
            draft_tokenizer = get_tokenizer(
                draft_model_config.model,
                trust_remote_code=draft_model_config.trust_remote_code,
            )
            self.vocab_mapping = VocabMapping(
                target_tokenizer=target_tokenizer,
                draft_tokenizer=draft_tokenizer,
                target_vocab_size=target_model_config.get_vocab_size(),
                draft_vocab_size=draft_model_config.get_vocab_size(),
                device=device,
            )
            self.pearl_vocab_projection = PearlVocabProjection.from_tokenizers(
                draft_tokenizer=draft_tokenizer,
                target_tokenizer=target_tokenizer,
                draft_vocab_size=draft_model_config.get_vocab_size(),
                target_vocab_size=target_model_config.get_vocab_size(),
            )
        else:
            self._raise_if_vocab_size_mismatch()
        self._raise_if_draft_tp_mismatch()

    def _maybe_share_lm_head(self, target_language_model) -> None:
        """Install the Ascend ACL graph wrapper without sharing the LM head.

        ``DraftModelProposer`` intentionally skips LM-head sharing, but the
        Ascend base performs ACL graph setup at the end of that hook. Delegate
        to the Ascend implementation: its method-specific sharing branches do
        not apply to ``draft_model`` and therefore retain the draft LM head.
        """
        AscendSpecDecodeBaseProposer._maybe_share_lm_head(self, target_language_model)

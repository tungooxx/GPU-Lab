"""Versioned future 9-arm runner entry with source-bound public history, NO SCIENCE.

The historical V5 source remains untouched. This is not a Kaggle submission.
"""
from __future__ import annotations
import json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(ROOT/'a1_stage1u_v8_real_native_public_binding_20261102'))
sys.path.insert(0,str(ROOT/'a1_stage1u_v9_streamed_prefill_20261102'))
sys.path.insert(0,str(ROOT/'a1_stage1u_v2_A2_material_repair_exact_9arm_20261031'))
import strict_native_public_input_guard_v8 as guard
import plancarry_qwen25_stage1u_v9_streamed_future_NO_SCIENCE as old

def proposed_arm_preflight_bound_v8(model,donor,public_ids,*,goal_ids,public_context_separator_ids,
                                    source_packet,tokenizer_adapter,native_receipts_by_arm,
                                    original_native_game_sha,cycle=0,uninterrupted_source_cache=None,
                                    frozen_eos_token_ids=None,max_generated_reply_tokens=128):
    """Source input verification precedes ALL model forwards, no release permission inferred."""
    pub=public_ids[0].tolist() if hasattr(public_ids,'tolist') and len(public_ids.shape)==2 else public_ids
    if not isinstance(pub,list):raise TypeError('REAL_PUBLIC_QWEN_INPUT_TENSOR_REQUIRED')
    goal=goal_ids[0].tolist() if hasattr(goal_ids,'tolist') and len(goal_ids.shape)==2 else goal_ids
    guard.guard_model_forward(cycle=cycle,public_ids=pub,goal_ids=goal,tokenizer=tokenizer_adapter)
    # Currently unreachable by design for the original owner-unapproved frozen source.
    # Future independently admitted new source pack requires an explicit versioned release.
    return old.proposed_arm_preflight(model,donor,public_ids,
        goal_ids=goal_ids,public_context_separator_ids=public_context_separator_ids,
        source_packet=source_packet,tokenizer_adapter=tokenizer_adapter,
        native_receipts_by_arm=native_receipts_by_arm,
        original_native_game_sha=original_native_game_sha,cycle=cycle,
        uninterrupted_source_cache=uninterrupted_source_cache,
        frozen_eos_token_ids=frozen_eos_token_ids,max_generated_reply_tokens=max_generated_reply_tokens)

def dry_run():
    snap=guard.load_immutable_snapshot()
    evidence=guard.verify_public_lineage(snap)
    return {'schema':'PLANCARRY_STAGE1U_V8_BOUND_FUTURE_MODEL_ENTRY_NO_SCIENCE',
            'status':'PRE_SCIENCE_RELEASE_FAIL_CLOSED',
            'gating':'PREFLIGHT_FAILED',
            'source_custody':evidence,
            'old_source_runner_mutated':False,
            'nine_arms':list(old.FROZEN_ARMS),
            'owner_approved_future_source':False,
            'no_model_or_GPU_loaded':True}
if __name__=='__main__':print(json.dumps(dry_run(),sort_keys=True))

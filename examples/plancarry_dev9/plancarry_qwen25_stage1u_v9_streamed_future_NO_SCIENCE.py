"""PlanCarry Stage1 U frozen-model Kaggle T4x2 source and cache protocol — PRE-SCIENCE ONLY.
Author's original plan may be wrong. It must NEVER be corrected/admitted via ALFWorld WIN.
This is candidate implementation source, NOT approved to run Qwen or ALFWorld.
Top-level entry only performs NO-GPU manifest validation and ALWAYS blocks model execution.
Separate independent review, exact SHA and scientific gate PASS are required before a
new authorized runner can invoke any model-using function. Never misreport dry-run as science.
"""
from __future__ import annotations
import hashlib
import json
import pathlib
import sys
from dataclasses import dataclass

MODEL_REPO = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
TRANSFORMERS_VERSION = "4.57.1"
WEIGHTS_DTYPE = "float16"
TORCH_DTYPE = "torch.float16"
GPU_COUNT = 2
N_RESETS = (0,1,2,3)

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent.parent/'a1_stage1u_v5_literal_text_9arm_20261101'))
from streamed_prefill_v9 import forward_streamed
from plancarry_stage1u_v2_exact_arms_public_text_source_custody import FROZEN_ARMS
ARM_NAMES = FROZEN_ARMS # exact frozen scientific Stage1U v2

SOURCE_DONOR_RULE = "SELF_GENERATED_PRECOMPACTION_QWEN_TOKENS_ONLY"
SCIENCE_GATE = "082d1bcf-0f34-43e8-924a-80104f79fa19"
ARTIFACT_SCHEMA = "PLANCARRY_STAGE1U_A2_MATERIAL_REPAIR_V3_RAW_REPLY_SOURCE_ONLY_NO_SCIENCE"

def jsha(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode("utf-8")).hexdigest()

def assert_public_packet(packet):
    required={"episode_id","source_public_messages","recipient_public_tokens_by_cycle","native_state_sha256_by_cycle","legal_menu_sha256_by_cycle","public_goal_only_tokens_by_cycle"}
    missing=required-set(packet)
    if missing:raise ValueError(f"MISSING_PUBLIC_RECEIPTS:{','.join(sorted(missing))}")
    if len(packet["recipient_public_tokens_by_cycle"])!=4:
        raise ValueError("EXACT_FOUR_CYCLE_PUBLIC_RECEIPTS_REQUIRED")
    if len(packet["native_state_sha256_by_cycle"])!=4 or len(packet["legal_menu_sha256_by_cycle"])!=4:
        raise ValueError("STATE_AND_MENU_PER_CYCLE_REQUIRED")
    if len(packet["public_goal_only_tokens_by_cycle"])!=4:
        raise ValueError("GOAL_CONTROL_PER_CYCLE_REQUIRED")
    if not isinstance(packet["source_public_messages"],list) or not packet["source_public_messages"]:
        raise ValueError("NONEMPTY_NATIVE_PUBLIC_SOURCE_REQUIRED")
    for i,x in enumerate(packet["recipient_public_tokens_by_cycle"]):
        if not isinstance(x,list) or not x or not all(isinstance(v,int) and v>=0 for v in x):
            raise ValueError(f"INVALID_PUBLIC_TOKENS_CYCLE_{i}")
    if any(not isinstance(x,str) or len(x)!=64 or any(c not in '0123456789abcdef' for c in x)
        for x in packet["native_state_sha256_by_cycle"]+packet["legal_menu_sha256_by_cycle"]):
        raise ValueError("NATIVE_STATE_AND_MENU_HASHES_REQUIRED")
    for i,x in enumerate(packet["public_goal_only_tokens_by_cycle"]):
        if not isinstance(x,list) or not x or any(type(v) is not int or v<0 for v in x):
            raise ValueError(f"GOAL_ONLY_TOKENS_REQUIRED_CYCLE_{i}")
    if packet.get("source_plan_text") or packet.get("task_gold") or packet.get("private_pddl"):
        raise ValueError("NO_SOURCE_PLAN_PREFILL_OR_PRIVATE_GOLD")
    return True

def assert_peer_branch_native_parity(native_receipts:dict):
    # The branch transcript is signed/retained externally; this only checks
    # that identical world state/menu is declared, never treats equality of hashes
    # as proof of native environment state fidelity by itself.
    required=set(ARM_NAMES)
    if set(native_receipts)!=required:raise ValueError("ARM_LEDGER_INCOMPLETE")
    for arm,recs in native_receipts.items():
        if len(recs)!=4:raise ValueError("FOUR_RECEIPTS_PER_ARM_REQUIRED")
    for i in range(4):
        world={(x[i]["native_state_sha256"],x[i]["legal_menu_sha256"],x[i]["public_turn_sha256"]) for x in native_receipts.values()}
        if len(world)!=1:raise ValueError(f"RECIPIENT_WORLD_MENU_PUBLIC_MISMATCH_CYCLE_{i}")
    return True

def enforce_hf_and_gpus(*, hf_version, cuda_devices, model_revision):
    if hf_version!=TRANSFORMERS_VERSION:raise RuntimeError("UNPINNED_HF_TRANSFORMERS_VERSION")
    if model_revision!=MODEL_REVISION:raise RuntimeError("UNPINNED_QWEN_MODEL_REVISION")
    if len(cuda_devices)!=GPU_COUNT or any("T4" not in d for d in cuda_devices):
        raise RuntimeError("OWNER_KAGGLE_TWO_T4_REQUIRED")
    return True

def hard_stop_science_not_authorized():
    raise RuntimeError("SCIENCE_GATE_PENDING_NO_QWEN_NO_KAGGLE_EXECUTION")

# The functions below are deliberately UNCALLED by the only __main__ entrypoint.
# Their validation semantics are part of a candidate independent-reviewed engineering
# release; they are NOT licensed merely by a string flag or a user-editable JSON boolean.

def source_generate_exact_hf_forward(model, source_ids, *, eos_ids: set[int],
                                     max_new_tokens: int, tokenizer_adapter):
    """Authorized-harness-only future runtime; genuine HF sampling, not greedy.
    The plan may be wrong; no WIN/format/validity selection or oracle filtering.
    """
    from qwen_source_original_sampling_kv import capture_original_qwen_generated_plan_kv
    if not hasattr(tokenizer_adapter, "decode"):
        raise TypeError("REAL_QWEN_TOKENIZER_ADAPTER_REQUIRED")
    receipt = capture_original_qwen_generated_plan_kv(
        model, tokenizer_adapter, source_ids, max_new_tokens=max_new_tokens)
    emitted = receipt["original_generated_tokens"]
    prefix = receipt["original_source_token_ids"]
    return {
        "source_live_cache_object": receipt["source_live_cache_object_NOT_FOR_HARD_RESET"],
        "source_token_ids": prefix,
        "generated_token_ids": emitted,
        "original_generated_plan_raw": receipt["original_generated_plan_raw"],
        "source_layers": receipt["original_source_layers_KV"],
        "source_prefix_len": receipt["source_prefix_len"],
        "emitted_token_sha256": jsha(emitted),
        "source_ids_sha256": jsha(prefix),
        "cache_position_after_forward": receipt["source_total_cache_len"],
        "source_generation_sampling": receipt["source_original_sampling"],
        "source_generation_genuine_qwen_not_filter_on_WIN": True,
        "source_final_emitted_token_KV_captured": True,
        "gate_remains_UNAPPROVED_in_current_build": True,
    }


def _get_layer_kv(cache):
    if hasattr(cache,"layers"):
        keys=[x.keys for x in cache.layers]
        vals=[x.values for x in cache.layers]
        if any(k is None or v is None for k,v in zip(keys,vals)):
            raise RuntimeError("SOURCE_CACHE_HAS_UNINITIALIZED_LAYER")
        return keys, vals
    if hasattr(cache,"key_cache") and hasattr(cache,"value_cache"):
        return list(cache.key_cache),list(cache.value_cache)
    raise TypeError("UNSUPPORTED_HF_DYNAMIC_CACHE_VERSION")

def _original_rope_rebase(layers,prefix_len,theta):
    import torch
    if theta<=0:raise ValueError("INVALID_ROPE_THETA")
    out=[]
    for key,value in layers:
        if key.shape[-2]<=prefix_len:raise ValueError("EMPTY_PLAN_TAIL")
        k=key[...,prefix_len:,:].contiguous().clone()
        v=value[...,prefix_len:,:].contiguous().clone()
        d=k.shape[-1]
        if d%2:raise RuntimeError("ROPE_HEAD_DIM_NOT_EVEN")
        inv=theta**(-torch.arange(0,d,2,dtype=torch.float64,device=k.device)/d)
        # Keys already have RoPE(pos+prefix). Shift phase by -prefix using split-half.
        theta_vec=(-float(prefix_len)*inv).to(torch.float64)
        cos=torch.cat((theta_vec.cos(),theta_vec.cos()),dim=-1).to(k.dtype)[None,None,None,:]
        sin=torch.cat((theta_vec.sin(),theta_vec.sin()),dim=-1).to(k.dtype)[None,None,None,:]
        half=d//2
        r=torch.cat((-k[...,half:],k[...,:half]),dim=-1)
        out.append((k*cos+r*sin,v))
    return tuple(out)

def _new_hf_cache_from_layers(kv):
    from transformers.cache_utils import DynamicCache
    fresh=DynamicCache()
    for i,(k,v) in enumerate(kv):
        fresh.update(k.detach().clone().contiguous(),v.detach().clone().contiguous(),i)
    return fresh

def recipient_forward_new_context(model, public_ids, *,carry_layers=None):
    """Fresh receiver with bounded prefill; every public token is consumed once."""
    recipient=None if carry_layers is None else _new_hf_cache_from_layers(carry_layers)
    result=forward_streamed(model,public_ids,restored_cache=recipient,chunk_size=256)
    return {"last_logits":result["last_logits"],
            "entry_new_receiver_empty":True,
            "restored_cache_tokens":result["restored_cache_tokens"],
            "public_tokens":result["input_tokens"],
            "post_receiver_cache_tokens":result["final_cache_tokens"],
            "prefill_chunks":result["chunks"],
            "all_tokens_processed_exactly_once":result["all_tokens_processed_exactly_once"],
            "live_receiver_cache_NOT_SERIALIZABLE":result["cache"]}

def _digest_tensor_bytes(tensor):
    import hashlib
    import torch
    t=tensor.detach().contiguous().cpu()
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()

def _validate_original_source_kv_tensor_receipts(donor,source_packet):
    """Actual Qwen K/V all layers incl final emitted token.
    This is an UNEXECUTED future runtime check, not an assertion of existing data.
    """
    layers=donor["source_layers"]
    receipts=source_packet["per_layer_original_generation_cache_receipts"]
    if len(layers)!=len(receipts):
        raise RuntimeError("SOURCE_ORIGINAL_LAYER_COUNT_MISMATCH")
    for (k,v),row in zip(layers,receipts):
        if k.shape[-2]!=row["sequence_length"] or v.shape[-2]!=row["sequence_length"]:
            raise RuntimeError("SOURCE_GENERATION_LAST_TOKEN_LENGTH_MISMATCH")
        if _digest_tensor_bytes(k)!=row["key_sha256"] or _digest_tensor_bytes(v)!=row["value_sha256"]:
            raise RuntimeError("SOURCE_GENERATED_KV_CUSTODY_SHA_MISMATCH")
    if donor["source_token_ids"]!=source_packet["source_prompt_token_ids"] or donor["generated_token_ids"]!=source_packet["actual_source_generated_token_ids"]:
        raise RuntimeError("SOURCE_DONOR_TOKENS_NOT_THE_SELF_GENERATED_ORIGINAL")
    return True

def uninterrupted_true_continuous_cycle(model,cache,public_ids):
    """Continuous upper bound, streamed to avoid long-history one-shot OOM."""
    if cache is None:
        raise RuntimeError("NO_LIVE_UNINTERRUPTED_QWEN_SOURCE_CACHE")
    past=int(cache.get_seq_length())
    if past<=0:
        raise RuntimeError("UNINTERRUPTED_SOURCE_KV_WAS_EMPTY")
    out=forward_streamed(model,public_ids,restored_cache=cache,chunk_size=256)
    return {"last_logits":out["last_logits"],
            "continuous_source_cache":out["cache"],"fresh_reset":False,
            "past_before":past,"cache_after":out["final_cache_tokens"],
            "effective_public_tokens":out["input_tokens"],
            "prefill_chunks":out["chunks"],
            "all_tokens_processed_exactly_once":out["all_tokens_processed_exactly_once"],
            "upper_bound_diagnostic_not_reset_peer":True}

def future_generate_raw_reply_from_cache(model,last_logits,live_cache,*,device,tokenizer_adapter,
                                        eos_token_ids,max_new_tokens):
    """Future-only exact greedy recipient continuation, never called by CLI.
    Carries actual emitted token through model/cache before stopping. Preserves
    EVERY raw token and verbatim decoded text, including malformed responses.
    """
    import torch
    if not getattr(tokenizer_adapter,"_provenance_verified",False):
        raise RuntimeError("UNVERIFIED_TOKENIZER_CANNOT_DECODE_RAW_MODEL_OUTPUT")
    if not isinstance(eos_token_ids,(set,tuple,list)) or not eos_token_ids or any(type(x) is not int or x<0 for x in eos_token_ids):
        raise RuntimeError("INVALID_FROZEN_QWEN_EOS_TOKEN_IDS")
    if type(max_new_tokens) is not int or not 1<=max_new_tokens<=512:
        raise RuntimeError("UNFROZEN_OR_UNBOUNDED_MODEL_OUTPUT_BUDGET")
    if live_cache is None:raise RuntimeError("MISSING_LIVE_RECEIVER_CACHE_FOR_ACTUAL_GENERATION")
    raw_ids=[]
    stopped="MAX_NEW_TOKENS"
    logits=last_logits
    first_cache_len=int(live_cache.get_seq_length())
    with torch.inference_mode():
        for j in range(max_new_tokens):
            raw=int(torch.argmax(logits,dim=-1).reshape(-1)[0].item())
            raw_ids.append(raw)
            ids=torch.tensor([[raw]],dtype=torch.long,device=device)
            previous_len=int(live_cache.get_seq_length())
            cursor=torch.tensor([previous_len],device=device,dtype=torch.long)
            mask=torch.ones((1,previous_len+1),dtype=torch.long,device=device)
            out=model(input_ids=ids,attention_mask=mask,
                      position_ids=cursor.unsqueeze(0),cache_position=cursor,
                      past_key_values=live_cache,use_cache=True,return_dict=True)
            live_cache=out.past_key_values
            if int(live_cache.get_seq_length())!=previous_len+1:
                raise RuntimeError("GENERATED_TOKEN_NOT_INCLUDED_IN_LIVE_KV")
            logits=out.logits[:,-1,:].detach().cpu()
            if raw in eos_token_ids:
                stopped="EOS"
                break
    decoded=tokenizer_adapter.decode_exact(raw_ids)
    if not isinstance(decoded,str):
        raise RuntimeError("REAL_TOKENIZER_DECODE_FAILED")
    return {"raw_generated_token_ids_including_specials":raw_ids,
            "raw_generated_token_ids_sha256":hashlib.sha256(
                json.dumps(raw_ids,separators=(',',':')).encode()).hexdigest(),
            "raw_generated_text_verbatim":decoded,
            "raw_generated_text_utf8_sha256":hashlib.sha256(decoded.encode('utf8')).hexdigest(),
            "stop_reason":stopped,"actual_emitted_token_count":len(raw_ids),
            "cache_length_before_generation":first_cache_len,
            "cache_length_after_generation":int(live_cache.get_seq_length()),
            "live_cache_object_NOT_SERIALIZABLE":live_cache,
            "task_goal_success_NOT_donor_admission":True,
            "malformed_or_empty_text_kept_in_ITT":True}


def proposed_arm_preflight(model,donor,public_ids,*,goal_ids,public_context_separator_ids,
                           source_packet,tokenizer_adapter,native_receipts_by_arm,
                           original_native_game_sha,cycle=0,uninterrupted_source_cache=None,
                           frozen_eos_token_ids=None,max_generated_reply_tokens=128):
    """No entrypoint can call this while scientific gate PENDING; kept solely
    as reviewer-inspectable future-model code, not authorized model science.
    All 9 arms EXACT frozen v2, no unapproved SOURCE_PREFIX_TEXT_ONLY substitution.
    TEXT means RAW input tokens into a fresh zero-KV recipient, not reencoded KV.
    """
    import torch
    from plancarry_stage1u_v2_exact_arms_public_text_source_custody import (
      FROZEN_ARMS,validate_exact_text_token_attestation,
      validate_native_receipt_ledger,plan_cycle_arm_dispatch)
    validate_exact_text_token_attestation(source_packet,tokenizer_adapter)
    parity=validate_native_receipt_ledger(native_receipts_by_arm,original_native_game_sha)
    if not parity["same_game_PDDL_public_parity"]:
        raise RuntimeError("NATIVE_PARITY_FAILED")
    _validate_original_source_kv_tensor_receipts(donor,source_packet)
    public_tokens=public_ids[0].tolist()
    goal_tokens=goal_ids[0].tolist() if getattr(goal_ids,"ndim",None)==2 else goal_ids
    specs=plan_cycle_arm_dispatch(source_packet,cycle,public_tokens,goal_tokens,
         public_context_separator_ids=public_context_separator_ids,
         uninterrupted_state_available=uninterrupted_source_cache is not None)
    original_full=donor["source_layers"]
    generated=donor["generated_token_ids"]
    prefix=donor["source_token_ids"]
    theta=float(getattr(model.config,"rope_theta"))
    rs=getattr(model.config,"rope_scaling",None)
    if rs and rs.get("rope_type",rs.get("type","default"))!="default":
        raise RuntimeError("UNSUPPORTED_NON_DEFAULT_QWEN_ROPE")
    source_tail=_original_rope_rebase(original_full,len(prefix),theta)
    source_prefix_only=tuple((k[...,:len(prefix),:].detach().clone(),
                              v[...,:len(prefix),:].detach().clone())
                             for k,v in original_full)
    def text_derived_kv(token_sequence):
        ids=torch.tensor([token_sequence],device=public_ids.device,dtype=torch.long)
        out=forward_streamed(model,ids,chunk_size=256)
        keys,vals=_get_layer_kv(out["cache"])
        return tuple((k.detach().clone(),v.detach().clone()) for k,v in zip(keys,vals))
    carriers={
      "ORIGINAL_GENERATION_PLAN_TAIL_KV_REBASED":source_tail,
      "ORIGINAL_FULL_SOURCE_KV_CONTEXT_DIAGNOSTIC":original_full,
      "ORIGINAL_SOURCE_PREFIX_ONLY_KV_CONTROL":source_prefix_only,
      "TEXT_DERIVED_PLAN_KV_PRIOR_ART":None,
    }
    # The only text-derived KV diagnostic. TEXT arms below never call this.
    if len(generated)>0:
        carriers["TEXT_DERIVED_PLAN_KV_PRIOR_ART"]=text_derived_kv(generated)
    outcomes={}
    new_upper_cache=uninterrupted_source_cache
    for arm in FROZEN_ARMS:
        route=specs[arm]
        if route["path"]=="CONTINUOUS_UNINTERRUPTED_MODEL_KV":
            if new_upper_cache is None:
                new_upper_cache=donor.get("source_live_cache_object")
            result=uninterrupted_true_continuous_cycle(model,new_upper_cache,public_ids)
            new_upper_cache=result.pop("continuous_source_cache")
            category="UNINTERRUPTED_DIAGNOSTIC_NOT_RESET_PEER"
        elif route["path"]=="FRESH_EMPTY_RECEIVER_PLUS_RESTORED_KV":
            if arm not in carriers:raise RuntimeError("UNFROZEN_OR_UNKNOWN_CARRIER")
            result=recipient_forward_new_context(model,public_ids,
                                                  carry_layers=carriers[arm])
            category="TRUE_FRESH_RECEIVER_KV_CARRIER"
        elif route["path"]=="FRESH_EMPTY_RECEIVER_DIRECT_TEXT_TOKENS":
            from literal_text_arm_adapter import render_then_tokenize_full_text
            literal=render_then_tokenize_full_text(
                arm=arm,
                source_token_ids=prefix,
                original_generated_token_ids=generated,
                public_separator_ids=public_context_separator_ids,
                public_full_history_ids=public_tokens,
                public_goal_ids=goal_tokens,
                tokenizer_adapter=tokenizer_adapter)
            text_ids=literal["token_ids"]
            ids=torch.tensor([text_ids],device=public_ids.device,dtype=torch.long)
            result=recipient_forward_new_context(model,ids,carry_layers=None)
            result["direct_TEXT_onepass_literal_receipt"]={
                 k:v for k,v in literal.items() if k!="token_ids"}
            if result["restored_cache_tokens"]!=0:
                raise RuntimeError("TEXT_REFERENCES_MUST_HAVE_ZERO_RESTORED_KV")
            category="TRUE_FRESH_RECEIVER_DIRECT_PUBLIC_TEXT"
        else:raise RuntimeError("UNKNOWN_FROZEN_ARM_EXECUTION_PATH")
        logits=result.pop("last_logits")
        live=result.pop("live_receiver_cache_NOT_SERIALIZABLE",None)
        if category=="UNINTERRUPTED_DIAGNOSTIC_NOT_RESET_PEER":
            live=new_upper_cache
        raw=future_generate_raw_reply_from_cache(model,logits,live,
            device=public_ids.device,tokenizer_adapter=tokenizer_adapter,
            eos_token_ids=frozen_eos_token_ids,
            max_new_tokens=max_generated_reply_tokens)
        live_after=raw.pop("live_cache_object_NOT_SERIALIZABLE")
        if category=="UNINTERRUPTED_DIAGNOSTIC_NOT_RESET_PEER":
            new_upper_cache=live_after
        outcomes[arm]={"arm":arm,"cycle":cycle,"category":category,
          "last_logits_sha256":hashlib.sha256(logits.numpy().tobytes()).hexdigest(),
          "native_goal_win_not_source_eligibility":True,
          "decoded_original_generated_source_plan_must_remain_unmodified":True,
          "predeclared_response_budget_tokens":max_generated_reply_tokens,
          **raw,**result}
    if tuple(outcomes)!=FROZEN_ARMS:
        raise RuntimeError("SOURCE_CODE_FROZEN_NINE_ARM_ORDER_CHANGED")
    return {"nine_arm_preflight_diagnostics":outcomes,
            "uninterrupted_live_cache_object_NOT_serializable":new_upper_cache,
            "full_Qwen_reply_generation_future_code_patched_but_NOT_EXECUTED":True,
            "native_hidden_rng_not_verified":True,
            "real_source_provenance_required":True,
            "scientific_gate_not_PASSED":True}


def dry_run_manifest():
    d={"schema":ARTIFACT_SCHEMA,"status":"NOT_ACCEPTED_FOR_TEST",
       "execution":"DRY_RUN_STD_LIB_ONLY_NO_MODEL_NO_GPU",
       "gate_id":SCIENCE_GATE,"gate_status":"PENDING",
       "model_repo":MODEL_REPO,"pinned_model_revision":MODEL_REVISION,
       "pinned_transformers_version":TRANSFORMERS_VERSION,"precision":WEIGHTS_DTYPE,
       "device_requirement":"Kaggle two NVIDIA Tesla T4, individually memory limited, no local RTX3050",
       "reset_cycles":list(N_RESETS),"arms":list(ARM_NAMES),
       "future_runner_functions":["source_generate_exact_hf_forward",
                                  "recipient_forward_new_context","proposed_arm_preflight"],
       "requested_native_receipts":["source_public_messages","recipient_public_tokens_by_cycle",
                                    "native_state_sha256_by_cycle","legal_menu_sha256_by_cycle",
                                    "public_goal_only_tokens_by_cycle"],
       "residual_gaps":["Actual 7B weights on Kaggle T4x2 load and model runtime not executed",
                        "No exact ALFWorld native environment scheduler implemented",
                        "No independent review PASS; scientific gate pending",
                        "Strong FULL PUBLIC source history consistency must be verified",
                        "No source progress transition oracle; static plan tail cannot update"],
       "source_is_model_self_authored_even_if_wrong":True,
       "no_source_plan_prefill":True,
       "no_borrowed_KV_claimed_novel":True,
       "no_science_metrics":True}
    return d

if __name__=="__main__":
    print(json.dumps(dry_run_manifest(),sort_keys=True,indent=2))
    # NEVER load a model or call source_generate_exact_hf_forward from this entry.
    hard_stop_science_not_authorized()

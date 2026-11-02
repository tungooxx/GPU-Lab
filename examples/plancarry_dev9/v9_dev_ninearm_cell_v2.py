# Isolated historical DEV diagnostic. Never imports or releases frozen Stage1 U.
import gc, hashlib, json, traceback, time
from pathlib import Path
from transformers import DynamicCache
OUT9=Path("/kaggle/working/plancarry_a1_9arm_dev")
OUT9.mkdir(parents=True,exist_ok=True)
RESULT9=OUT9/"NINE_ARM_FOUR_CUT_TECHNICAL.json"
ARMS9=(
"UNINTERRUPTED_CONTEXT_UPPER_BOUND",
"ORIGINAL_GENERATION_PLAN_TAIL_KV_REBASED",
"ORIGINAL_FULL_SOURCE_KV_CONTEXT_DIAGNOSTIC",
"ORIGINAL_SOURCE_PREFIX_ONLY_KV_CONTROL",
"TEXT_DERIVED_PLAN_KV_PRIOR_ART",
"COMPLETE_PUBLIC_SOURCE_TRANSCRIPT_AND_VERBATIM_PLAN_TEXT_STRONG_REFERENCE",
"VERBATIM_PLAN_TEXT_ONLY",
"NO_CARRY_PUBLIC_OBSERVATIONS",
"GOAL_ONLY_PUBLIC_CONTEXT")
r9={"schema":"PLANCARRY_A1_HISTORICAL_DEV_9ARM_QWEN_FOUR_CUT_TECH_V1",
    "source_scope":"HISTORICAL_EXPOSED_TRAIN_DEV_ONLY","gate":"PREFLIGHT_FAILED",
    "scientific_claim":"NONE","frozen_stage1_executed":False,"arms":list(ARMS9),
    "cycles":[],"errors":[],"status":"STARTED"}
def save9():
    RESULT9.write_text(json.dumps(r9,indent=2,sort_keys=True,ensure_ascii=False,default=str))
save9()
try:
    source_len=int(source_ids.shape[-1])
    source_layers=tuple((layer.keys.detach(),layer.values.detach()) for layer in oldcache.layers)
    prefix_layers=tuple((k[...,:source_len,:],v[...,:source_len,:]) for k,v in source_layers)
    plan_layers=tuple((k.detach(),v.detach()) for k,v in original_plan_tail)
    assert len(source_layers)==len(prefix_layers)==len(plan_layers)==28
    # Text-derived plan KV is built from literal original emitted tokens only.
    plan_ids=torch.tensor([raw],device="cuda:0",dtype=torch.long)
    def cache_from_layers(layers):
        cache=DynamicCache()
        if layers is not None:
            for ix,(k,v) in enumerate(layers):
                cache.update(k.detach().clone().contiguous(),v.detach().clone().contiguous(),ix)
        return cache
    def forward_ids(ids,layers=None,continuous=None):
        cache=continuous if continuous is not None else cache_from_layers(layers)
        before=int(cache.get_seq_length())
        total=int(ids.shape[-1])
        if total<1: raise RuntimeError("EMPTY_RECEIVER_INPUT")
        logits=None
        for offset in range(0,total,256):
            chunk=ids[:,offset:offset+256]
            end=before+offset+int(chunk.shape[-1])
            pos=torch.arange(before+offset,end,device="cuda:0",dtype=torch.long)
            mask=torch.ones((1,end),device="cuda:0",dtype=torch.long)
            with torch.inference_mode():
                result=model(input_ids=chunk,attention_mask=mask,position_ids=pos.unsqueeze(0),
                             cache_position=pos,past_key_values=cache,use_cache=True,
                             return_dict=True,logits_to_keep=1)
            cache=result.past_key_values
            logits=result.logits[0,-1].detach().float().cpu()
            del result
        if int(cache.get_seq_length())!=before+total:
            raise RuntimeError("CHUNKED_CACHE_LEDGER_MISMATCH")
        receipt={"restored_or_prior_cache_tokens":before,"input_tokens":total,
                 "chunks":(total+255)//256,"final_cache_tokens":int(cache.get_seq_length()),
                 "argmax":int(logits.argmax()),"logits_sha256":hashlib.sha256(logits.numpy().tobytes()).hexdigest(),
                 "all_tokens_processed_exactly_once":True}
        return receipt,cache
    derived_receipt,derived_cache=forward_ids(plan_ids)
    derived_layers=tuple((layer.keys.detach().clone(),layer.values.detach().clone()) for layer in derived_cache.layers)
    del derived_cache
    def generate_raw(cache,first_argmax,max_tokens=128):
        raw_ids=[]
        next_id=first_argmax
        reason="MAX_TOKENS"
        eos={tok.eos_token_id,tok.convert_tokens_to_ids("<|im_end|>")}
        for _ in range(max_tokens):
            raw_ids.append(next_id)
            before=int(cache.get_seq_length())
            token=torch.tensor([[next_id]],device="cuda:0",dtype=torch.long)
            pos=torch.tensor([before],device="cuda:0",dtype=torch.long)
            mask=torch.ones((1,before+1),device="cuda:0",dtype=torch.long)
            with torch.inference_mode():
                out=model(input_ids=token,attention_mask=mask,position_ids=pos.unsqueeze(0),
                          cache_position=pos,past_key_values=cache,use_cache=True,
                          return_dict=True,logits_to_keep=1)
            cache=out.past_key_values
            if int(cache.get_seq_length())!=before+1:
                raise RuntimeError("RAW_REPLY_TOKEN_NOT_COMMITTED_TO_KV")
            next_id=int(out.logits[0,-1].argmax())
            del out
            if raw_ids[-1] in eos:
                reason="EOS"
                break
        raw_text=tok.decode(raw_ids,skip_special_tokens=False)
        return {"raw_token_ids":raw_ids,"raw_text_verbatim":raw_text,
                "raw_text_sha256":hashlib.sha256(raw_text.encode()).hexdigest(),
                "stop_reason":reason,"raw_token_count":len(raw_ids),
                "reply_cache_after":int(cache.get_seq_length())},cache
    continuous_cache=cache_from_layers(source_layers)
    previous_history=None
    for cycle,history in enumerate(FULL_PUBLIC_HISTORY_BY_CUT):
        public_ids=tok.apply_chat_template([{"role":"user","content":history}],tokenize=True,
            add_generation_prompt=True,return_tensors="pt").to("cuda:0")
        full_text=source_literal+original_plan_literal+"\n\n[RECIPIENT PUBLIC OBSERVATIONS FOLLOW]\n"+tok.decode(public_ids[0].tolist(),skip_special_tokens=False)
        text_only=original_plan_literal+"\n\n[RECIPIENT PUBLIC OBSERVATIONS FOLLOW]\n"+tok.decode(public_ids[0].tolist(),skip_special_tokens=False)
        goal_only="Your task is to: look at alarmclock under the desklamp.\nCurrent cut: "+str(cycle)+"\nWhat should you do next?"
        def asids(s):
            return torch.tensor([tok.encode(s,add_special_tokens=False)],device="cuda:0",dtype=torch.long)
        routes={
          ARMS9[1]:(public_ids,plan_layers),
          ARMS9[2]:(public_ids,source_layers),
          ARMS9[3]:(public_ids,prefix_layers),
          ARMS9[4]:(public_ids,derived_layers),
          ARMS9[5]:(asids(full_text),None),
          ARMS9[6]:(asids(text_only),None),
          ARMS9[7]:(public_ids,None),
          ARMS9[8]:(asids(goal_only),None)}
        row={"cycle":cycle,"arms":{},"technical_only":True}
        # Upper bound is continuous by cache identity, but public history is
        # reintroduced in full each cut; not a matched peer or efficacy result.
        if previous_history is None:
            upper_ids=public_ids
        else:
            import os
            common=len(os.path.commonprefix([previous_history,history]))
            delta=history[common:]
            if not delta.startswith("Action executed after cut"):
                raise RuntimeError("CONTINUOUS_DELTA_NOT_NATIVE_ACTION")
            upper_ids=asids("\n"+delta)
        receipt,continuous_cache=forward_ids(upper_ids,continuous=continuous_cache)
        generated,continuous_cache=generate_raw(continuous_cache,receipt["argmax"])
        receipt.update(generated)
        receipt["category"]="UNMATCHED_CONTINUOUS_APPEND_ONLY_DIAGNOSTIC"
        row["arms"][ARMS9[0]]=receipt
        previous_history=history
        for arm in ARMS9[1:]:
            ids,layers=routes[arm]
            receipt,cache=forward_ids(ids,layers)
            generated,cache=generate_raw(cache,receipt["argmax"])
            receipt.update(generated)
            receipt["category"]="FRESH_RECEIVER"
            row["arms"][arm]=receipt
            del cache
            gc.collect()
            torch.cuda.empty_cache()
        if len(row["arms"])!=9:raise RuntimeError("INCOMPLETE_NINE_ARM_CYCLE")
        r9["cycles"].append(row)
        save9()
        del routes,public_ids
        gc.collect()
    r9["status"]="COMPLETE_DEV_TECHNICAL_ONLY"
    r9["limitation"]="Uninterrupted branch appends public deltas after model replies but follows historical fixed actions; native model-driven rollouts and frozen Stage1 semantics untested."
    save9()
except BaseException as exc:
    r9["status"]="FAILED_TECHNICAL_ONLY"
    r9["errors"].append({"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()})
    save9()
    raise

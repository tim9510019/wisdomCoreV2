import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import Dataset

MODEL_ID = "google/gemma-3-1b-it"
B_SIZE = 512

class QuantumRouterEngineAC:
    def __init__(self):
        print(f"🚀 [啟動] 量子疊加態數據引擎 AC 版 (多樣性保留 + N->N+B) (及時生成版)")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.squad = load_dataset("squad_v2", split="train")
        self.fineweb = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train[:100000]")
        self.haystack_texts = [ex["text"] for ex in self.fineweb]
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.code_templates = [
            ("void update_router(int gate_id) {{ if (gate_id == {0}) active = true; }}", True),
            ("export const ROUTE_CONFIG = {{ mode: 'FFT', depth: {0} }};", True),
            ("def check_gradient_hook(grad): return grad * 1.5 # Heart Booster ID: {0}", True),
            ("/* Quantum Tunneling Protocol Static Segment */", False)
        ]

    def _generate_trinity_logic(self):
        q1 = self.squad[random.randint(0, len(self.squad)-1)]
        q2 = self.squad[random.randint(0, len(self.squad)-1)]
        subject = q1['context'][:60].strip().replace('\n', ' ')
        ans_list = q2['answers']['text']
        secret_val = ans_list[0] if ans_list else f"TOKEN_{random.randint(1000,9999)}"
        
        fmt = random.choice(["json", "xml", "natural", "protocol"])
        if fmt == "json":
            fa = f'{{"node": "ALPHA", "subject": "{subject}"}}'
            fb = f'{{"node": "BETA", "value": "{secret_val}", "status": "verified"}}'
        elif fmt == "protocol":
            fa = f"SET_SUBJECT >>> {subject} [SIG_A]"
            fb = f"SET_VALUE >>> {secret_val} [SIG_B]"
        else:
            fa = f"實體標識為「{subject}」。"
            fb = f"該實體的解碼核心值為：{secret_val}。"

        fc = random.choice([
            f"!! 警告：與 {subject} 相關的所有路徑已在執行前被撤銷 !!",
            f"SYSTEM_OVERRIDE: 解碼值 {secret_val} 已失效，請標記為 REDACTED。",
            f"if (entity == '{subject}') {{ return NULL; // ACCESS_DENIED }}"
        ])
        
        query = f"\n[系統指令] 結合 '{subject}' 的解碼值給出結果，若失效回應 'REDACTED'。\n輸出："
        return fa, fb, fc, query, secret_val

    def _get_heterogeneous_filler(self, length):
        tokens = []
        while len(tokens) < length:
            if random.random() < 0.7:
                text = random.choice(self.haystack_texts)
            else:
                tpl, has_ph = random.choice(self.code_templates)
                text = tpl.format(random.randint(100, 999)) if has_ph else tpl
            tokens.extend(self.tokenizer(text, add_special_tokens=False).input_ids)
        return tokens[:length]

    def create_triplet_ac(self, target_base):
        target_len = int(target_base * random.uniform(0.98, 1.02))
        fa, fb, fc, query, secret_val = self._generate_trinity_logic()
        
        q_ids = self.tokenizer.encode(query, add_special_tokens=False)
        t_a = self.tokenizer.encode(fa, add_special_tokens=False)
        t_b = self.tokenizer.encode(fb, add_special_tokens=False)
        t_c = self.tokenizer.encode(fc, add_special_tokens=False)
        
        n_len = target_len - B_SIZE
        available_filler_len = n_len - len(q_ids)
        base_filler = self._get_heterogeneous_filler(available_filler_len)
        
        def inject_into_ocean(filler, content_list):
            res = list(filler)
            for p, content in sorted(content_list, key=lambda x: x[0], reverse=True):
                idx = int(len(filler) * p)
                res = res[:idx] + content + res[idx:] # 安全插入，不覆蓋
            return res[:available_filler_len] + q_ids # 安全截斷背景，保證 Query 在絕對尾部

        pos_a = random.uniform(0.05, 0.45)
        pos_b, pos_c = random.uniform(pos_a + 0.15, 0.90), random.uniform(0.10, 0.95)

        n_pos_ids = inject_into_ocean(base_filler, [(pos_a, t_a), (pos_b, t_b)])
        n_neg_ids = inject_into_ocean(base_filler, [(pos_a, t_a), (pos_c, t_c)])

        def build_b_block(ans_text):
            ans_ids = self.tokenizer.encode(ans_text, add_special_tokens=False)
            ans_ids.append(self.tokenizer.eos_token_id)
            if len(ans_ids) > B_SIZE: return ans_ids[:B_SIZE]
            return ans_ids + [self.tokenizer.pad_token_id] * (B_SIZE - len(ans_ids))

        pos_final = n_pos_ids + build_b_block(f" {secret_val}")
        neg_final = n_neg_ids + build_b_block(" REDACTED")

        # 這裡為了符合 transformers Trainer 的 collate_fn 預期，包成 torch.tensor
        return {
            "pos_ids": pos_final, 
            "neg_ids": neg_final
        }


class DynamicACDataset(Dataset):
    def __init__(self, engine, target_lengths, num_samples, seed=2026):
        self.engine = engine
        self.target_lengths = target_lengths
        self.num_samples = num_samples
        self.seed = seed

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 及時產生一筆數據
        base_len = random.choice(self.target_lengths)
        return self.engine.create_triplet_ac(base_len)

# ==========================================
# 1. 訓練包裝器 (動態 GC & Chunked Loss)
# ==========================================

class AGIV2GForCausalLM(nn.Module):
    def __init__(self, base_model, use_gc=True):
        super().__init__()
        self.base_model = base_model
        self.use_gc = use_gc

    def forward(self, input_ids, labels=None, **kwargs):
        # 🌟 攔截由 CPTDataCollator 傳遞下來的物理隔離邊界 (n_split_index)
        n_split_index = kwargs.get("n_split_index", None)

        hidden_states = self.base_model.embedding(input_ids)
        # 保留 Gemma 架構必須的縮放
        hidden_states = hidden_states * math.sqrt(self.base_model.D)
        
        for i, block in enumerate(self.base_model.blocks):
            shift_size = (block.C // 2) if (i % 2 != 0) else 0
            if self.training and self.use_gc:
                hidden_states = checkpoint.checkpoint(
                    block, hidden_states, shift_size, use_reentrant=False, n_split_index=n_split_index # 🌟 穿透傳遞給底層 AGIBlock
                )
            else:
                hidden_states = block(hidden_states, shift_size=shift_size, n_split_index=n_split_index) # 🌟 穿透傳遞給底層 AGIBlock
                
        hidden_states = self.base_model.final_norm(hidden_states)
        
        loss = None
        logits = None
        
        if labels is not None:
            shift_hidden = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(reduction="sum", ignore_index=-100)
            total_loss = 0.0
            valid_tokens = 0.0 
            chunk_size = 2048 
            seq_len = shift_hidden.size(1)
            
            for i in range(0, seq_len, chunk_size):
                end_idx = min(i + chunk_size, seq_len)
                c_hidden = shift_hidden[:, i:end_idx, :]
                c_logits = self.base_model.fc_out(c_hidden)
                c_labels = shift_labels[:, i:end_idx]
                
                c_loss = loss_fct(c_logits.reshape(-1, c_logits.size(-1)).float(), c_labels.reshape(-1))
                total_loss += c_loss
                
                valid_tokens += (c_labels != -100).sum().item()
                del c_logits, c_hidden
                
            loss = total_loss / max(valid_tokens, 1)
        else:
            logits = self.base_model.fc_out(hidden_states)
            
        return {"loss": loss, "logits": logits} if logits is not None else {"loss": loss}

# ==========================================
# 2. 權重移植、凍結與動態監控 Callback
# ==========================================

def transplant_and_freeze(model_id, agiv2_base):
    print(f"🔄 從 {model_id} 移植知識庫 (使用 dtype=bf16)...")
    src_model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cpu"
    )
    src_sd = src_model.state_dict()
    new_sd = {}

    new_sd['embedding.weight'] = src_sd.get('model.embed_tokens.weight')
    # 保留 Gemma 可能的權重綁定邏輯
    new_sd['fc_out.weight'] = src_sd.get('lm_head.weight') if 'lm_head.weight' in src_sd else src_sd.get('model.embed_tokens.weight')
    new_sd['final_norm.weight'] = src_sd.get('model.norm.weight')

    for i in range(26):
        s_pre = f"model.layers.{i}."
        a_pre = f"blocks.{i}."
        
        new_sd[f"{a_pre}input_layernorm.weight"] = src_sd.get(f"{s_pre}input_layernorm.weight")
        new_sd[f"{a_pre}post_attention_layernorm.weight"] = src_sd.get(f"{s_pre}post_attention_layernorm.weight")
        new_sd[f"{a_pre}pre_feedforward_layernorm.weight"] = src_sd.get(f"{s_pre}pre_feedforward_layernorm.weight")
        new_sd[f"{a_pre}post_feedforward_layernorm.weight"] = src_sd.get(f"{s_pre}post_feedforward_layernorm.weight")
        
        new_sd[f"{a_pre}q_norm.weight"] = src_sd.get(f"{s_pre}self_attn.q_norm.weight")
        new_sd[f"{a_pre}k_norm.weight"] = src_sd.get(f"{s_pre}self_attn.k_norm.weight")
        
        new_sd[f"{a_pre}ffn.gate_proj.weight"] = src_sd.get(f"{s_pre}mlp.gate_proj.weight")
        new_sd[f"{a_pre}ffn.up_proj.weight"] = src_sd.get(f"{s_pre}mlp.up_proj.weight")
        new_sd[f"{a_pre}ffn.down_proj.weight"] = src_sd.get(f"{s_pre}mlp.down_proj.weight")

        new_sd[f"{a_pre}W_q_loc.weight"] = src_sd.get(f"{s_pre}self_attn.q_proj.weight")
        new_sd[f"{a_pre}W_k_loc.weight"] = src_sd.get(f"{s_pre}self_attn.k_proj.weight")
        new_sd[f"{a_pre}W_v_loc.weight"] = src_sd.get(f"{s_pre}self_attn.v_proj.weight")
        new_sd[f"{a_pre}o_proj_loc.weight"] = src_sd.get(f"{s_pre}self_attn.o_proj.weight")

    new_sd = {k: v for k, v in new_sd.items() if v is not None}
    agiv2_base.load_state_dict(new_sd, strict=False)
    
    frozen_count, trainable_count = 0, 0
    for name, param in agiv2_base.named_parameters():
        param.requires_grad = name not in new_sd
        if param.requires_grad:
            param.data = param.data.to(torch.bfloat16)
            trainable_count += param.numel()
        else:
            frozen_count += param.numel()

    print(f"✅ 知識庫移植完成！")
    print(f"🔒 凍結 Gemma 3 參數: {frozen_count:,} | 🟢 新架構可訓練參數: {trainable_count:,}")

    del src_model
    import gc; gc.collect(); torch.cuda.empty_cache()
    return agiv2_base
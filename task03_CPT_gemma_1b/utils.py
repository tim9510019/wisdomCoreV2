import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import math
import queue
import threading
import concurrent.futures
import json
import pyarrow.parquet as pq

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import Dataset
import boto3
import botocore
from smart_open import open as smart_open

# ==========================================
# [ 雲端觀測區 ] 全域 S3 實體連線初始化
# ==========================================
# 使用無簽名配置以進行高頻率匿名存取
s3_client = boto3.client(
    "s3",
    region_name="us-west-2",
    config=botocore.config.Config(signature_version=botocore.UNSIGNED),
)

MODEL_ID = "google/gemma-3-1b-it"
B_SIZE = 512

# ==========================================
# 資料處理與異步 I/O 共用模組 (1K / 4K 共用)
# ==========================================


class AsyncS3Prefetcher:
    """背景非同步抓取 S3 程式碼的緩衝池"""

    def __init__(self, stream, max_workers=32, queue_size=500):
        self.stream = stream
        self.queue = queue.Queue(maxsize=queue_size)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.stop_event = threading.Event()
        self.producer_thread = threading.Thread(target=self._produce, daemon=True)
        self.producer_thread.start()
        print(
            f"⚡ 異步 S3 預取引擎啟動：{max_workers} 執行緒映射實體檔案 (針對 smollm)..."
        )

    def _fetch(self, blob_id: str) -> str:
        try:
            s3_url = f"s3://softwareheritage/content/{blob_id}"
            with smart_open(
                s3_url, "rb", compression=".gz", transport_params={"client": s3_client}
            ) as f:
                return f.read().decode("utf-8", errors="ignore")
        except Exception:
            return ""

    def _produce(self):
        try:
            for sample in self.stream:
                if self.stop_event.is_set():
                    break
                blob_id = sample.get("blob_id")
                if blob_id:
                    self.queue.put(self.executor.submit(self._fetch, blob_id))
        except Exception as e:
            print(f"\n⚠️ S3 預取引擎異常: {e}")
        finally:
            self.queue.put(None)

    def get_next_text(self) -> str:
        future = self.queue.get()
        if future is None:
            raise StopIteration("S3 程式碼資料流已枯竭")
        return future.result()


class AsyncRedPajamaFilter:
    """針對 SlimPajama 巨獸矩陣的獨立過濾器，精準萃取 GitHub"""

    def __init__(self, stream, queue_size=500):
        self.stream = stream
        self.queue = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.producer_thread = threading.Thread(target=self._produce, daemon=True)
        self.producer_thread.start()
        print("⚡ 異步過濾引擎啟動：獨立執行緒全速吞噬 SlimPajama，精準萃取 GitHub...")

    def _produce(self):
        try:
            for sample in self.stream:
                if self.stop_event.is_set():
                    break

                set_name = sample.get("redpajama_set_name")
                if not set_name:
                    meta = sample.get("meta", {})
                    if isinstance(meta, str):
                        try:
                            meta = json.loads(meta)
                        except:
                            meta = {}
                    set_name = meta.get("redpajama_set_name", "")

                if set_name == "RedPajamaGithub":
                    text = sample.get(
                        "text", sample.get("content", sample.get("code", ""))
                    )
                    if text and text.strip():
                        self.queue.put(text)
        except Exception as e:
            print(f"\n⚠️ RedPajama 過濾引擎異常: {e}")
        finally:
            self.queue.put(None)

    def get_next_text(self) -> str:
        text = self.queue.get()
        if text is None:
            raise StopIteration("RedPajama 資料流已徹底耗盡")
        return text


def verify_existing_matrix(output_file, target_sequences, tag="實體", exact_match=True):
    """實體拓撲檢驗，支援絕對比對 (1K) 或容許極小誤差比對 (4K)"""
    if not os.path.exists(output_file):
        print(f"🌌 觀測結果：{tag}磁區不存在，準備無中生有。")
        return False

    try:
        pf = pq.ParquetFile(output_file)
        num_rows = pf.metadata.num_rows
        print(f"📊 當前序列數 (Rows): {num_rows} / 預期序列數: {target_sequences}")

        if exact_match:
            if num_rows == target_sequences:
                print(f"✅ 拓撲完整性 100%！{tag}矩陣已完美定錨，無須重複運算。")
                return True
        else:
            if num_rows >= target_sequences * 0.99:
                print(f"✅ 拓撲完整性達標！{tag}矩陣已定錨。")
                return True

        print("⚠️ 序列數量不符，宇宙發生了坍縮。準備啟動超光速引擎進行覆寫...")
        return False
    except Exception as e:
        print(f"⚠️ 磁區讀取失敗 ({e})，檔案內部結構已損毀。準備啟動引擎覆寫...")
        return False


def get_boundary_ids(tokenizer):
    """預計算高維度拓撲邊界 Token IDs"""
    boundary_ids = set()
    boundary_chars = ["\n", ".", "?", "!", ";", "。", "！", "？", "；"]
    for s in (
        boundary_chars
        + [f" {c}" for c in boundary_chars]
        + [f"a{c}" for c in boundary_chars]
    ):
        tokens = tokenizer.encode(s, add_special_tokens=False)
        if tokens:
            boundary_ids.add(tokens[-1])
    return boundary_ids


def find_topological_boundary(
    seq,
    target_n,
    doc_sep_id,
    boundary_ids,
    offset_large,
    margin_large,
    offset_small,
    margin_small,
):
    """O(1) 邊界尋路演算法"""
    for offset in range(offset_large):
        left_idx = target_n - offset
        right_idx = target_n + offset
        if left_idx > margin_large and seq[left_idx] == doc_sep_id:
            return left_idx + 1
        if right_idx < len(seq) - margin_large and seq[right_idx] == doc_sep_id:
            return right_idx + 1

    for offset in range(offset_small):
        left_idx = target_n - offset
        right_idx = target_n + offset
        if left_idx > margin_small and seq[left_idx] in boundary_ids:
            return left_idx + 1
        if right_idx < len(seq) - margin_small and seq[right_idx] in boundary_ids:
            return right_idx + 1

    return target_n


# ==========================================
# 量子疊加態數據引擎 AC 版 & 訓練模型包裝器
# ==========================================


class QuantumRouterEngineAC:
    def __init__(self):
        print(f"🚀 [啟動] 量子疊加態數據引擎 AC 版 (多樣性保留 + N->N+B) (及時生成版)")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.squad = load_dataset("squad_v2", split="train")
        self.fineweb = load_dataset(
            "HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train[:100000]"
        )
        self.haystack_texts = [ex["text"] for ex in self.fineweb]

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.code_templates = [
            (
                "void update_router(int gate_id) {{ if (gate_id == {0}) active = true; }}",
                True,
            ),
            ("export const ROUTE_CONFIG = {{ mode: 'FFT', depth: {0} }};", True),
            (
                "def check_gradient_hook(grad): return grad * 1.5 # Heart Booster ID: {0}",
                True,
            ),
            ("/* Quantum Tunneling Protocol Static Segment */", False),
        ]

    def _generate_trinity_logic(self):
        q1 = self.squad[random.randint(0, len(self.squad) - 1)]
        q2 = self.squad[random.randint(0, len(self.squad) - 1)]
        subject = q1["context"][:60].strip().replace("\n", " ")
        ans_list = q2["answers"]["text"]
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

        fc = random.choice(
            [
                f"!! 警告：與 {subject} 相關的所有路徑已在執行前被撤銷 !!",
                f"SYSTEM_OVERRIDE: 解碼值 {secret_val} 已失效，請標記為 REDACTED。",
                f"if (entity == '{subject}') {{ return NULL; // ACCESS_DENIED }}",
            ]
        )

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
                res = res[:idx] + content + res[idx:]
            return res[:available_filler_len] + q_ids

        pos_a = random.uniform(0.05, 0.45)
        pos_b, pos_c = random.uniform(pos_a + 0.15, 0.90), random.uniform(0.10, 0.95)

        n_pos_ids = inject_into_ocean(base_filler, [(pos_a, t_a), (pos_b, t_b)])
        n_neg_ids = inject_into_ocean(base_filler, [(pos_a, t_a), (pos_c, t_c)])

        def build_b_block(ans_text):
            ans_ids = self.tokenizer.encode(ans_text, add_special_tokens=False)
            ans_ids.append(self.tokenizer.eos_token_id)
            if len(ans_ids) > B_SIZE:
                return ans_ids[:B_SIZE]
            return ans_ids + [self.tokenizer.pad_token_id] * (B_SIZE - len(ans_ids))

        pos_final = n_pos_ids + build_b_block(f" {secret_val}")
        neg_final = n_neg_ids + build_b_block(" REDACTED")

        return {"pos_ids": pos_final, "neg_ids": neg_final}


class DynamicACDataset(Dataset):
    def __init__(self, engine, target_lengths, num_samples, seed=2026):
        self.engine = engine
        self.target_lengths = target_lengths
        self.num_samples = num_samples
        self.seed = seed

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
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
        n_split_index = kwargs.get("n_split_index", None)

        hidden_states = self.base_model.embedding(input_ids)
        hidden_states = hidden_states * math.sqrt(self.base_model.D)

        for i, block in enumerate(self.base_model.blocks):
            shift_size = (block.C // 2) if (i % 2 != 0) else 0
            if self.training and self.use_gc:
                hidden_states = checkpoint.checkpoint(
                    block,
                    hidden_states,
                    shift_size,
                    use_reentrant=False,
                    n_split_index=n_split_index,
                )
            else:
                hidden_states = block(
                    hidden_states, shift_size=shift_size, n_split_index=n_split_index
                )

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

                c_loss = loss_fct(
                    c_logits.reshape(-1, c_logits.size(-1)).float(),
                    c_labels.reshape(-1),
                )
                total_loss += c_loss

                valid_tokens += (c_labels != -100).sum().item()
                del c_logits, c_hidden

            loss = total_loss / max(valid_tokens, 1)

            gate_anchor_loss = 0.0
            gate_count = 0
            for block in self.base_model.blocks:
                if hasattr(block, "gate_mem"):
                    gate_anchor_loss += (block.gate_fft.sum() - 1.0) ** 2
                    gate_anchor_loss += (block.gate_mem.sum() - 1.0) ** 2
                    gate_count += 2

            if gate_count > 0:
                gate_anchor_loss = gate_anchor_loss / gate_count
                loss = loss + 0.05 * gate_anchor_loss

        else:
            logits = self.base_model.fc_out(hidden_states)

        return (
            {"loss": loss, "logits": logits} if logits is not None else {"loss": loss}
        )


class AGIV2GForCausalLMT(nn.Module):
    def __init__(self, base_model, use_gc=True, gate_entropy_lambda=0.01):
        super().__init__()
        self.base_model = base_model
        self.use_gc = use_gc
        self.gate_entropy_lambda = gate_entropy_lambda

    def forward(self, input_ids, labels=None, **kwargs):
        n_split_index = kwargs.get("n_split_index", None)

        hidden_states = self.base_model.embedding(input_ids)
        hidden_states = hidden_states * math.sqrt(self.base_model.D)

        for i, block in enumerate(self.base_model.blocks):
            shift_size = (block.C // 2) if (i % 2 != 0) else 0
            if self.training and self.use_gc:
                hidden_states = checkpoint.checkpoint(
                    block,
                    hidden_states,
                    shift_size,
                    use_reentrant=False,
                    n_split_index=n_split_index,
                )
            else:
                hidden_states = block(
                    hidden_states, shift_size=shift_size, n_split_index=n_split_index
                )

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

                c_loss = loss_fct(
                    c_logits.reshape(-1, c_logits.size(-1)).float(),
                    c_labels.reshape(-1),
                )
                total_loss += c_loss

                valid_tokens += (c_labels != -100).sum().item()
                del c_logits, c_hidden

            loss = total_loss / max(valid_tokens, 1)

            if self.training:
                # 🌟 訓練時：加入閘門負熵正則化損失 (防止任何閘門完全歸零)
                if hasattr(self.base_model, "compute_gate_neg_entropy"):
                    gate_neg_entropy = self.base_model.compute_gate_neg_entropy()
                    # neg_entropy 值域 [-log3, 0]，加上 log3 使其值域變為 [0, log3]
                    gate_entropy_loss = gate_neg_entropy + math.log(3.0)

                    # 側通道儲存：供 callback 分開顯示訓練期間的分項數值
                    self._last_ce_loss = loss.detach().item()
                    self._last_gate_entropy_loss = gate_entropy_loss.detach().item()

                    # 總 loss = CE loss + λ * 負熵懲罰
                    loss = loss + self.gate_entropy_lambda * gate_entropy_loss
                else:
                    if not getattr(self, "_warned_no_gate_entropy", False):
                        print(
                            "⚠️ [AGIV2GForCausalLMT] base_model 缺少 compute_gate_neg_entropy()，"
                            "閘門負熵損失已停用。"
                        )
                        self._warned_no_gate_entropy = True
                    self._last_ce_loss = loss.detach().item()
                    self._last_gate_entropy_loss = 0.0
            # Eval 時：直接回傳純 CE loss，eval_loss 數值完全乾淨

        else:
            logits = self.base_model.fc_out(hidden_states)

        return (
            {"loss": loss, "logits": logits} if logits is not None else {"loss": loss}
        )


# ==========================================
# 3. AGIV3 訓練包裝器 (Byte Latent 漏斗架構)
# ==========================================
class AGIV3ForCausalLM(nn.Module):
    def __init__(self, base_model, use_gc=True, gate_entropy_lambda=0.1):
        super().__init__()
        self.base_model = base_model
        self.use_gc = use_gc
        self.gate_entropy_lambda = gate_entropy_lambda

    def forward(self, input_ids, labels=None, **kwargs):
        n_split_index = kwargs.get("n_split_index", None)

        # AGIV3 是漏斗架構，整個 forward 必須統一處理，不能逐塊拆散
        if self.training and self.use_gc:
            logits = checkpoint.checkpoint(
                self.base_model,
                input_ids,
                use_reentrant=False,
                n_split_index=n_split_index,
            )
        else:
            logits = self.base_model(input_ids, n_split_index=n_split_index)

        loss = None
        if labels is not None:
            # Shift 邏輯：Byte N 預測 Byte N+1
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)).float(),
                shift_labels.view(-1),
            )

            if self.training:
                # 🌟 訓練時：加入閘門負熵正則化損失 (防止任何閘門完全歸零)
                if hasattr(self.base_model, "compute_gate_neg_entropy"):
                    gate_neg_entropy = self.base_model.compute_gate_neg_entropy()
                    # neg_entropy 值域 [-log3, 0]，加上 log3 使其值域變為 [0, log3]
                    gate_entropy_loss = gate_neg_entropy + math.log(3.0)

                    # 側通道儲存：供 callback 分開顯示訓練期間的分項數值
                    self._last_ce_loss = loss.detach().item()
                    self._last_gate_entropy_loss = gate_entropy_loss.detach().item()

                    # 總 loss = CE loss + λ * 負熵懲罰
                    loss = loss + self.gate_entropy_lambda * gate_entropy_loss
                else:
                    # ⚠️ base_model 缺少 compute_gate_neg_entropy，跳過閘門損失
                    if not getattr(self, "_warned_no_gate_entropy", False):
                        print(
                            "⚠️ [AGIV3ForCausalLM] base_model 缺少 compute_gate_neg_entropy()，"
                            "閘門負熵損失已停用。請確認 AGIV3 版本是否正確。"
                        )
                        self._warned_no_gate_entropy = True
                    self._last_ce_loss = loss.detach().item()
                    self._last_gate_entropy_loss = 0.0
            # Eval 時：直接回傳純 CE loss，eval_loss 數值完全乾淨

        return {"loss": loss, "logits": logits}


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

    new_sd["embedding.weight"] = src_sd.get("model.embed_tokens.weight")
    new_sd["fc_out.weight"] = (
        src_sd.get("lm_head.weight")
        if "lm_head.weight" in src_sd
        else src_sd.get("model.embed_tokens.weight")
    )
    new_sd["final_norm.weight"] = src_sd.get("model.norm.weight")

    for i in range(26):
        s_pre = f"model.layers.{i}."
        a_pre = f"blocks.{i}."

        new_sd[f"{a_pre}input_layernorm.weight"] = src_sd.get(
            f"{s_pre}input_layernorm.weight"
        )
        new_sd[f"{a_pre}post_attention_layernorm.weight"] = src_sd.get(
            f"{s_pre}post_attention_layernorm.weight"
        )
        new_sd[f"{a_pre}pre_feedforward_layernorm.weight"] = src_sd.get(
            f"{s_pre}pre_feedforward_layernorm.weight"
        )
        new_sd[f"{a_pre}post_feedforward_layernorm.weight"] = src_sd.get(
            f"{s_pre}post_feedforward_layernorm.weight"
        )

        new_sd[f"{a_pre}q_norm.weight"] = src_sd.get(f"{s_pre}self_attn.q_norm.weight")
        new_sd[f"{a_pre}k_norm.weight"] = src_sd.get(f"{s_pre}self_attn.k_norm.weight")

        new_sd[f"{a_pre}ffn.gate_proj.weight"] = src_sd.get(
            f"{s_pre}mlp.gate_proj.weight"
        )
        new_sd[f"{a_pre}ffn.up_proj.weight"] = src_sd.get(f"{s_pre}mlp.up_proj.weight")
        new_sd[f"{a_pre}ffn.down_proj.weight"] = src_sd.get(
            f"{s_pre}mlp.down_proj.weight"
        )

        new_sd[f"{a_pre}W_q_loc.weight"] = src_sd.get(f"{s_pre}self_attn.q_proj.weight")
        new_sd[f"{a_pre}W_k_loc.weight"] = src_sd.get(f"{s_pre}self_attn.k_proj.weight")
        new_sd[f"{a_pre}W_v_loc.weight"] = src_sd.get(f"{s_pre}self_attn.v_proj.weight")
        new_sd[f"{a_pre}o_proj_loc.weight"] = src_sd.get(
            f"{s_pre}self_attn.o_proj.weight"
        )

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
    print(
        f"🔒 凍結 Gemma 3 參數: {frozen_count:,} | 🟢 新架構可訓練參數: {trainable_count:,}"
    )

    del src_model
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    return agiv2_base

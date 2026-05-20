import sys, os
from transformers import AutoModelForCausalLM
import torch

sys.path.append("/mnt/e/antiplay/wisdomCoreV2")
from task03_CPT_gemma_1b.AGIV2G import AGIV2G
from task03_CPT_gemma_1b.train import transplant_and_freeze as tg

base_g = AGIV2G(vocab_size=262144, D=1152, hidden_dim=6912, num_blocks=26)
base_g = tg("google/gemma-3-1b-it", base_g)
trainable_g = sum(p.numel() for p in base_g.parameters() if p.requires_grad)

from task02_CPT_llama_8b.AGIV2L import AGIV2L
from task02_CPT_llama_8b.train import transplant_and_freeze as tl

base_l = AGIV2L(vocab_size=128256, D=4096, hidden_dim=14336, num_blocks=32)
base_l = tl("NousResearch/Meta-Llama-3-8B", base_l)
trainable_l = sum(p.numel() for p in base_l.parameters() if p.requires_grad)

print(f"Gemma 1B Trainable: {trainable_g}")
print(f"Llama 8B Trainable: {trainable_l}")

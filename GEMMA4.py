import torch
import torch.nn as nn
from transformers import Gemma4ForConditionalGeneration, AutoProcessor

class GEMMA4(Gemma4ForConditionalGeneration):
    """
    Gemma 4 Multimodal Model Class.
    Inherits from Gemma4ForConditionalGeneration to support all native Gemma 4 
    layers, weights, and high-performance inference mechanisms (e.g. FlashAttention-2, GQA).
    """
    @classmethod
    def load_model(cls, model_id="google/gemma-4-E2B-it", device_map="auto", torch_dtype=torch.bfloat16, **kwargs):
        """
        Loads the pre-trained Gemma 4 weights and returns the model.
        """
        print(f"Loading native Gemma 4 weights from {model_id}...")
        model = cls.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=torch_dtype,
            **kwargs
        )
        return model

class GEMMA4Processor:
    """
    Gemma 4 Processor wrapper.
    Manages tokenization, chat templates, and formatting for text, image, and audio inputs.
    """
    def __init__(self, model_id="google/gemma-4-E2B-it"):
        self.processor = AutoProcessor.from_pretrained(model_id)
        
    def apply_chat_template(self, messages, add_generation_prompt=True):
        """
        Formated conversation template into standard Gemma 4 chat formats.
        """
        return self.processor.apply_chat_template(messages, add_generation_prompt=add_generation_prompt)
        
    def __call__(self, text=None, images=None, audio=None, videos=None, **kwargs):
        """
        Preprocesses inputs and returns them as PyTorch tensors.
        """
        return self.processor(text=text, images=images, audio=audio, videos=videos, return_tensors="pt", **kwargs)
        
    def decode(self, *args, **kwargs):
        """
        Decodes output token IDs back to human-readable strings.
        """
        return self.processor.decode(*args, **kwargs)

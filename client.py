import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from compel import Compel, ReturnedEmbeddingsType
import time
import os


class DiffusersClient:
    """
    Diffusers Client
    """

    def __init__(self, model_file_path, device="cuda", cache_dir="./cache"):
        print("="*30)
        print("Loading Pipeline...")
        print("="*30)
        start = time.time()
        self.model_file_path = model_file_path
        self.device = device
        self.cache_dir = cache_dir
        self.pipe = StableDiffusionXLPipeline.from_single_file(
            model_file_path,
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir=self.cache_dir,
            local_files_only=True,
            load_safety_checker=False
        ).to(self.device)

        # compile
        # ref: https://torch.classcat.com/2023/11/03/huggingface-blog-simple-sdxl-optimizations/
        #self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=True)

        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe.enable_model_cpu_offload()
        #self.pipe.enable_sequential_cpu_offload()
        self.pipe.scheduler.use_karras_sigmas = False  # Karras

        # lora paths initialization
        self.lora_info = None

        print("="*30)
        print(f"◆◆◆Loding time: {time.time() - start} [sec]")
        print("="*30)


    def get_pipe(self):
        return self.pipe


    def generate(self, prompt, ng_prompt, width, height, steps=25, scale=5, seed=1, clip_skip=2, lora_info=None):
        # unloading
        if lora_info is None and self.lora_info != None:
            self.pipe.unload_lora_weights

        if not lora_info is None:
            self.lora_info = lora_info
            self.pipe.load_lora_weights(".", lora_info["path"], adapter_name="adapter")
            self.pipe.set_adapters(["adapter"], adapter_weights=[lora_info["weights"]])

        compel = Compel(
            tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2],
            text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True]
        )
        conditioning, pooled = compel(prompt)
        ng_conditioning, ng_pooled = compel(ng_prompt)
        generator = [torch.Generator().manual_seed(seed) for _ in range(len(prompt))]

        # inference
        start = time.time()
        print("="*30)
        print("Inferencing Start...")
        print("="*30)
        imgs = self.pipe(
            prompt_embeds=conditioning,
            pooled_prompt_embeds=pooled,
            negative_prompt_embeds=ng_conditioning,
            negative_pooled_prompt_embeds=ng_pooled,
            generator=generator,
            num_inference_steps=steps, 
            width=width, 
            height=height,
            guidance_scale=scale,
            clip_skip=clip_skip,
        ).images
        print("="*30)
        print(f"Generating Time: {time.time() - start} [sec]")
        print("="*30)

        return imgs

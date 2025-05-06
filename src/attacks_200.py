import os
import textwrap

import torch

def report_mem(tag=""):
    alloc = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"[{tag}] Allocated: {alloc:.1f} MB | Reserved: {reserved:.1f} MB")

report_mem()

import torch
device = 'cuda' if torch.cuda.is_available() else 'mps'
from torchvision.transforms.functional import to_pil_image
def get_processor_stats(processor, device='cuda'):
    mean = torch.tensor(processor.image_processor.image_mean, device=device).view(1, 3, 1, 1)
    std  = torch.tensor(processor.image_processor.image_std, device=device).view(1, 3, 1, 1)
    return mean, std
    
def get_normalized_bounds(processor, device='cuda'):
    Mean, Std  = get_processor_stats(processor, device)
    pixel_min = (0 - Mean)/ Std
    pixel_max = (1 - Mean)/ Std
    return pixel_min, pixel_max
    
def unnormalize_pixel_values(tensor, mean, std):
    return tensor * std + mean

def normalized_tensor_to_pil(tensor, processor):
    device = tensor.device
    mean, std = get_processor_stats(processor, device)
    unnorm = unnormalize_pixel_values(tensor, mean, std) * 255
    # print(unnorm)
    unnorm = torch.clamp(unnorm, 0, 255).byte()   # rescale to [0, 1] for visualization
    # print(unnorm)
    pil_img = to_pil_image(unnorm.squeeze(0).cpu())  # [1, 3, H, W] -> [3, H, W]
    return pil_img

# def prepare_target(caption, tokenizer, max_length, device="cuda"):
#     encoded = tokenizer(
#         caption,
#         padding="max_length",
#         truncation=True,
#         max_length=max_length,
#         return_tensors="pt"
#     )
#     return encoded.input_ids.to(device)


helper = {
    "en": "You are an assistant that captions images. Provide a short and accurate caption for the image without any additional text or prefixes.",
    "bn": "আপনি এমন একজন সহকারী যিনি ছবির জন্য ক্যাপশন লেখেন। ছবিটির জন্য একটি সংক্ষিপ্ত ও সঠিক ক্যাপশন দিন, কোনও অতিরিক্ত টেক্সট বা প্রিফিক্স ছাড়াই।",
    "de": "Du bist ein Assistent, der Bilder beschriftet. Gib eine kurze und präzise Bildunterschrift für das Bild ohne zusätzlichen Text oder Präfixe.",
    "ko": "당신은 이미지에 캡션을 다는 도우미입니다. 추가 텍스트나 접두사 없이 짧고 정확한 캡션을 작성하세요.",
    "ru": "Вы помощник, создающий подписи к изображениям. Дайте короткую и точную подпись к изображению без дополнительного текста или префиксов.",
    "zh": "你是一位为图像添加标题的助手。请提供简短且准确的标题，不加任何额外的文本或前缀。"
}


PROMPT = {
    "en": "Caption the image in a short, factual sentence.",
    "bn": "ছবিটির জন্য একটি সংক্ষিপ্ত ও তথ্যভিত্তিক বাক্যে ক্যাপশন দিন।",
    "de": "Beschrifte das Bild mit einem kurzen, sachlichen Satz.",
    "ko": "이미지를 짧고 사실적인 문장으로 설명하세요.",
    "ru": "Подпишите изображение коротким, фактическим предложением.",
    "zh": "请用简短且客观的句子为图片添加标题。"
}

import matplotlib.pyplot as plt


def show_attack_results(processor, original, adv, title=None):
    mean, std = get_processor_stats(processor, device=original.device)
    adv = unnormalize_pixel_values(adv, mean, std)
    original = unnormalize_pixel_values(original, mean, std)

    perturbation = adv - original
    pert_rescaled = (perturbation - perturbation.min()) / (perturbation.max() - perturbation.min() + 1e-8)  # normalize to [0,1]

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(original.squeeze().permute(1, 2, 0).cpu())
    axs[0].set_title('Original Image')
    axs[1].imshow(adv.squeeze().permute(1, 2, 0).cpu())
    axs[1].set_title('Adversarial Image')
    axs[2].imshow(pert_rescaled.squeeze().permute(1, 2, 0).cpu())
    axs[2].set_title('Perturbation (normalized)')
    
    for ax in axs: ax.axis('off')
    if title: fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def show_caption_diff(model, processor, image, adv_image, lang='en'):
    messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": helper[lang]}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": PROMPT[lang]}
                    ]
                }
            ]
    inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]
    # Generate output from model
    with torch.inference_mode():
        generation = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False
        )
        generation = generation[0][input_len:]
    
    text_clean = processor.decode(generation, skip_special_tokens=True)
    print(text_clean)


    adv_messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": helper[lang]}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": adv_image},
                        {"type": "text", "text": PROMPT[lang]}
                    ]
                }
            ]

    

    adv_inputs = processor.apply_chat_template(
                adv_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(model.device, dtype=torch.bfloat16)

    adv_input_len = inputs["input_ids"].shape[-1]

    # Generate output from model
    with torch.inference_mode():
        adv_generation = model.generate(
            **adv_inputs,
            max_new_tokens=100,
            do_sample=False
        )
        adv_generation = adv_generation[0][adv_input_len:]

    
    text_adv = processor.decode(adv_generation, skip_special_tokens=True)

    # print(" Original Caption:", text_clean)
    # print(" Adversarial Caption:", text_adv)
    return text_clean, text_adv

# def show_vqa_diff(model, processor, image, adv_image, question):
#     print(question)
#     inputs = processor(images=image, text=question, return_tensors="pt").to(model.device)
#     adv_inputs = processor(images=adv_image, text=question, return_tensors="pt").to(model.device)

#     with torch.no_grad():
#         ans_clean = model.generate(**inputs)
#         ans_adv = model.generate(**adv_inputs)

#     # text_clean = processor.tokenizer.decode(ans_clean[0], skip_special_tokens=True)
#     # text_adv = processor.tokenizer.decode(ans_adv[0], skip_special_tokens=True)
#     text_clean = processor.decode(ans_clean[0], skip_special_tokens=True)
#     text_adv = processor.decode(ans_adv[0], skip_special_tokens=True)

#     return text_clean, text_adv


# Basic modular implementations of adversarial attack components for VLM tasks (e.g., captioning, VQA)

import torch
import torch.nn.functional as F
from enum import Enum
import random

# -----------------------------------
# Norm Type Enum
# -----------------------------------
class NormType(Enum):
    Linf = "Linf"
    L2 = "L2"

# -----------------------------------
# Norm-based Projection
# -----------------------------------
def project(delta, epsilon, norm_type):
    if norm_type == NormType.Linf:
        return torch.clamp(delta, -epsilon, epsilon)
    elif norm_type == NormType.L2:
        norm = torch.norm(delta.view(delta.size(0), -1), dim=1, keepdim=True).view(-1, 1, 1, 1)
        factor = torch.min(torch.ones_like(norm), epsilon / (norm + 1e-10))
        return delta * factor

# -----------------------------------
# Gradient Normalization
# -----------------------------------
def normalize_grad(grad, norm_type):
    if norm_type == NormType.Linf:
        return grad.sign()
    elif norm_type == NormType.L2:
        norm = torch.norm(grad.view(grad.size(0), -1), dim=1, keepdim=True).view(-1, 1, 1, 1)
        return grad / (norm + 1e-10)

# -----------------------------------
# Random Initialization of Perturbation
# -----------------------------------
def random_init(image, epsilon, norm_type, pixel_min=-1, pixel_max=1):
    # TODO: (is it really required). To have pixel_min and pixel_max initialised in the range of pixel_values
    
    delta = torch.zeros_like(image).uniform_(pixel_min, pixel_min)
    delta = project(delta, epsilon, norm_type)
    return delta.detach()


def pgd_attack(model, processor, image, input_text,  epsilon, alpha, steps, norm_type=NormType.Linf, random_start=True, vqa=False, lang='en'):
    # report_mem('Beginning of PGD Attack fn')
    messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": helper[lang]}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": PROMPT[lang]}
                    ]
                }
            ]
    
    inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(model.device, dtype=torch.bfloat16)

    original_pixel_values = inputs['pixel_values'].clone().detach()

    # Pertubation initialisation
    pixel_min, pixel_max = get_normalized_bounds(processor, device=original_pixel_values.device)
    delta = random_init(original_pixel_values, epsilon, norm_type) if random_start else torch.zeros_like(original_pixel_values)
    delta.requires_grad_(True)

    
    
    for _ in range(steps):
        # print(_)
        adv_pixel_values = original_pixel_values + delta
        
        new_inputs = {**inputs, 'pixel_values': adv_pixel_values}
        

        # report_mem('before forward')
        outputs = model(**new_inputs)
        # report_mem('after forward')

        logits = outputs.logits

        loss = -logits.mean() # untargeted attack

        model.zero_grad()
        loss.backward()
        # report_mem('after backward calculation')

        grad = delta.grad.detach()
        delta.data = delta + alpha * normalize_grad(grad, norm_type)
        delta.data = project(delta.data, epsilon, norm_type)

        delta.grad.zero_()
        del logits, loss, outputs, grad
        torch.cuda.empty_cache()
        # report_mem('End of loop')

        

    adv_pixel_values = torch.clamp(original_pixel_values + delta.data,
                                   min=pixel_min, max=pixel_max)
    
    return original_pixel_values, adv_pixel_values.detach()#, delta.detach()
        


def mim_attack(model, processor, image, input_text,  epsilon, alpha, steps, norm_type=NormType.Linf, mu=1.0, random_start=False, vqa=False, lang='en'):
    messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": helper[lang]}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": PROMPT[lang]}
                    ]
                }
            ]
    
    inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(model.device, dtype=torch.bfloat16)

    original_pixel_values = inputs['pixel_values'].clone().detach()
    delta = random_init(original_pixel_values, epsilon, norm_type) if random_start else torch.zeros_like(original_pixel_values)
    delta.requires_grad_(True)
    momentum = torch.zeros_like(original_pixel_values)

    pixel_min, pixel_max = get_normalized_bounds(processor, device=original_pixel_values.device)

    for _ in range(steps):
        adv_pixel_values = original_pixel_values + delta
        new_inputs = {**inputs, 'pixel_values': adv_pixel_values}
            
        outputs = model(**new_inputs)
        logits = outputs.logits

        loss = -logits.mean()  # untargeted loss

        model.zero_grad()
        loss.backward()

        grad = delta.grad.detach()
        grad = normalize_grad(grad, norm_type)

        # Update momentum and apply it
        momentum = mu * momentum + grad
        delta.data = delta + alpha * normalize_grad(momentum, norm_type)

        # Project and clip
        delta.data = project(delta.data, epsilon, norm_type)
        # delta.data = torch.clamp(original_pixel_values + delta.data, min=pixel_min, max=pixel_max) - original_pixel_values

        delta.grad.zero_()
        del logits, loss, outputs, grad
        torch.cuda.empty_cache()

    adv_pixel_values = torch.clamp(original_pixel_values + delta.data, min=pixel_min, max=pixel_max)
    return original_pixel_values, adv_pixel_values.detach()

# Create random size augmention for diverse input attack
def input_diversity(image, resize_rate=1.1, diversity_prob=0.5):
    if torch.rand(1).item() > diversity_prob:
        return image
    img_size = image.shape[-1]
    new_size = random.randint(int(img_size), int(img_size * resize_rate))
    image_resized = F.interpolate(image, size=(new_size, new_size), mode='bilinear', align_corners=False)
    pad_top = random.randint(0, int(img_size * resize_rate) - new_size)
    pad_bottom = int(img_size * resize_rate) - new_size - pad_top
    pad_left = random.randint(0, int(img_size * resize_rate) - new_size)
    pad_right = int(img_size * resize_rate) - new_size - pad_left
    image_padded = F.pad(image_resized, (pad_left, pad_right, pad_top, pad_bottom), value=0)
    return F.interpolate(image_padded, size=(img_size, img_size), mode='bilinear', align_corners=False)


def dim_attack(model, processor, image, input_text,  epsilon, alpha, steps, norm_type=NormType.Linf, resize_rate=1.1, diversity_prob=0.5,  random_start=False, vqa=False, lang='en'):
    messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": helper[lang]}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": PROMPT[lang]}
                    ]
                }
            ]
    
    inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(model.device, dtype=torch.bfloat16)

    original_pixel_values = inputs['pixel_values'].clone().detach()
    delta = random_init(original_pixel_values, epsilon, norm_type) if random_start else torch.zeros_like(original_pixel_values)
    delta.requires_grad_(True)

    pixel_min, pixel_max = get_normalized_bounds(processor, device=original_pixel_values.device)

    for _ in range(steps):
        adv_pixel_values = original_pixel_values + delta
        transformed = input_diversity(adv_pixel_values, resize_rate, diversity_prob)

        new_inputs = {**inputs, 'pixel_values': transformed}

        outputs = model(**new_inputs)
        logits = outputs.logits

        loss = -logits.mean()  # you can swap this with a proper loss_fn if needed

        model.zero_grad()
        loss.backward()

        grad = delta.grad.detach()
        delta.data = delta + alpha * normalize_grad(grad, norm_type)
        delta.data = project(delta.data, epsilon, norm_type)
        # delta.data = torch.clamp(original_pixel_values + delta.data, min=pixel_min, max=pixel_max) - original_pixel_values

        delta.grad.zero_()
        del logits, loss, outputs, grad
        torch.cuda.empty_cache()

    adv_pixel_values = torch.clamp(original_pixel_values + delta.data, min=pixel_min, max=pixel_max)
    return original_pixel_values, adv_pixel_values.detach()


# def get_admix_data(image_dir, image_list, processor, device, num_admix=20):
#     # Randomly sample a subset of images
#     sampled_filenames = random.sample(image_list, num_admix)

#     image_batch = []
#     for fname in sampled_filenames:
#         path = os.path.join(image_dir, fname)
#         img = Image.open(path).convert('RGB')
#         image_batch.append(img)

#     # Use processor to get pixel_values
#     dummy_text = "What is in the picture?"  # or any fixed prompt
#     inputs = processor(images=image_batch, text=[dummy_text] * num_admix, return_tensors="pt").to(device)

#     return inputs['pixel_values']  # shape: (num_admix, 3, 224, 224)

def get_admix_data(hf_dataset, processor, device, num_admix=20):
    # Randomly sample indices
    sampled_indices = random.sample(range(len(hf_dataset)), num_admix)
    
    image_batch = []
    for idx in sampled_indices:
        sample = hf_dataset[idx]
        img = sample["image"]
        
        # # Ensure image is decoded
        # if hasattr(img, "decode"):
        #     img = img.decode()
        
        image_batch.append(img)

    dummy_text = "What is in the picture?"
    inputs = processor(images=image_batch, text=[dummy_text] * num_admix, return_tensors="pt").to(device)

    return inputs["pixel_values"] 
    
def admix_attack(model, processor, image, input_text,  epsilon, alpha, steps, norm_type=NormType.Linf,
                 admix_data=None, portion=0.2, repeat=3, random_start=False, vqa=False, lang='en'):
    
    messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": helper[lang]}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": PROMPT[lang]}
                    ]
                }
            ]
    
    inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(model.device, dtype=torch.bfloat16)

    original_pixel_values = inputs['pixel_values'].clone().detach()  # shape: (1, 3, 224, 224)
    delta = random_init(original_pixel_values, epsilon, norm_type) if random_start else torch.zeros_like(original_pixel_values)
    delta.requires_grad_(True)

    pixel_min, pixel_max = get_normalized_bounds(processor, device=original_pixel_values.device)
    B = original_pixel_values.shape[0]  # should be 1

    for _ in range(steps):
        mixed_batch = []
        for _ in range(repeat):
            idx = torch.randint(0, admix_data.shape[0], (1,))
            lam = 1.0 - portion  # equivalent to λ = 0.8 if portion = 0.2

            # Interpolated mix
            admix_img = admix_data[idx].to(original_pixel_values.device)
            x_adv = original_pixel_values + delta
            mixed = lam * x_adv + (1 - lam) * admix_img
            mixed_batch.append(mixed)

        mixed_pixel_values = torch.cat(mixed_batch, dim=0)  # (repeat, 3, 224, 224)

        repeated_inputs = processor(images=[image] * repeat, text=[input_text] * repeat, return_tensors="pt").to(device)
        repeated_inputs['pixel_values'] = mixed_pixel_values

        outputs = model(**repeated_inputs)
        logits = outputs.logits
        loss = -logits.mean()  # untargeted

        model.zero_grad()
        loss.backward()

        grad = delta.grad.detach()
        delta.data = delta + alpha * normalize_grad(grad, norm_type)
        delta.data = project(delta.data, epsilon, norm_type)
        # delta.data = torch.clamp(original_pixel_values + delta.data, min=pixel_min, max=pixel_max) - original_pixel_values

        del logits, loss, outputs, grad
        torch.cuda.empty_cache()
        delta.grad.zero_()

    adv_pixel_values = torch.clamp(original_pixel_values + delta.data, min=pixel_min, max=pixel_max)
    return original_pixel_values, adv_pixel_values.detach()

def si_attack(model, processor, image, input_text,  epsilon, alpha, steps, norm_type=NormType.Linf, scales=[1.0, 0.9, 0.8], random_start=False, vqa=False, lang='en'):
    messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": helper[lang]}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": PROMPT[lang]}
                    ]
                }
            ]
    
    inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(model.device, dtype=torch.bfloat16)

    original_pixel_values = inputs['pixel_values'].clone().detach()  # (1, 3, 224, 224)
    delta = random_init(original_pixel_values, epsilon, norm_type) if random_start else torch.zeros_like(original_pixel_values)
    delta.requires_grad_(True)

    pixel_min, pixel_max = get_normalized_bounds(processor, device=original_pixel_values.device)

    for _ in range(steps):
        grad = torch.zeros_like(original_pixel_values)

        for s in scales:
            # 1. Scale and resize back
            scaled = F.interpolate(original_pixel_values + delta, scale_factor=s, mode='bilinear', align_corners=False)
            rescaled = F.interpolate(scaled, size=original_pixel_values.shape[-2:], mode='bilinear', align_corners=False)

            # 2. Forward pass
            scaled_inputs = {**inputs, 'pixel_values': rescaled}
            
            outputs = model(**scaled_inputs)
            logits = outputs.logits

            # 3. Backward pass
            loss = -logits.mean()
            model.zero_grad()
            loss.backward()

            grad += delta.grad.detach()
            delta.grad.zero_()
            del logits, loss, outputs
        

        grad /= len(scales)
        delta.data = delta + alpha * normalize_grad(grad, norm_type)
        delta.data = project(delta.data, epsilon, norm_type)
        # delta.data = torch.clamp(original_pixel_values + delta.data, min=pixel_min, max=pixel_max) - original_pixel_values
        del logits, loss, outputs, grad
        torch.cuda.empty_cache()

    adv_pixel_values = torch.clamp(original_pixel_values + delta.data, min=pixel_min, max=pixel_max)
    return original_pixel_values, adv_pixel_values.detach()

import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt


# Step 1: Load Model + Processor
device = 'cuda' if torch.cuda.is_available() else 'mps'

from transformers import AutoProcessor, Gemma3ForConditionalGeneration
# Load Gemma model and processor
model_id = "google/gemma-3-4b-it"
model = Gemma3ForConditionalGeneration.from_pretrained(model_id).to(device).eval()
processor = AutoProcessor.from_pretrained(model_id)
report_mem('After loading model')

from sentence_transformers import SentenceTransformer, util
model_sim = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2").to(device)

def check_similarity(cap1, cap2, threshold=0.50):
    emb1 = model_sim.encode(cap1, convert_to_tensor=True)
    emb2 = model_sim.encode(cap2, convert_to_tensor=True)
    sim_score = util.cos_sim(emb1, emb2).item()
    return sim_score >= threshold, sim_score
report_mem('After loading sentence transformer')

from datasets import load_dataset

# Load vqa dataset
ds = load_dataset(
    "hwaseem04/Aya-testing",
    data_files={"xGQA_vqa": "data/xGQA_vqa-00000-of-00001.parquet"}
)

# Load caption dataset
ds2 = load_dataset(
    "hwaseem04/Aya-testing",
    data_files={"xm3600_captioning": "data/xm3600_captioning-00000-of-00001.parquet"}
)

def auto_attack_captioning(
    model,
    processor,
    image,
    input_text,
    epsilon,
    alpha,
    steps,
    attack_fn,
    attack_kwargs=None,
    random_start=False,
    max_attempts=4,
    similarity_threshold=0.50,
):
    attack_kwargs = attack_kwargs or {}
    for attempt in range(max_attempts):
        # Call any attack function with flexible arguments
        original_pixel_values, adv_pixel_values = attack_fn(
            model=model,
            processor=processor,
            image=image,
            input_text=input_text,
            epsilon=epsilon,
            alpha=alpha,
            steps=steps,
            **attack_kwargs
        )

        # Convert and evaluate
        adv_img = normalized_tensor_to_pil(adv_pixel_values, processor)
        orig_caption, adv_caption = show_caption_diff(model, processor, image, adv_img)
        similar, score = check_similarity(orig_caption, adv_caption, threshold=similarity_threshold)

        # print(f"[Attempt {attempt+1}] Similarity = {score:.4f}")
        if not similar:
            return original_pixel_values, adv_pixel_values, orig_caption, adv_caption, adv_img, 1

        print("Caption too similar. Retrying attack...")
        alpha += 0.01
        epsilon += 0.1
        
    return original_pixel_values, adv_pixel_values, orig_caption, adv_caption, adv_img, 0


import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

from tqdm import tqdm
# Languages to iterate over
languages = ["en"]#, "bn", "de", "ko", "ru", "zh"] # For now, learn adversarial image using 'en' language. Apply it on other language captions and see if it is transferable.
dataset = ds2['xm3600_captioning']


adv_results = [] 
sample_count = 0  

total_samples = len(dataset) if hasattr(dataset, '__len__') else 3600  # Estimate if needed
for sample in tqdm(dataset, desc="Iterating samples", total=total_samples):
    if sample_count >= 200:
        break
    try:
        new_data = {**sample}
        image = sample["image"]
        sample_id = sample["sample_id"]

        sample_count += 1
        # for lang in languages:
        lang = 'en'
        print('*'*20, f'Language: {lang}, sample_id: {sample_id}', '*'*20)
        prompt_col = f"prompt_{lang}"
        caption_col = f"captions_{lang}"

        input_text = sample[prompt_col]

        epsilon, alpha, steps = 1.0, 0.01, 20

        print('==== PGD Attack ====')
        original_pixel_values, adv_pixel_values, orig_caption, adv_caption, adv_img, success = auto_attack_captioning(
                                                                                    model, processor, image, input_text,
                                                                                    epsilon, alpha, steps,
                                                                                    attack_fn=pgd_attack,
                                                                                    attack_kwargs={'norm_type':NormType.Linf, 'random_start':False})
        print(" Original Caption:", orig_caption)
        print(" Adversarial Caption:", adv_caption)
        # show_attack_results(processor, original_pixel_values, adv_pixel_values)
        new_data['adv_image_en_pgd'] = adv_img
        new_data['pgd_success'] = success
        print(success)
        print('==== MIM Attack ====')
        original_pixel_values, adv_pixel_values, orig_caption, adv_caption, adv_img, success = auto_attack_captioning(
                                                                                    model, processor, image, input_text,
                                                                                    epsilon, alpha, steps,
                                                                                    attack_fn=mim_attack,
                                                                                    attack_kwargs={'norm_type':NormType.Linf, 'random_start':False, 'mu':1.0})
        
        print(" Original Caption:", orig_caption)
        print(" Adversarial Caption:", adv_caption)
        # show_attack_results(processor, original_pixel_values, adv_pixel_values)
        new_data['adv_image_en_mim'] = adv_img
        new_data['mim_success'] = success
        print(success)
        print('==== DIM Attack ====')
        original_pixel_values, adv_pixel_values, orig_caption, adv_caption, adv_img, success = auto_attack_captioning(
                                                                                    model, processor, image, input_text,
                                                                                    epsilon, alpha, steps,
                                                                                    attack_fn=dim_attack,
                                                                                    attack_kwargs={'norm_type':NormType.Linf, 'random_start':False, 'resize_rate':1.1, 'diversity_prob':0.5})
        
        print(" Original Caption:", orig_caption)
        print(" Adversarial Caption:", adv_caption)
        # show_attack_results(processor, original_pixel_values, adv_pixel_values)
        new_data['adv_image_en_dim'] = adv_img
        new_data['dim_success'] = success
        print(success)
        print('\n\n')
        # break 

        adv_results.append(new_data)

        # if sample_id > 10:
        #     break
    except Exception as E:
        print(E, sample_id)
    
from datasets import Dataset

new_dataset = Dataset.from_list(adv_results)


new_dataset.save_to_disk('adv_dataset')
import numpy as np
import torch
from PIL import Image
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from transformers import AutoProcessor
from scripts.visualize import visualize_attention 

# Load Processor & VLA
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
config = OpenVLAConfig.from_pretrained("openvla/openvla-7b")

vla = OpenVLAForActionPrediction.from_pretrained(
    "openvla/openvla-7b",
    config=config,
    attn_implementation="sdpa",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).to("cuda:0")


# Grab image input & format prompt
image = np.random.rand(3, 224, 224)
random_array = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
image = Image.fromarray(random_array)

prompt = "In: What action should the robot take to pick the red object?\nOut:"

# Predict Action (7-DoF; un-normalize for BridgeData V2)
inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
outputs = vla(
    **inputs,
    output_attentions=True,  # Enable attention output
    return_dict=True,  # Ensure we get a dictionary output
    save_attention=True,  # Save attention weights
)
attention_rollout = vla.attention_rollout(outputs)
attention_rollout = attention_rollout.cpu()
#image_token_indices = processor.get_image_token_indices(inputs["input_ids"])
#breakpoint()  # This will pause execution and drop you into a debugging console

#action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

image_token_indices = list(range(1, 14*14+1))
sequence_length = attention_rollout.shape[-1]
action_token_indices = list(range(sequence_length-7, sequence_length))

visualize_attention(
    attention_rollout,
    image,
    image_token_indices,
    action_token_indices,
    title="Attention Rollout",
)
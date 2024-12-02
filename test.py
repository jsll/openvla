import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

# Load Processor & VLA
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    # attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    attn_implementation="sdpa",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).to("cuda:0")

# Grab image input & format prompt
image = np.random.rand(3, 224, 224)
random_array = np.random.randint(0, 256, size=(244, 244, 3), dtype=np.uint8)
image = Image.fromarray(random_array)

prompt = "In: What action should the robot take to pick the red object?\nOut:"

# Predict Action (7-DoF; un-normalize for BridgeData V2)
inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
outputs = vla(
    **inputs,
    output_attentions=True,  # Enable attention output
    return_dict=True,  # Ensure we get a dictionary output
)
print(type(outputs.attentions))
breakpoint()  # This will pause execution and drop you into a debugging console


action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

# print(action)

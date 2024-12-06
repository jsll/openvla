import argparse
import io
import itertools

import datasets
import torch
from PIL import Image
from transformers import AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from scripts.visualize import visualize_attention


def test(prompt):
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

    dataset_names = ["stanford_hydra_dataset_converted_externally_to_rlds"]
    data_per_dataset = {}
    for dataset_name in dataset_names:
        ds = datasets.load_dataset(
            "jxu124/OpenX-Embodiment",
            dataset_name,
            streaming=True,
            split="train",
            trust_remote_code=True,
        )  # IterDataset
        random_item = next(itertools.islice(ds, 10, 10 + 1))
        data_per_dataset[dataset_name] = random_item
    # Grab image input & format prompt
    data = data_per_dataset["stanford_hydra_dataset_converted_externally_to_rlds"]
    image = Image.open(io.BytesIO(data["data.pickle"]["steps"][0]["observation"]["image"]["bytes"]))
    image = image.resize((224, 224))

    # image = np.random.rand(3, 224, 224)
    # random_array = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
    # image = Image.fromarray(random_array)

    # prompt = "In: What action should the robot take to open the oven?\nOut:"

    # Predict Action (7-DoF; un-normalize for BridgeData V2)
    last_attention = None
    while True:
        prompt = input("In: What action should the robot take to ")
        if prompt == "q":
            break
        prompt = f"In: What action should the robot take to {prompt}?\nOut:"

        print(prompt)
        inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
        # outputs = vla(
        #    **inputs,
        #    output_attentions=True,  # Enable attention output
        #    return_dict=True,  # Ensure we get a dictionary output
        #    save_attention=True,  # Save attention weights
        # )
        # attention_matrices = vla.process_attention_matrices(outputs)
        # print(attention_matrices["self_attention"].shape)
        # attention_rollout = vla.attention_rollout(attention_matrices["self_attention"])
        # attention_rollout = attention_rollout.cpu()
        # if last_attention is None:
        #    last_attention = attention_rollout

        # print(attention_rollout.shape)
        # image_token_indices = processor.get_image_token_indices(inputs["input_ids"])
        # breakpoint()  # This will pause execution and drop you into a debugging console
        # action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

        # print("Action 1: ", action)
        action, attention_data = vla.predict_action_with_attention(
            **inputs, unnorm_key="bridge_orig", do_sample=False, head_fusion="mean"
        )

        print("Action: ", action)
        attention_rollout = attention_data["attention_rollout"].cpu()
        if last_attention is None:
            last_attention = attention_rollout

        # print(attention_data["attention_rollout"].shape)
        # breakpoint()
        image_token_indices = list(range(1, 14 * 14 + 1))
        sequence_length = attention_rollout.shape[-1]
        action_token_indices = list(range(sequence_length - 7, sequence_length))
        # print(image_token_indices)
        # print(action_token_indices)
        diff = 0
        print(action_token_indices)
        print(attention_rollout.shape)
        for action_token_index in action_token_indices[:-1]:
            # diff += (
            #    attention_rollout[0, action_token_index, image_token_indices]
            #    - last_attention[0, action_token_index, image_token_indices]
            # ).sum()
            print(action_token_index)
            diff += (
                attention_rollout[0, action_token_index, image_token_indices]
                - attention_rollout[0, action_token_index + 1, image_token_indices]
            ).sum()

            # print(attention_rollout[0, action_token_index, image_token_indices])
            # break
        print("Diff: ", diff)
        last_attention = attention_rollout

        visualize_attention(
            attention_rollout,
            image,
            image_token_indices,
            action_token_indices,
            title=prompt,
        )


if __name__ == "__main__":
    # Make an argparser that takes as input the possible prompt to openvla
    # and the image to be used
    args = None
    arpgarse = argparse.ArgumentParser()
    arpgarse.add_argument("--prompt", type=str, default="In: What action should the robot take to open the oven?\nOut:")
    args = arpgarse.parse_args()
    if "In:" not in args.prompt:
        args.prompt = f"In: {args.prompt}"
    if "Out:" not in args.prompt:
        args.prompt = f"{args.prompt}\nOut:"

    test(args.prompt)

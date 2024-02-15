"""Script to run end-to-end evaluation on the benchmark.

Modified from https://github.com/web-arena-x/webarena/blob/main/run.py.
"""
import argparse
import glob
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import List
import ipdb
import openai
import requests
import torch
from beartype import beartype
from PIL import Image
import numpy as np
from agent import (
    PromptAgent,
    construct_agent,
)
from agent.prompts import *
from browser_env import (
    Action,
    ActionTypes,
    ScriptBrowserEnv,
    StateInfo,
    Trajectory,
    create_stop_action,
)
from browser_env.actions import is_equivalent
from browser_env.helper_functions import (
    RenderHelper,
    get_action_description,
)
from evaluation_harness import evaluator_router, image_utils

LOG_FOLDER = "log_files"
Path(LOG_FOLDER).mkdir(parents=True, exist_ok=True)
LOG_FILE_NAME = f"{LOG_FOLDER}/log_{time.strftime('%Y%m%d%H%M%S', time.localtime())}_{random.randint(0, 10000)}.log"

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(LOG_FILE_NAME)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

# Set the log format
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
humanEvals = ['config_files/test_shopping/209.json', 'config_files/test_reddit/106.json', 'config_files/test_classifieds/186.json', 'config_files/test_classifieds/204.json', 'config_files/test_classifieds/113.json', 'config_files/test_classifieds/91.json', 'config_files/test_classifieds/135.json', 'config_files/test_classifieds/70.json', 'config_files/test_classifieds/144.json', 'config_files/test_classifieds/15.json', 'config_files/test_classifieds/221.json', 'config_files/test_classifieds/128.json', 'config_files/test_classifieds/182.json', 'config_files/test_classifieds/104.json', 'config_files/test_classifieds/57.json', 'config_files/test_classifieds/211.json', 'config_files/test_classifieds/83.json', 'config_files/test_classifieds/47.json', 'config_files/test_classifieds/189.json', 'config_files/test_classifieds/212.json', 'config_files/test_classifieds/100.json', 'config_files/test_classifieds/167.json', 'config_files/test_classifieds/4.json', 'config_files/test_classifieds/27.json', 'config_files/test_classifieds/31.json', 'config_files/test_classifieds/55.json', 'config_files/test_classifieds/169.json', 'config_files/test_classifieds/51.json', 'config_files/test_classifieds/85.json', 'config_files/test_classifieds/43.json', 'config_files/test_classifieds/177.json', 'config_files/test_classifieds/71.json', 'config_files/test_classifieds/14.json', 'config_files/test_classifieds/87.json', 'config_files/test_classifieds/34.json', 'config_files/test_classifieds/8.json', 'config_files/test_classifieds/7.json', 'config_files/test_classifieds/179.json', 'config_files/test_classifieds/163.json', 'config_files/test_classifieds/36.json', 'config_files/test_classifieds/116.json', 'config_files/test_classifieds/11.json', 'config_files/test_classifieds/184.json', 'config_files/test_classifieds/78.json', 'config_files/test_classifieds/110.json', 'config_files/test_classifieds/22.json', 'config_files/test_classifieds/21.json', 'config_files/test_classifieds/151.json', 'config_files/test_classifieds/226.json', 'config_files/test_classifieds/123.json', 'config_files/test_classifieds/170.json', 'config_files/test_classifieds/140.json', 'config_files/test_classifieds/66.json', 'config_files/test_classifieds/46.json', 'config_files/test_classifieds/202.json', 'config_files/test_classifieds/38.json', 'config_files/test_classifieds/162.json', 'config_files/test_classifieds/101.json', 'config_files/test_reddit/209.json', 'config_files/test_reddit/90.json', 'config_files/test_reddit/204.json', 'config_files/test_reddit/89.json', 'config_files/test_reddit/113.json', 'config_files/test_reddit/157.json', 'config_files/test_reddit/33.json', 'config_files/test_reddit/133.json', 'config_files/test_reddit/147.json', 'config_files/test_reddit/156.json', 'config_files/test_reddit/96.json', 'config_files/test_reddit/180.json', 'config_files/test_reddit/201.json', 'config_files/test_reddit/128.json', 'config_files/test_reddit/2.json', 'config_files/test_reddit/182.json', 'config_files/test_reddit/205.json', 'config_files/test_reddit/106.json', 'config_files/test_reddit/28.json', 'config_files/test_reddit/122.json', 'config_files/test_reddit/178.json', 'config_files/test_reddit/199.json', 'config_files/test_reddit/174.json', 'config_files/test_reddit/189.json', 'config_files/test_reddit/17.json', 'config_files/test_reddit/167.json', 'config_files/test_reddit/130.json', 'config_files/test_reddit/119.json', 'config_files/test_reddit/73.json', 'config_files/test_reddit/55.json', 'config_files/test_reddit/169.json', 'config_files/test_reddit/143.json', 'config_files/test_reddit/49.json', 'config_files/test_reddit/207.json', 'config_files/test_reddit/71.json', 'config_files/test_reddit/14.json', 'config_files/test_reddit/34.json', 'config_files/test_reddit/103.json', 'config_files/test_reddit/41.json', 'config_files/test_reddit/179.json', 'config_files/test_reddit/88.json', 'config_files/test_reddit/188.json', 'config_files/test_reddit/193.json', 'config_files/test_reddit/116.json', 'config_files/test_reddit/164.json', 'config_files/test_reddit/80.json', 'config_files/test_reddit/5.json', 'config_files/test_reddit/3.json', 'config_files/test_reddit/11.json', 'config_files/test_reddit/184.json', 'config_files/test_reddit/194.json', 'config_files/test_reddit/161.json', 'config_files/test_reddit/22.json', 'config_files/test_reddit/21.json', 'config_files/test_reddit/124.json', 'config_files/test_reddit/61.json', 'config_files/test_reddit/140.json', 'config_files/test_reddit/138.json', 'config_files/test_reddit/9.json', 'config_files/test_reddit/152.json', 'config_files/test_reddit/111.json', 'config_files/test_reddit/38.json', 'config_files/test_reddit/101.json', 'config_files/test_shopping/209.json', 'config_files/test_shopping/225.json', 'config_files/test_shopping/272.json', 'config_files/test_shopping/90.json', 'config_files/test_shopping/173.json', 'config_files/test_shopping/185.json', 'config_files/test_shopping/282.json', 'config_files/test_shopping/245.json', 'config_files/test_shopping/98.json', 'config_files/test_shopping/219.json', 'config_files/test_shopping/216.json', 'config_files/test_shopping/298.json', 'config_files/test_shopping/131.json', 'config_files/test_shopping/156.json', 'config_files/test_shopping/135.json', 'config_files/test_shopping/70.json', 'config_files/test_shopping/247.json', 'config_files/test_shopping/332.json', 'config_files/test_shopping/386.json', 'config_files/test_shopping/294.json', 'config_files/test_shopping/286.json', 'config_files/test_shopping/366.json', 'config_files/test_shopping/443.json', 'config_files/test_shopping/141.json', 'config_files/test_shopping/222.json', 'config_files/test_shopping/422.json', 'config_files/test_shopping/431.json', 'config_files/test_shopping/95.json', 'config_files/test_shopping/201.json', 'config_files/test_shopping/187.json', 'config_files/test_shopping/320.json', 'config_files/test_shopping/314.json', 'config_files/test_shopping/2.json', 'config_files/test_shopping/447.json', 'config_files/test_shopping/205.json', 'config_files/test_shopping/251.json', 'config_files/test_shopping/104.json', 'config_files/test_shopping/346.json', 'config_files/test_shopping/233.json', 'config_files/test_shopping/83.json', 'config_files/test_shopping/288.json', 'config_files/test_shopping/420.json', 'config_files/test_shopping/303.json', 'config_files/test_shopping/47.json', 'config_files/test_shopping/397.json', 'config_files/test_shopping/197.json', 'config_files/test_shopping/383.json', 'config_files/test_shopping/212.json', 'config_files/test_shopping/268.json', 'config_files/test_shopping/379.json', 'config_files/test_shopping/102.json', 'config_files/test_shopping/154.json', 'config_files/test_shopping/137.json', 'config_files/test_shopping/119.json', 'config_files/test_shopping/310.json', 'config_files/test_shopping/72.json', 'config_files/test_shopping/323.json', 'config_files/test_shopping/27.json', 'config_files/test_shopping/31.json', 'config_files/test_shopping/40.json', 'config_files/test_shopping/291.json', 'config_files/test_shopping/276.json', 'config_files/test_shopping/143.json', 'config_files/test_shopping/255.json', 'config_files/test_shopping/428.json', 'config_files/test_shopping/373.json', 'config_files/test_shopping/43.json', 'config_files/test_shopping/434.json', 'config_files/test_shopping/35.json', 'config_files/test_shopping/181.json', 'config_files/test_shopping/409.json', 'config_files/test_shopping/14.json', 'config_files/test_shopping/87.json', 'config_files/test_shopping/349.json', 'config_files/test_shopping/210.json', 'config_files/test_shopping/439.json', 'config_files/test_shopping/179.json', 'config_files/test_shopping/18.json', 'config_files/test_shopping/230.json', 'config_files/test_shopping/237.json', 'config_files/test_shopping/195.json', 'config_files/test_shopping/239.json', 'config_files/test_shopping/3.json', 'config_files/test_shopping/337.json', 'config_files/test_shopping/328.json', 'config_files/test_shopping/451.json', 'config_files/test_shopping/11.json', 'config_files/test_shopping/463.json', 'config_files/test_shopping/148.json', 'config_files/test_shopping/6.json', 'config_files/test_shopping/436.json', 'config_files/test_shopping/64.json', 'config_files/test_shopping/265.json', 'config_files/test_shopping/150.json', 'config_files/test_shopping/37.json', 'config_files/test_shopping/21.json', 'config_files/test_shopping/361.json', 'config_files/test_shopping/258.json', 'config_files/test_shopping/297.json', 'config_files/test_shopping/166.json', 'config_files/test_shopping/343.json', 'config_files/test_shopping/123.json', 'config_files/test_shopping/444.json', 'config_files/test_shopping/326.json', 'config_files/test_shopping/140.json', 'config_files/test_shopping/377.json', 'config_files/test_shopping/461.json', 'config_files/test_shopping/306.json', 'config_files/test_shopping/111.json', 'config_files/test_shopping/370.json', 'config_files/test_shopping/162.json', 'config_files/test_shopping/261.json', 'config_files/test_shopping/26.json', 'config_files/test_shopping/453.json']
evalMode = "Human"
def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )
    parser.add_argument(
        "--render", action="store_true", help="Render the browser"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Debug mode"
    )

    parser.add_argument(
        "--slow_mo",
        type=int,
        default=0,
        help="Slow down the browser by the specified amount",
    )
    parser.add_argument(
        "--action_set_tag", default="id_accessibility_tree", help="Action type"
    )
    parser.add_argument(
        "--observation_type",
        choices=[
            "accessibility_tree",
            "accessibility_tree_with_captioner",
            "html",
            "image",
            "image_som",
        ],
        default="accessibility_tree",
        help="Observation type",
    )
    parser.add_argument(
        "--current_viewport_only",
        action="store_true",
        help="Only use the current viewport for the observation",
    )
    parser.add_argument("--viewport_width", type=int, default=1280)
    parser.add_argument("--viewport_height", type=int, default=2048)
    parser.add_argument("--save_trace_enabled", action="store_true")
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)

    parser.add_argument("--max_steps", type=int, default=30)

    # agent config
    parser.add_argument("--agent_type", type=str, default="prompt")
    parser.add_argument(
        "--instruction_path",
        type=str,
        default="agents/prompts/state_action_agent.json",
    )
    parser.add_argument(
        "--parsing_failure_th",
        help="When consecutive parsing failures exceed this threshold, the agent will terminate early.",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--repeating_action_failure_th",
        help="When consecutive repeated actions exceed this threshold, the agent will terminate early.",
        type=int,
        default=5,
    )

    parser.add_argument("--episode_in_try_except", action="store_true", default=False, help="Continue to next episode if assertion error occurs? ")
    parser.add_argument("--skip_if_finished", action="store_true", default=False, help="Continue to next episode if assertion error occurs? ")

    parser.add_argument("--test_config_base_dir", type=str)

    parser.add_argument(
        "--eval_captioning_model_device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run eval captioning model on. By default, runs it on CPU.",
    )
    parser.add_argument(
        "--eval_captioning_model",
        type=str,
        default="Salesforce/blip2-flan-t5-xl",
        choices=["Salesforce/blip2-flan-t5-xl"],
        help="Captioning backbone for VQA-type evals.",
    )
    parser.add_argument(
        "--captioning_model",
        type=str,
        default="Salesforce/blip2-flan-t5-xl",
        choices=["Salesforce/blip2-flan-t5-xl", "llava-hf/llava-1.5-7b-hf"],
        help="Captioning backbone for accessibility tree alt text.",
    )

    # lm config
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0613")
    parser.add_argument("--mode", type=str, default="chat")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--context_length", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=384)
    parser.add_argument("--stop_token", type=str, default=None)
    parser.add_argument(
        "--max_retry",
        type=int,
        help="max retry times to perform generations when parsing fails",
        default=1,
    )
    parser.add_argument(
        "--max_obs_length",
        type=int,
        help="when not zero, will truncate the observation to this length before feeding to the model",
        default=3840,
    )

    # example config
    parser.add_argument("--test_start_idx", type=int, default=0)
    parser.add_argument("--test_end_idx", type=int, default=910)

    # logging related
    parser.add_argument("--result_dir", type=str, default="")
    args = parser.parse_args()

    # check the whether the action space is compatible with the observation space
    if (
        args.action_set_tag == "id_accessibility_tree"
        and args.observation_type
        not in [
            "accessibility_tree",
            "accessibility_tree_with_captioner",
            "image_som",
        ]
    ):
        raise ValueError(
            f"Action type {args.action_set_tag} is incompatible with the observation type {args.observation_type}"
        )

    return args


@beartype
def early_stop(
    trajectory: Trajectory, max_steps: int, thresholds: dict[str, int]
) -> tuple[bool, str]:
    """Check whether need to stop early"""

    # reach the max step
    num_steps = (len(trajectory) - 1) / 2
    if num_steps >= max_steps:
        return True, f"Reach max steps {max_steps}"

    last_k_actions: list[Action]
    action_seq: list[Action]

    # Case: parsing failure for k times
    k = thresholds["parsing_failure"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    if len(last_k_actions) >= k:
        if all(
            [
                action["action_type"] == ActionTypes.NONE
                for action in last_k_actions
            ]
        ):
            return True, f"Failed to parse actions for {k} times"

    # Case: same action for k times
    k = thresholds["repeating_action"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    action_seq = trajectory[1::2]  # type: ignore[assignment]

    if len(action_seq) == 0:
        return False, ""

    last_action: Action = action_seq[-1]

    if last_action["action_type"] != ActionTypes.TYPE:
        if len(last_k_actions) >= k:
            if all(
                [
                    is_equivalent(action, last_action)
                    for action in last_k_actions
                ]
            ):
                return True, f"Same action for {k} times"

    else:
        # check the action sequence
        if (
            sum([is_equivalent(action, last_action) for action in action_seq])
            >= k
        ):
            return True, f"Same typing action for {k} times"

    return False, ""


@beartype
def test(
    args: argparse.Namespace,
    config_file_list: list[str]
) -> None:
    scores = []
    trajSOM = {}
    trajImages = {}
    trajActions = {}
    trajInputImages = {}
    max_steps = args.max_steps

    early_stop_thresholds = {
        "parsing_failure": args.parsing_failure_th,
        "repeating_action": args.repeating_action_failure_th,
    }

    if args.observation_type in [
        "accessibility_tree_with_captioner",
        "image_som",
    ]:
        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        caption_image_fn = image_utils.get_captioning_fn(
            device, dtype, args.captioning_model
        )
    else:
        caption_image_fn = None

    # Load a (possibly different) captioning model for running VQA evals.
    if (
        caption_image_fn
        and args.eval_captioning_model == args.captioning_model
    ):
        eval_caption_image_fn = caption_image_fn
    else:
        eval_caption_image_fn = image_utils.get_captioning_fn(
            args.eval_captioning_model_device,
            torch.float16
            if (
                torch.cuda.is_available()
                and args.eval_captioning_model_device == "cuda"
            )
            else torch.float32,
            args.eval_captioning_model,
        )

    agent = construct_agent(
        args,
        captioning_fn=caption_image_fn
        if args.observation_type == "accessibility_tree_with_captioner"
        else None,
    )  # NOTE: captioning_fn here is used for captioning input images.

    env = ScriptBrowserEnv(
        headless=not args.render,
        slow_mo=args.slow_mo,
        observation_type=args.observation_type,
        current_viewport_only=args.current_viewport_only,
        viewport_size={
            "width": args.viewport_width,
            "height": args.viewport_height,
        },
        save_trace_enabled=args.save_trace_enabled,
        sleep_after_execution=args.sleep_after_execution,
        # NOTE: captioning_fn here is used for LLM + captioning baselines.
        # This can be different from the captioning model used for evals.
        captioning_fn=caption_image_fn,
    )

    for config_file in config_file_list:
        actionList = []
        imageList = []
        somList = []
        input_images = []
        # try:
        if (1):
            render_helper = RenderHelper(
                config_file, args.result_dir, args.action_set_tag
            )

            # Load task.
            with open(config_file) as f:
                _c = json.load(f)
                intent = _c["intent"]
                task_id = _c["task_id"]
                image_paths = _c.get("image", None)
                images = []

                # Load input images for the task, if any.
                if image_paths is not None:
                    inputCount = 0
                    if isinstance(image_paths, str):
                        image_paths = [image_paths]
                    for image_path in image_paths:
                        if "“" in image_path and "”" in image_path:
                            image_path = image_path.replace("“", "").replace("”", "")
                        # Load image either from the web or from a local path.
                        if image_path.startswith("http"):
                            input_image = Image.open(requests.get(image_path, stream=True).raw)
                        else:
                            input_image = Image.open(image_path)
                        directory = f'output/{config_file}'
                        if not os.path.exists(directory):
                            os.makedirs(directory)
                            print(f"Directory '{directory}' created successfully.")
                        input_image.save(f'output/{config_file}/input_image{inputCount}.png')
                        inputCount += 1
                    print("IMAGE SAVED")
                    input_images.append(input_image)
                    trajInputImages[config_file] = input_images

            logger.info(f"[Config file]: {config_file}")
            logger.info(f"[Intent]: {intent}")

            agent.reset(config_file)
            trajectory: Trajectory = []
            obs, info = env.reset(options={"config_file": config_file})
            state_info: StateInfo = {"observation": obs, "info": info}
            trajectory.append(state_info)

            meta_data = {"action_history": ["None"]}
            count = 0
            while True:
                imageList.append(obs["image"])
                somList.append(obs["text"])
                early_stop_flag, stop_info = early_stop(
                    trajectory, max_steps, early_stop_thresholds
                )
                if evalMode != "Human":
                    if early_stop_flag:
                        action = create_stop_action(f"Early stop: {stop_info}")
                    else:
                        try:
                            action = agent.next_action(
                                trajectory,
                                intent,
                                images=images,
                                meta_data=meta_data,
                            )
                        except ValueError as e:
                         #get the error message
                            action = create_stop_action(f"ERROR: {str(e)}")
                else:
                    action = {
                        'action_type': None,
                        'coords': None,
                        'element_role': None,
                        'element_name': None,
                        'text': None,
                        'page_number': None,
                        'url': None,
                        'nth': None,
                        'pw_code': None,
                        'element_id': None,
                        'key_comb': None,
                        'direction': None,
                        'answer': None,
                        'raw_prediction': None
                    }
                    import matplotlib.pyplot as plt
        
                    image = np.float32(obs['image'][:,:,:3])
                    # Create a new figure
                    image = image.astype(np.uint8) 
                    plt.figure(1, (24,24)); plt.clf()
                    plt.rcParams['figure.dpi']=500
                    plt.rcParams['savefig.dpi']=500
                    plt.imshow(image)
                    directory = f'output/{config_file}'
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                        print(f"Directory '{directory}' created successfully.")
                    print(f"IMAGE {count} PLOTTING")
                    plt.savefig(f'output/{config_file}/image{count}.png')
                    print("IMAGE SAVED")
                    #plt.show()
                    #ipdb.set_trace()
                    print("You can choose from the following options: click, type, scroll, hover, go back, go forward, go to url, key press, stop")
                    actionType = input("Choose an Initial Action Type:")
                    actionType == actionType.lower()
                    obs_nodes_info = state_info['info']['observation_metadata']['image']['obs_nodes_info']
                    elementChosen = False
                    if actionType == 'click':
                        action["action_type"] = ActionTypes.CLICK
                        print("Choose a Set-of-Marks element on the page to click")
                        elementID = input("SOM Element:")
                        while elementChosen == False:
                            if elementID in obs_nodes_info.keys():
                                action["element_id"] = elementID
                                elementChosen = True
                            else:
                                print("Choose a new Set-of-Marks element on the page to click. The previous chosen element was invalid")
                                elementID = input("SOM Element")
                    elif actionType == 'type':
                        action["action_type"] = ActionTypes.TYPE
                        print("Choose a Set-of-Marks element on the page to click")
                        elementID = input("SOM Element:")
                        while elementChosen == False:
                            if elementID in obs_nodes_info.keys():
                                action["element_id"] = elementID
                                elementChosen = True
                            else:
                                print("Choose a new Set-of-Marks element on the page to click. The previous chosen element was invalid")
                                elementID = input("SOM Element")
                        print("Write the text to input in the element")
                        elementText = input("Text:")
                        action["text"] = elementText
                    elif actionType == "key press":
                        action["action_type"] = ActionTypes.KEY_PRESS
                        print("Type out the name of the key to press. For example, Enter for the enter button.")
                        action["key_comb"] = input("Key:")
                    elif actionType == "scroll":
                        directionChosen = False
                        action["action_type"] = ActionTypes.SCROLL
                        print("Choose what direction to scroll: down or up")
                        direction = input("Direction:")
                        while directionChosen == False:
                            if direction in ["down", "up"]:
                                action["direction"] = direction
                                directionChosen = True
                            else:
                                print("Your previous direction was not valid. Choose what direction to scroll: down or up")
                                direction = input("Direction:")
                    elif actionType == "go to url":
                        action["action_type"] == ActionTypes.GOTO_URL
                        action["url"] = input("enter the url you would like to go to")
                    elif actionType == "hover":
                        action["action_type"] = ActionTypes.HOVER
                        print("Choose a Set-of-Marks element on the page to click")
                        elementID = input("SOM Element:")
                        while elementChosen == False:
                            if elementID in obs_nodes_info.keys():
                                action["element_id"] = elementID
                                elementChosen = True
                            else:
                                print("Choose a new Set-of-Marks element on the page to click. The previous chosen element was invalid")
                                elementID = input("SOM Element")
                    #elif actionType == "go back":
                    #    action["action_type"] = ActionTypes.GO_BACK      
                    #elif actionType == "go forward":
                    #    action["action_type"] = ActionTypes.GO_FORWARD
                    elif actionType == "stop":
                        action["action_type"] = ActionTypes.STOP
                        print("Answer the prompt or give the reason for stopping. If you have arrived to the desired page, enter 'displayed'")
                        action["answer"] = input("Answer:") 
                    else:
                        input("Did not identify a possible action from your input. Please give another action.")           
                trajectory.append(action)
                actionList.append(action)
                #ipdb.set_trace()
                action_str = get_action_description(
                    action,
                    state_info["info"]["observation_metadata"],
                    action_set_tag=args.action_set_tag,
                    prompt_constructor=agent.prompt_constructor
                    if isinstance(agent, PromptAgent)
                    else None,
                )
                render_helper.render(
                    action, state_info, meta_data, args.render_screenshot
                )
                meta_data["action_history"].append(action_str)

                if action["action_type"] == ActionTypes.STOP:
                    break

                visualize_results = False
                if visualize_results:
                    print(action)
                    tag = 'gpt4v'
                    import matplotlib.pyplot as plt
                    rgb = rgb.astype(np.uint8) 
                    plt.figure(1, (24,24)); plt.clf()
                    plt.rcParams['figure.dpi']=500
                    plt.rcParams['savefig.dpi']=500
                    plt.imshow(rgb)
                    plt.title(f'{action_str}')
                    os.makedirs(f'output/{tag}', exist_ok=True)
                    plt.savefig(f'output/{tag}/image{count}.png')
                count += 1

                obs, _, terminated, _, info = env.step(action)
                state_info = {"observation": obs, "info": info}
                trajectory.append(state_info)

                if terminated:
                    # add a action place holder
                    trajectory.append(create_stop_action(""))
                    break

            trajActions[config_file] = actionList
            trajSOM[config_file] = somList
            trajImages[config_file] = imageList

            # NOTE: eval_caption_image_fn is used for running eval_vqa functions.
            evaluator = evaluator_router(
                config_file, captioning_fn=eval_caption_image_fn
            )
            score = evaluator(
                trajectory=trajectory,
                config_file=config_file,
                page=env.page,
                client=env.get_page_client(env.page),
            )

            scores.append(score)

            if score == 1:
                logger.info(f"[Result] (PASS) {config_file}")
            else:
                logger.info(f"[Result] (FAIL) {config_file}")

            if args.save_trace_enabled:
                env.save_trace(
                    Path(args.result_dir) / "traces" / f"{task_id}.zip"
                )
        # except openai.OpenAIError as e:
        #     if args.debug:
        #         raise
        #     logger.info(f"[OpenAI Error] {repr(e)}")
        # except Exception as e:
        #     if args.debug:
        #         raise
        #     logger.info(f"[Unhandled Error] {repr(e)}]")
        #     import traceback

        #     # write to error file
        #     with open(Path(args.result_dir) / "error.txt", "a") as f:
        #         f.write(f"[Config file]: {config_file}\n")
        #         f.write(f"[Unhandled Error] {repr(e)}\n")
        #         f.write(traceback.format_exc())  # write stack trace to file

        render_helper.close()
    import pickle
    ipdb.set_trace()
    # Assuming you have dictionaries `dict1`, `dict2`, and `dict3`
    # Write dictionary with numpy arrays to disk
    with open('states.pkl', 'wb') as f:
        pickle.dump(trajSOM, f)

    # Write dictionary with lists of strings to disk
    with open('images.pkl', 'wb') as f:
        pickle.dump(trajImages, f)

    # Write dictionary with lists of dictionaries to disk
    with open('actions.pkl', 'wb') as f:
        pickle.dump(trajActions, f)
    
    with open('input_images.pkl', 'wb') as f:
        pickle.dump(input_images, f)

    env.close()
    logger.info(f"Average score: {sum(scores) / len(scores)}")


def prepare(args: argparse.Namespace) -> None:
    # convert prompt python files to json
    from agent.prompts import to_json

    to_json.run()

    # prepare result dir
    result_dir = args.result_dir
    if not result_dir:
        result_dir = (
            f"cache/results_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
        )
    if not Path(result_dir).exists():
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        args.result_dir = result_dir
        logger.info(f"Create result dir: {result_dir}")

    if not (Path(result_dir) / "traces").exists():
        (Path(result_dir) / "traces").mkdir(parents=True)

    # log the log file
    with open(os.path.join(result_dir, "log_files.txt"), "a+") as f:
        f.write(f"{LOG_FILE_NAME}\n")


def get_unfinished(config_files: list[str], result_dir: str) -> list[str]:
    result_files = glob.glob(f"{result_dir}/*.html")
    task_ids = [
        os.path.basename(f).split(".")[0].split("_")[1] for f in result_files
    ]
    unfinished_configs = []
    for config_file in config_files:
        task_id = os.path.basename(config_file).split(".")[0]
        if task_id not in task_ids:
            unfinished_configs.append(config_file)
    return unfinished_configs


@beartype
def dump_config(args: argparse.Namespace) -> None:
    config_file = Path(args.result_dir) / "config.json"
    if not config_file.exists():
        with open(config_file, "w") as f:
            json.dump(vars(args), f, indent=4)
            logger.info(f"Dump config to {config_file}")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = config()
    args.sleep_after_execution = 2.5
    prepare(args)

    test_config_base_dir = args.test_config_base_dir

    test_file_list = []
    st_idx = args.test_start_idx
    ed_idx = args.test_end_idx
    for i in range(st_idx, ed_idx):
        test_file_list.append(os.path.join(test_config_base_dir, f"{i}.json"))
    if args.skip_if_finished:
        test_file_list = get_unfinished(test_file_list, args.result_dir)
    print(f"Total {len(test_file_list)} tasks left")
    args.render = False
    args.render_screenshot = True
    args.save_trace_enabled = True

    args.current_viewport_only = True
    dump_config(args)

    test(args, humanEvals[1:])
import json
import logging
import os
import re
import shutil
import sys
import time
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from datasets import DatasetDict, load_dataset
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as async_tqdm
from transformers import HfArgumentParser

logger = logging.getLogger(__name__)

@dataclass
class OpenAIArguments:
    """Arguments specific to OpenAI API configuration."""
    
    api_key: Optional[str] = field(
        default=None,
        metadata={"help": "OpenAI API key. If not provided, will look for OPENAI_API_KEY environment variable."},
    )
    model: str = field(
        default="gpt-4o-mini",
        metadata={"help": "OpenAI model to use for generation."},
    )
    max_retries: int = field(
        default=5,
        metadata={"help": "Maximum number of retries for API calls."},
    )
    retry_delay: float = field(
        default=1.0,
        metadata={"help": "Delay between retries in seconds."},
    )
    batch_size: int = field(
        default=20,
        metadata={"help": "Number of prompts to process in parallel."},
    )
    temperature: float = field(
        default=0.7,
        metadata={"help": "Sampling temperature for generation."},
    )

@dataclass
class DataArguments:
    """Arguments pertaining to data processing."""

    output_dir: str = field(
        metadata={
            "help": "Where to save the processed dataset."
        },
    )
    dataset_name: str = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)"},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use."},
    )
    dataset_split_name: Optional[str] = field(
        default=None,
        metadata={"help": "The split name of the dataset to use."},
    )
    dataset_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to cache directory for saving and loading datasets"},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of samples to process - use for debugging."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for preprocessing."},
    )
    push_to_hub: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to push the processed dataset to the Hub."},
    )
    hub_dataset_id: Optional[str] = field(
        default=None,
        metadata={"help": "Repository namespace if pushing to the Hugging Face Hub."},
    )
    overwrite_output_dir: Optional[bool] = field(
        default=False,
        metadata={"help": "Overwrite the content of the output directory."},
    )
    speaker_name: Optional[str] = field(
        default=None,
        metadata={"help": "If `is_single_speaker`, specifies the speaker name."},
    )
    is_single_speaker: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use a single speaker prompt."},
    )
    is_new_speaker_prompt: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use the newest speaker prompt."},
    )
    speaker_id_column: Optional[str] = field(
        default=None,
        metadata={"help": "Speaker id column name. Only used if creating a dataset with multiple speaker names."},
    )
    speaker_ids_to_name_json: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a JSON file mapping speaker ids to names."},
    )
    accent_column: Optional[str] = field(
        default=None,
        metadata={"help": "Accent column name, if any."},
    )

    def __post_init__(self):
        if self.push_to_hub and self.hub_dataset_id is None:
            raise ValueError("You must specify the `hub_dataset_id` when setting `--push_to_hub=True`")

CHECKPOINT_PREFIX = "checkpoint"
_RE_CHECKPOINT = re.compile(r"^checkpoint-(\d+)\.json$")

PROMPT = """You will be given six descriptive keywords related to an audio sample of a person's speech. These keywords include:
1. The gender (e.g., male, female)
2. The level of reverberation (e.g., very roomy sounding, quite roomy sounding, slightly roomy sounding, moderate reverberation, slightly confined sounding, quite confined sounding, very confined sounding)
3. The amount of noise the sample (e.g., very noisy, quite noisy, slightly noisy, moderate ambient sound, slightly clear, quite clear, very clear)
4. The tone of the speaker's voice (e.g., very monotone, quite monotone, slightly monotone, moderate intonation, slightly expressive, quite expressive, very expressive)
5. The pace of the speaker's delivery (e.g., very slowly, quite slowly, slightly slowly, moderate speed, slightly fast, quite fast, very fast)
6. The pitch of the speaker's voice (e.g., very low pitch, quite low pitch, slightly low pitch, moderate pitch, slightly high pitch, quite high pitch, very high pitch)
Your task is to create a text description using these keywords that accurately describes the speech sample while ensuring the description remains grammatically correct and easy to understand. You should rearrange the keyword order as necessary, and substitute synonymous terms where appropriate. If the amount of noise is 'very noisy' and the level of reverberation is 'very roomy sounding', include terms like 'very bad recording' in the description. Likewise, if the amount of noise is 'very clear' and the level of reverberation is 'very confined sounding', include terms like 'very good recording' in the description. Otherwise, do not add extra details beyond what has been provided, and only return the generated description.
For example, given the following keywords: 'female', 'slightly roomy sounding', 'slightly noisy', 'very expressive', 'slightly low pitch', 'very slowly', a valid description would be: 'a woman with a deep voice speaks slowly but has an animated delivery in an echoey room with some background noise'.
For the keywords: '[gender]', '[reverberation]', '[noise]', '[speech_monotony]', '[pitch]', '[speaking_rate]', the corresponding description is:"
"""

NEW_PROMPT = """You will be given six descriptive keywords related to an audio sample of a person's speech. These keywords include:
1. The gender (male, female)
2. The level of reverberation (very distant-sounding, distant-sounding, slightly distant-sounding, slightly close-sounding, very close-sounding)
3. The amount of noise in the sample (extremely noisy, very noisy, noisy, slightly noisy, almost no noise, very clear)
4. The tone of the speaker's voice (very monotone, monotone, slightly expressive and animated, expressive and animated, very expressive and animated)
5. The pace of the speaker's delivery (very slowly, slowly, slightly slowly, moderate speed, slightly fast, fast, very fast)
6. The pitch of the speaker's voice (very low-pitch, low-pitch, slightly low-pitch, moderate pitch, slightly high-pitch, high-pitch, very high-pitch)

Your task is to create a text description using these keywords that accurately describes the speech sample.
If the amount of noise is 'very noisy' and the level of reverberation is 'very distant-sounding', you must include terms such as 'very poor recording' or `very bad recording` in the description. 
Likewise, if the amount of noise is 'very clear' and the level of reverberation is 'very close-sounding', you must include terms like 'very good recording' or `excellent recording` in the description. 
You can randomly omit the following terms, as they are default terms: 'moderate speed' and 'moderate pitch'.
Do not add extra details beyond what has been provided above. You can change the order of keywords, and replace synonymous terms.

For example, given the following keywords: 'female', 'slightly distant-sounding', 'noisy', 'very expressive and animated', 'very slowly', 'moderate pitch', a valid description would be: 'A woman speaks very slowly but has a very animated delivery. The recording is noisy and there is some roominess.'
Another valid description would be: 'In a noisy room, a female speaker delivers a very animated and expressive speech, at a very slow pace.'
Another valid description would be: 'A woman enunciates a very expressive speech. Her voice is slightly distant-sounding, with some background noise present. She speaks very slowly with a moderate pitch but a very expressive tone.'

Ensure that the generated description is grammatically correct, easy to understand, and concise. Only return one and only one description.

For the keywords: '[gender]', '[reverberation]', '[sdr_noise]', '[speech_monotony]', '[speaking_rate]', '[pitch]', the corresponding description is:
"""

NEW_PROMPT_WITH_ACCENT = """You will be given 7 descriptive keywords related to an audio sample of a person's speech. These keywords include:
1. The gender (male, female)
2. The level of reverberation (very distant-sounding, distant-sounding, slightly distant-sounding, slightly close-sounding, very close-sounding)
3. The amount of noise in the sample (extremely noisy, very noisy, noisy, slightly noisy, almost no noise, very clear)
4. The tone of the speaker's voice (very monotone, monotone, slightly expressive and animated, expressive and animated, very expressive and animated)
5. The pace of the speaker's delivery (very slowly, slowly, slightly slowly, moderate speed, slightly fast, fast, very fast)
6. The pitch of the speaker's voice (very low-pitch, low-pitch, slightly low-pitch, moderate pitch, slightly high-pitch, high-pitch, very high-pitch)
7. The accent of the speaker.

Your task is to create a text description using these keywords that accurately describes the speech sample.
If the amount of noise is 'very noisy' and the level of reverberation is 'very distant-sounding', you must include terms such as 'very poor recording' or `very bad recording` in the description. 
Likewise, if the amount of noise is 'very clear' and the level of reverberation is 'very close-sounding', you must include terms like 'very good recording' or `excellent recording` in the description. 
You can randomly omit the following terms, as they are default terms: 'moderate speed' and 'moderate pitch'.
Do not add extra details beyond what has been provided above. You can change the order of keywords, and replace synonymous terms.

For example, given the following keywords: 'female', 'slightly distant-sounding', 'noisy', 'very expressive and animated', 'very slowly', 'moderate pitch', 'Chinese', a valid description would be: 'A woman with a Chinese accent speaks very slowly but has a very animated delivery. The recording is noisy and there is some roominess.'
Another valid description would be: 'In a noisy room, a female speaker with a Chinese accent delivers a very animated and expressive speech, at a very slow pace.'
Another valid description would be: 'A woman with a Chinese accent enunciates a very expressive speech. Her voice is slightly distant-sounding, with some background noise present. She speaks very slowly with a moderate pitch but a very expressive tone.'

Ensure that the generated description is grammatically correct, easy to understand, and concise. Only return one and only one description.

For the keywords: '[gender]', '[reverberation]', '[sdr_noise]', '[speech_monotony]', '[speaking_rate]', '[pitch]', '[accent]', the corresponding description is:
"""


NEW_SINGLE_SPEAKER_PROMPT = """You will be given four descriptive keywords related to an audio sample of [speaker_name]'s speech. These keywords include:
1. The level of reverberation (very distant-sounding, distant-sounding, slightly distant-sounding, slightly close-sounding, very close-sounding)
3. The amount of noise in the sample (extremely noisy, very noisy, noisy, slightly noisy, almost no noise, very clear)
3. The tone of the speaker's voice (very monotone, monotone, slightly expressive and animated, expressive and animated, very expressive and animated)
4. The pace of the speaker's delivery (very slowly, slowly, slightly slowly, moderate speed, slightly fast, fast, very fast)

Your task is to create a text description using these keywords that accurately describes [speaker_name]'s speech sample.
If the amount of noise is 'very noisy' and the level of reverberation is 'very distant-sounding', you must include terms such as 'very poor recording' or `very bad recording` in the description. 
Likewise, if the amount of noise is 'very clear' and the level of reverberation is 'very close-sounding', you must include terms like 'very good recording' or `excellent recording` in the description. 
You can randomly omit the following terms, as they are default terms: 'moderate speed' and 'moderate pitch'.
Do not add extra details beyond what has been provided above. You can change the order of keywords, and replace synonymous terms.

For example, given the following keywords: 'slightly distant-sounding', 'clear', 'very expressive and animated', 'slightly fast', a valid description would be: '[speaker_name] speaks slightly fast but has a very animated delivery in a room with slight echo but no background noise.'
Another valid description would be: `In a very animated voice, [speaker_name] delivers words slightly quickly. The room is quite, but there's a bit of echo.'

Ensure that the generated description is grammatically correct, easy to understand, and concise. Only return one and only one description.

For the keywords: ''[reverberation]', '[sdr_noise]', '[speech_monotony]', '[speaking_rate]', the corresponding description is:
"""

SINGLE_SPEAKER_PROMPT = """You will be given four descriptive keywords related to an audio sample of [speaker_name]'s speech. These keywords include:
1. The level of reverberation (e.g., very roomy sounding, quite roomy sounding, slightly roomy sounding, moderate reverberation, slightly confined sounding, quite confined sounding, very confined sounding)
2. The amount of noise the sample (e.g., very noisy, quite noisy, slightly noisy, moderate ambient sound, slightly clear, quite clear, very clear)
3. The tone of the speaker's voice (e.g., very monotone, quite monotone, slightly monotone, moderate intonation, slightly expressive, quite expressive, very expressive)
4. The pace of the speaker's delivery (e.g., very slowly, quite slowly, slightly slowly, moderate speed, slightly fast, quite fast, very fast)

Your task is to create a single and only short text description using these keywords that accurately describes the speech sample while ensuring the description remains grammatically correct and easy to understand. You should rearrange the keyword order as necessary, and substitute synonymous terms where appropriate. If the amount of noise is 'very noisy' and the level of reverberation is 'very roomy sounding', you must include terms like 'very bad recording' in the description. Likewise, if the amount of noise is 'very clear' and the level of reverberation is 'very confined sounding', you must include terms like 'very good recording' in the description. Otherwise, do not add extra details beyond what has been provided, and only return the generated description.

For example, given the following keywords: 'slightly roomy sounding', 'quite noisy', 'very expressive', 'very slowly', a valid description would be: '[speaker_name] speaks very slowly but has an animated delivery in an echoey room with background noise.'.
Feel free to change the order of keywords, and to use synonyms, for example, with the previous keywords: `In a very expressive voice, [speaker_name] pronounces her words incredibly slowly. There's some background noise in this room with a bit of echo.'.

For the keywords: ''[reverberation]', '[noise]', '[speech_monotony]', '[speaking_rate]', the corresponding description is:
"""


def save_checkpoint(output_dir: str, generated_texts: List[str], step: int) -> None:
    """Save generated texts checkpoint."""
    checkpoint_path = f"{CHECKPOINT_PREFIX}-{step}.json"
    output_path = os.path.join(output_dir, checkpoint_path)
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as file:
        json.dump(generated_texts, file)
    logger.info(f"Saved checkpoint to {output_path}")

def load_checkpoint(checkpoint_path: str) -> List[str]:
    """Load generated texts from checkpoint."""
    with open(checkpoint_path, "r") as file:
        generated_texts = json.load(file)
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    return generated_texts

def sorted_checkpoints(output_dir: str) -> List[str]:
    """Sort checkpoints from oldest to newest."""
    glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{CHECKPOINT_PREFIX}-*")]
    ordering_and_checkpoint_path = []

    for path in glob_checkpoints:
        regex_match = re.match(f".*{CHECKPOINT_PREFIX}-([0-9]+)", path)
        if regex_match and regex_match.groups():
            ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    return [checkpoint[1] for checkpoint in sorted(ordering_and_checkpoint_path)]

def rotate_checkpoints(save_total_limit: Optional[int], output_dir: str) -> None:
    """Delete old checkpoints when limit is reached."""
    if not save_total_limit or save_total_limit <= 0:
        return

    checkpoints_sorted = sorted_checkpoints(output_dir)
    if len(checkpoints_sorted) <= save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]

    for checkpoint in checkpoints_to_be_deleted:
        logger.info(f"Deleting older checkpoint [{checkpoint}] due to save_total_limit")
        os.remove(checkpoint)

def get_last_checkpoint(folder: str) -> Tuple[List[str], int]:
    """Get the last checkpoint and current step."""
    if not os.path.exists(folder) or not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)
        return [], 0

    checkpoints = [path for path in os.listdir(folder) if _RE_CHECKPOINT.search(path)]
    if not checkpoints:
        return [], 0

    last_checkpoint = os.path.join(folder, max(checkpoints, key=lambda x: int(_RE_CHECKPOINT.search(x).groups()[0])))
    match = re.search(r"checkpoint-(\d+).json", last_checkpoint)
    cur_step = int(match.group(1))
    
    generated_texts = load_checkpoint(last_checkpoint)
    return generated_texts, cur_step

class OpenAIGenerator:
    """Handles text generation using OpenAI API."""
    
    def __init__(self, args: OpenAIArguments):
        self.client = AsyncOpenAI(api_key=args.api_key or os.getenv("OPENAI_API_KEY"))
        self.model = args.model
        self.max_retries = args.max_retries
        self.retry_delay = args.retry_delay
        self.temperature = args.temperature

    async def generate_single(self, prompt: str) -> str:
        """Generate text using OpenAI API with retry logic."""
        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                await asyncio.sleep(self.retry_delay * (attempt + 1))

    async def generate_batch(self, prompts: List[str]) -> List[str]:
        """Generate texts for a batch of prompts concurrently."""
        tasks = [self.generate_single(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)

async def process_dataset(
    raw_datasets: DatasetDict,
    generator: OpenAIGenerator,
    data_args: DataArguments,
    openai_args: OpenAIArguments,
    prepare_prompt: callable
) -> DatasetDict:
    """Process dataset with async batch processing."""
    
    for split in raw_datasets:
        logger.info(f"Processing split: {split}")
        split_output_dir = os.path.join(data_args.output_dir, split)
        
        # Load checkpoint if exists
        generated_texts, cur_step = get_last_checkpoint(split_output_dir)
        total_samples = len(raw_datasets[split])
        
        if cur_step > 0:
            logger.info(f"Resuming from step {cur_step}")
        
        # Process samples in batches
        async for i in async_tqdm(range(cur_step, total_samples, openai_args.batch_size)):
            batch_end = min(i + openai_args.batch_size, total_samples)
            batch = raw_datasets[split].select(range(i, batch_end))
            
            # Generate prompts for batch
            prompts = [prepare_prompt(sample) for sample in batch]
            
            # Generate descriptions concurrently
            batch_texts = await generator.generate_batch(prompts)
            generated_texts.extend(batch_texts)
            
            # Save checkpoint
            save_checkpoint(split_output_dir, generated_texts, batch_end)
            
        # Add generated texts to dataset
        raw_datasets[split] = raw_datasets[split].add_column("text_description", generated_texts)
    
    return raw_datasets

async def async_main():
    parser = HfArgumentParser((OpenAIArguments, DataArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        openai_args, data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        openai_args, data_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Validate arguments
    if data_args.is_single_speaker and data_args.speaker_name is None:
        raise ValueError("`is_single_speaker=True` but `speaker_name` is not specified.")

    if not data_args.is_single_speaker and data_args.speaker_name:
        raise ValueError(f"`is_single_speaker=False` but `speaker_name` is specified.")

    # Clean output directory if requested
    if data_args.overwrite_output_dir and os.path.exists(data_args.output_dir):
        logger.info("Cleaning output dir from previous run...")
        shutil.rmtree(data_args.output_dir)

    # Load dataset
    logger.info("Loading dataset...")
    if data_args.dataset_split_name:
        raw_datasets = DatasetDict()
        for split in data_args.dataset_split_name.split("+"):
            raw_datasets[split] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=split,
                cache_dir=data_args.dataset_cache_dir,
            )
    else:
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=data_args.dataset_cache_dir,
        )

    # Remove speaker embeddings column if present
    for split in raw_datasets:
        if "speaker_embeddings" in raw_datasets[split].features:
            raw_datasets[split] = raw_datasets[split].remove_columns(["speaker_embeddings"])

    # Limit samples if specified
    if data_args.max_eval_samples:
        for split in raw_datasets:
            raw_datasets[split] = raw_datasets[split].select(range(data_args.max_eval_samples))

    # Define expected columns
    EXPECTED_COLUMNS = {"gender", "pitch", "noise", "reverberation", "speech_monotony", "speaking_rate"}
    if data_args.is_single_speaker:
        EXPECTED_COLUMNS = {"noise", "reverberation", "speech_monotony", "speaking_rate"}
    
    if data_args.is_new_speaker_prompt:
        EXPECTED_COLUMNS.remove("noise")
        EXPECTED_COLUMNS.add("sdr_noise")

    # Load speaker mappings if specified
    speaker_ids_to_name = {}
    if data_args.speaker_id_column and data_args.speaker_ids_to_name_json:
        if data_args.is_single_speaker:
            raise ValueError("Cannot use both single speaker and speaker mapping.")
        
        EXPECTED_COLUMNS.add(data_args.speaker_id_column)
        with open(data_args.speaker_ids_to_name_json, "r") as f:
            speaker_ids_to_name = json.load(f)

    # Validate dataset features
    raw_datasets_features = set(raw_datasets[next(iter(raw_datasets))].features.keys())
    if not EXPECTED_COLUMNS.issubset(raw_datasets_features):
        missing_columns = EXPECTED_COLUMNS - raw_datasets_features
        raise ValueError(f"Missing columns {missing_columns} from dataset. Got {raw_datasets_features}")

    # Initialize OpenAI generator
    generator = OpenAIGenerator(openai_args)

    def prepare_prompt(sample):
        """Prepare prompt for a single sample."""
        if data_args.is_single_speaker:
            base_prompt = SINGLE_SPEAKER_PROMPT if not data_args.is_new_speaker_prompt else NEW_SINGLE_SPEAKER_PROMPT
            prompt = base_prompt.replace("[speaker_name]", data_args.speaker_name)
        elif data_args.speaker_id_column and speaker_ids_to_name.get(str(sample.get(data_args.speaker_id_column))):
            name = speaker_ids_to_name[str(sample[data_args.speaker_id_column])]
            base_prompt = SINGLE_SPEAKER_PROMPT if not data_args.is_new_speaker_prompt else NEW_SINGLE_SPEAKER_PROMPT
            prompt = base_prompt.replace("[speaker_name]", name)
        elif data_args.is_new_speaker_prompt and data_args.accent_column:
            prompt = NEW_PROMPT if sample.get(data_args.accent_column) == "Unidentified" else NEW_PROMPT_WITH_ACCENT
        elif data_args.is_new_speaker_prompt:
            prompt = NEW_PROMPT
        else:
            prompt = PROMPT

        for key in EXPECTED_COLUMNS:
            prompt = prompt.replace(f"[{key}]", str(sample[key]))
        
        if data_args.accent_column and sample.get(data_args.accent_column) != "Unidentified":
            prompt = prompt.replace("[accent]", sample[data_args.accent_column])
        
        return prompt

    # Process dataset with async batch processing
    raw_datasets = await process_dataset(
        raw_datasets=raw_datasets,
        generator=generator,
        data_args=data_args,
        openai_args=openai_args,
        prepare_prompt=prepare_prompt
    )

    # Save processed dataset
    logger.info(f"Saving processed dataset to {data_args.output_dir}")
    raw_datasets.save_to_disk(data_args.output_dir)
    
    # Push to hub if requested
    if data_args.push_to_hub:
        raw_datasets.push_to_hub(
            data_args.hub_dataset_id,
            config_name=data_args.dataset_config_name or "default",
        )

def main():
    """Entry point that runs the async main function."""
    asyncio.run(async_main())

if __name__ == "__main__":
    main() 
from typing import List, Dict, Optional
import json
import os
from transformers import AutoTokenizer, pipeline
from prompt_dict import Prompts
from utils import sequential_split, Timer
import time

def sequential_merge(chunks: List[str], tokenizer: AutoTokenizer, overlap: int) -> str:
    """
    Merges a list of text chunks into a single string, handling overlap during decoding.
    
    Args:
        chunks: List of text chunks to merge.
        tokenizer: Transformer tokenizer for decoding.
        overlap: Number of tokens to overlap between chunks.
    
    Returns:
        Merged text string.
    """
    if not chunks:
        return ""
    
    result = chunks[0]
    for chunk in chunks[1:]:
        token_ids = tokenizer(chunk, return_tensors="pt")["input_ids"][0][overlap:]
        result += tokenizer.decode(token_ids)
    return result

def load_cache_summary(cache_path: str) -> Optional[Dict]:
    """
    Loads the tree structure from a cached JSON file if it exists.
    
    Args:
        cache_path: Path to the cache file.
    
    Returns:
        Dictionary containing the tree structure, or None if file doesn't exist.
    """
    try:
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as file:
                return json.load(file)
        return None
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading cache file: {e}")
        return None

def summarize_leaf(text: str, llm: pipeline, language: str) -> str:
    """
    Summarizes a single text chunk using the appropriate language prompt.
    
    Args:
        text: Text to summarize.
        llm: Hugging Face pipeline for text generation.
        language: Language code ("en" or "zh").
    
    Returns:
        Summarized text.
    """
    try:
        prompt_key = "summarize_details" if language == "en" else "summarize_details_zh"
        prompt = Prompts[prompt_key].format(content=text)
        summary = llm(prompt)[0]["generated_text"][len(prompt):]
        return summary
    except KeyError as e:
        print(f"Prompt key error: {e}")
        return ""
    except Exception as e:
        print(f"Error generating leaf summary: {e}")
        return ""

def summarize_summary(text: str, llm: pipeline, language: str) -> str:
    """
    Summarizes a merged summary text using the appropriate language prompt.
    
    Args:
        text: Summary text to further summarize.
        llm: Hugging Face pipeline for text generation.
        language: Language code ("en" or "zh").
    
    Returns:
        Summarized text.
    """
    try:
        prompt_key = "summarize_summary" if language == "en" else "summarize_summary_zh"
        prompt = Prompts[prompt_key].format(summary=text)
        summary = llm(prompt)[0]["generated_text"][len(prompt):]
        return summary
    except KeyError as e:
        print(f"Prompt key error: {e}")
        return ""
    except Exception as e:
        print(f"Error generating summary: {e}")
        return ""

def build_tree(
    text_chunks: List[str],
    llm: pipeline,
    cache_folder: str,
    tokenizer: AutoTokenizer,
    length: int,
    overlap: int,
    merge_num: int,
    language: str
) -> tuple[Dict, float]:
    """
    Constructs a hierarchical summary tree from text chunks.
    
    Args:
        text_chunks: List of input text chunks.
        llm: Hugging Face pipeline for text generation.
        cache_folder: Directory to store cache files.
        tokenizer: Transformer tokenizer for text processing.
        length: Length of each text chunk (unused in this function but kept for compatibility).
        overlap: Number of tokens to overlap between chunks.
        merge_num: Number of chunks to merge for summarization.
        language: Language code ("en" or "zh").
    
    Returns:
        Tuple containing the tree dictionary and build time in seconds (-1 if cached).
    """
    timer = Timer()
    timer.start()
    
    # Check for cached tree
    cache_file = os.path.join(cache_folder, "tree.json")
    cached_tree = load_cache_summary(cache_file)
    if cached_tree is not None:
        return cached_tree, -1

    tree_structure: Dict = {}
    
    # Initialize leaf nodes
    for idx, chunk in enumerate(text_chunks):
        tree_structure[f"leaf_{idx}"] = {
            "text": chunk,
            "children": None,
            "parent": None
        }

    # Summarize first level
    current_level = 0
    summary_count = 0
    summaries = []
    
    for i in range(0, len(text_chunks), merge_num):
        chunk_group = text_chunks[i:i + merge_num]
        merged_text = sequential_merge(chunk_group, tokenizer, overlap)
        summary_text = summarize_leaf(merged_text, llm, language)
        
        summary_id = f"summary_{current_level}_{summary_count}"
        tree_structure[summary_id] = {
            "text": summary_text,
            "children": [f"leaf_{j}" for j in range(i, min(i + merge_num, len(text_chunks)))],
            "parent": None
        }
        
        # Update parent references for leaf nodes
        for j in range(i, min(i + merge_num, len(text_chunks))):
            tree_structure[f"leaf_{j}"]["parent"] = summary_id
        
        summaries.append(summary_text)
        summary_count += 1

    # Build higher levels
    while len(summaries) > 1.2 * merge_num:
        current_level += 1
        new_summaries = []
        new_summary_count = 0
        
        for i in range(0, len(summaries), merge_num):
            summary_group = summaries[i:i + merge_num]
            merged_summary = sequential_merge(summary_group, tokenizer, 0)
            summary_text = summarize_summary(merged_summary, llm, language)
            
            summary_id = f"summary_{current_level}_{new_summary_count}"
            tree_structure[summary_id] = {
                "text": summary_text,
                "children": [f"summary_{current_level-1}_{j}" for j in range(i, min(i + merge_num, len(summaries)))],
                "parent": None
            }
            
            # Update parent references for previous level summaries
            for j in range(i, min(i + merge_num, len(summaries))):
                tree_structure[f"summary_{current_level-1}_{j}"]["parent"] = summary_id
            
            new_summaries.append(summary_text)
            new_summary_count += 1
        
        summaries = new_summaries

    # Save the tree to cache
    try:
        os.makedirs(cache_folder, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as file:
            json.dump(tree_structure, file, ensure_ascii=False, indent=4)
    except IOError as e:
        print(f"Error saving cache file: {e}")

    return tree_structure, timer.elapsed()

if __name__ == "__main__":
    pass
from typing import List, Dict, Tuple, Optional
import os
import json
import spacy
import networkx as nx
from itertools import combinations
import time

def load_nlp(language: str = "en") -> spacy.Language:
    """
    Loads the appropriate spaCy model for the specified language.
    
    Args:
        language: Language code ("en" for English, "zh" for Chinese).
    
    Returns:
        Loaded spaCy language model.
    
    Raises:
        OSError: If the model cannot be loaded or downloaded.
    """
    model_name = "en_core_web_lg" if language == "en" else "zh_core_web_lg"
    try:
        return spacy.load(model_name)
    except OSError:
        print(f"Downloading spaCy model {model_name}...")
        try:
            spacy.cli.download(model_name)
            return spacy.load(model_name)
        except Exception as e:
            raise OSError(f"Failed to load or download spaCy model {model_name}: {e}")

def naive_extract_graph(text: str, nlp: spacy.Language) -> Dict:
    """
    Extracts nouns and their co-occurrences from text to form a graph structure.
    
    Args:
        text: Input text to process.
        nlp: SpaCy language model for processing.
    
    Returns:
        Dictionary containing nouns, co-occurrence pairs, double nouns, and appearance counts.
    """
    try:
        doc = nlp(text)
        noun_pairs = {}
        all_nouns = set()
        double_nouns = {}
        appearance_count = {}

        for sent in doc.sents:
            sentence_terms = []
            ent_positions = set()

            # Process named entities
            for ent in sent.ents:
                if ent.label_ == "PERSON":
                    name_parts = ent.text.split()
                    if len(name_parts) >= 2:
                        for name in name_parts:
                            double_nouns[name] = name_parts
                            sentence_terms.append(name)
                            appearance_count[name] = appearance_count.get(name, 0) + 1
                    else:
                        sentence_terms.append(ent.text)
                        appearance_count[ent.text] = appearance_count.get(ent.text, 0) + 1
                elif ent.label_ in ["ORG", "GPE"]:
                    sentence_terms.append(ent.text)
                    appearance_count[ent.text] = appearance_count.get(ent.text, 0) + 1
                for token in ent:
                    ent_positions.add(token.i)

            # Process other nouns and proper nouns
            for token in sent:
                if token.i in ent_positions:
                    continue
                if token.pos_ in ["NOUN", "PROPN"] and token.text.strip():
                    term = token.lemma_.lower() if token.pos_ == "NOUN" else token.text
                    sentence_terms.append(term)
                    appearance_count[term] = appearance_count.get(term, 0) + 1

            all_nouns.update(sentence_terms)

            # Count co-occurrences
            for term1, term2 in combinations(sentence_terms, 2):
                pair = tuple(sorted([term1, term2]))
                noun_pairs[pair] = noun_pairs.get(pair, 0) + 1

        return {
            "nouns": list(all_nouns),
            "cooccurrence": noun_pairs,
            "double_nouns": double_nouns,
            "appearance_count": appearance_count
        }
    except Exception as e:
        print(f"Error in naive_extract_graph: {e}")
        return {
            "nouns": [],
            "cooccurrence": {},
            "double_nouns": {},
            "appearance_count": {}
        }

def build_graph(triplets: List[Tuple[str, str, int]]) -> nx.Graph:
    """
    Builds a NetworkX graph from triplets, merging weights of duplicate edges.
    
    Args:
        triplets: List of [node1, node2, weight] tuples.
    
    Returns:
        NetworkX graph with merged edge weights.
    """
    graph = nx.Graph()
    edge_weights = {}
    
    for n1, n2, weight in triplets:
        edge = tuple(sorted([n1, n2]))
        edge_weights[edge] = edge_weights.get(edge, 0) + weight
    
    for (n1, n2), weight in edge_weights.items():
        graph.add_edge(n1, n2, weight=weight)
    
    return graph

def load_cache(cache_path: str) -> Tuple[nx.Graph, Dict, Dict]:
    """
    Loads cached graph, index, and appearance count from JSON files.
    
    Args:
        cache_path: Directory containing cache files.
    
    Returns:
        Tuple of (NetworkX graph, index dictionary, appearance count dictionary).
    
    Raises:
        IOError: If cache files cannot be read or are invalid.
    """
    try:
        graph_file = os.path.join(cache_path, "graph.json")
        index_file = os.path.join(cache_path, "index.json")
        appearance_file = os.path.join(cache_path, "appearance_count.json")
        
        with open(graph_file, "r", encoding="utf-8") as f:
            edges = json.load(f)
        with open(index_file, "r", encoding="utf-8") as f:
            index = json.load(f)
        with open(appearance_file, "r", encoding="utf-8") as f:
            appearance_count = json.load(f)
        
        graph = build_graph(edges)
        return graph, index, appearance_count
    except (json.JSONDecodeError, IOError) as e:
        raise IOError(f"Error loading cache files: {e}")

def save_graph(result: List, cache_path: str) -> None:
    """
    Saves graph edges to a JSON file.
    
    Args:
        result: List of edge triplets to save.
        cache_path: Path to the output JSON file.
    """
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
    except IOError as e:
        print(f"Error saving graph: {e}")

def save_index(result: Dict, cache_path: str) -> None:
    """
    Saves index dictionary to a JSON file.
    
    Args:
        result: Index dictionary to save.
        cache_path: Path to the output JSON file.
    """
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
    except IOError as e:
        print(f"Error saving index: {e}")

def save_appearance_count(result: Dict, cache_path: str) -> None:
    """
    Saves appearance count dictionary to a JSON file.
    
    Args:
        result: Appearance count dictionary to save.
        cache_path: Path to the output JSON file.
    """
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
    except IOError as e:
        print(f"Error saving appearance count: {e}")

def extract_graph(
    text: List[str],
    cache_folder: str,
    nlp: spacy.Language,
    use_cache: bool = True,
    reextract: bool = False
) -> Tuple[Tuple[nx.Graph, Dict, Dict], float]:
    """
    Extracts a graph from text chunks, including nodes, edges, and appearance counts.
    
    Args:
        text: List of text chunks to process.
        cache_folder: Directory to store cache files.
        nlp: SpaCy language model for text processing.
        use_cache: Whether to load from cache if available.
        reextract: Whether to force re-extraction without saving.
    
    Returns:
        Tuple of (graph, index, appearance_count) and processing time (-1 if cached).
    """
    timer = time.time()
    
    # Check for cached data
    cache_files = [
        os.path.join(cache_folder, "graph.json"),
        os.path.join(cache_folder, "index.json"),
        os.path.join(cache_folder, "appearance_count.json")
    ]
    if use_cache and all(os.path.exists(f) for f in cache_files):
        return load_cache(cache_folder), -1

    edges = []
    index = {}
    appearance_count = {}

    try:
        for i, chunk in enumerate(text):
            result = naive_extract_graph(chunk, nlp)
            appearance_count[f"leaf_{i}"] = result["appearance_count"]

            for noun in result["nouns"]:
                if noun not in index:
                    index[noun] = []
                index[noun].append(f"leaf_{i}")
            
            for noun, count in result["appearance_count"].items():
                appearance_count[noun] = appearance_count.get(noun, 0) + count

            for (head, tail), weight in result["cooccurrence"].items():
                edges.append([head, tail, weight])

        graph = build_graph(edges)
        
        if reextract:
            save_appearance_count(appearance_count, cache_files[2])
            return (graph, index, appearance_count), -1

        # Save results
        save_graph(edges, cache_files[0])
        save_index(index, cache_files[1])
        save_appearance_count(appearance_count, cache_files[2])
        
        return (graph, index, appearance_count), time.time() - timer
    except Exception as e:
        print(f"Error in extract_graph: {e}")
        return (nx.Graph(), {}, {}), -1

if __name__ == "__main__":
    edges = [
        ('a', 'b', 1),
        ('a', 'b', 3),
        ('b', 'a', 2)
    ]
    
    G = build_graph(edges)
    
    for u, v, w in G.edges(data='weight'):
        print(f"Edge ({u}, {v}): weight = {w}")
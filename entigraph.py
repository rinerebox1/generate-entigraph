import sys
import os
# sys.path.append(os.path.dirname(os.path.dirname(__file__))) # This line might cause issues in Docker if 'utils' is not structured as a package.
import json
from tqdm import tqdm
import glob
from typing import Optional, Dict, Any # Added for type hinting

# from inference.devapi import get_llm_response # Removed this import
# Assuming io_utils is in a directory named 'utils' at the same level as entigraph.py
# If utils.io_utils cannot be found, Python will raise an ImportError.
# To make this work in Docker, the 'utils' directory must be copied into the Docker image
# and be importable. A common way is to ensure 'utils' is a Python package (contains __init__.py).
# For simplicity, if jload and jdump are simple functions, they could be included directly in this script
# or ensure the Dockerfile copies the 'utils' directory correctly and Python's path includes it.

# Placeholder for io_utils functions if they are not complex.
# If they are complex or part of a larger library, ensure the utils module is correctly packaged and installed/copied.
def jload(filepath: str) -> Any:
    """Loads a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: File not found at {filepath} during jload.")
        return None # Or raise error, or return default
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from {filepath}.")
        return None # Or raise error, or return default

def jdump(data: Any, filepath: str) -> None:
    """Dumps data to a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

# from tasks.quality import QuALITY # Removed this import
# from utils.io_utils import set_openrouter_key # Removed this import
import random
from openai import OpenAI
# import os # Already imported

# Placeholder for API key, will be set by set_openrouter_key
OPENROUTER_API_KEY = None

SYSTEM_PROMPT_GENERATE_ENTITIES = """\
You are an expert in extracting key entities and summarizing technical documents.
From the provided document content, please extract a list of salient entities (people, places, concepts, technologies, etc.) and provide a concise summary of the document.
Respond with a JSON object containing two keys: "entities" (a list of strings) and "summary" (a string).
Example:
{
  "entities": ["Entity A", "Concept B", "Technology C"],
  "summary": "This document explains how Entity A interacts with Concept B using Technology C."
}
"""

SYSTEM_PROMPT_GENERATE_TWO_ENTITY_RELATIONS = """\
You are an expert in identifying relationships between entities in a technical document.
Given the document content and two entities, describe the relationship between these two entities based on the information present in the document.
If no clear relationship is described, state that.
Be concise and informative.
"""

SYSTEM_PROMPT_GENERATE_THREE_ENTITY_RELATIONS = """\
You are an expert in identifying complex relationships between multiple entities in a technical document.
Given the document content and three entities, describe how these three entities are interrelated based on the information present in the document.
If no clear relationship involving all three is described, state that.
Be concise and informative.
"""

# エンティティとサマリーを抽出するためのJSONスキーマを定義
JSON_SCHEMA_FOR_ENTITIES = {
    "type": "json_object", # Changed from "json_schema" as per OpenAI's current spec for response_format
                           # and to ensure the model strictly returns a JSON object.
                           # If your OpenAI library version or the model expects "json_schema", revert this.
    # "json_schema": { # This part might be specific to certain library versions or tools,
    #                  # for standard OpenAI API, just "type": "json_object" is often enough.
    #   "name": "extract_entities_and_summary",
    #   "strict": True,
    #   "schema": {
    #     "type": "object",
    #     "properties": {
    #       "entities": {
    #         "type": "array",
    #         "items": { "type": "string" },
    #         "description": "A list of salient entities (people, places, concepts, technologies, etc.) found in the document."
    #       },
    #       "summary": {
    #         "type": "string",
    #         "description": "A concise summary of the document, typically one or two sentences."
    #       }
    #     },
    #     "required": ["entities", "summary"],
    #     "additionalProperties": False
    #   }
    # }
}

def set_openrouter_key():
    """Sets the OpenRouter API key from the environment variable."""
    global OPENROUTER_API_KEY
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
    if not OPENROUTER_API_KEY:
        # For a script running in Docker, printing a warning is fine.
        # For critical operations, you might want to raise an error.
        print("Warning: OPENROUTER_API_KEY environment variable not set. API calls will likely fail.")

def get_llm_response(prompt: str,
          model: str,
          system_message: str,
          response_format: Optional[Dict[str, Any]] = None):
    """
    Sends a request to the OpenRouter API and returns the response.
    Accepts an optional response_format dictionary for structured outputs.
    """
    if not OPENROUTER_API_KEY:
        # This check is important. If the key isn't set, API calls will fail.
        raise ValueError("OpenRouter API key not set. Call set_openrouter_key() or ensure OPENROUTER_API_KEY environment variable is set.")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]

    api_kwargs = {
        "model": model,
        "messages": messages,
    }
    if response_format:
        api_kwargs["response_format"] = response_format

    try:
        completion = client.chat.completions.create(**api_kwargs)
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error calling LLM: {e}")
        # Depending on how you want to handle errors, you might return None, raise the exception, or retry.
        # For this script, returning None and letting the calling function handle it seems reasonable.
        return None


def generate_entities(document_content: str,
                      system_message: str,
                      openrouter_model: str):
    prompt = f"""
    ### Document Content:
    {document_content}
    """
    response_data = None
    retries = 0
    max_retries = 3 # Added a retry limit

    while not response_data and retries < max_retries:
        completion = get_llm_response(prompt,
                               openrouter_model,
                               system_message,
                               response_format=JSON_SCHEMA_FOR_ENTITIES) # Pass the schema for structured output
        if completion is None: # LLM call failed
            retries += 1
            print(f"LLM call failed for entity generation. Retry {retries}/{max_retries}...")
            if retries >= max_retries:
                print("Max retries reached for entity generation. Giving up.")
                return None # Indicate failure
            continue

        try:
            response_data_candidate = json.loads(completion)
            if isinstance(response_data_candidate, dict) and \
               'entities' in response_data_candidate and isinstance(response_data_candidate['entities'], list) and \
               'summary' in response_data_candidate and isinstance(response_data_candidate['summary'], str):
                response_data = response_data_candidate
            else:
                print(f"Invalid JSON structure in response: {completion}. Expected 'entities' (list) and 'summary' (str). Retrying...")
                # response_data remains None, loop will retry
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON: {str(e)}. Response was: {completion}. Retrying...")
            # response_data remains None, loop will retry
        except Exception as e: # Catch any other unexpected errors during processing
            print(f"An unexpected error occurred while processing entities: {str(e)}. Response: {completion}. Retrying...")

        if not response_data: # If still no valid data after try-except
            retries += 1
            if retries >= max_retries:
                print(f"Max retries reached for entity generation after invalid/failed responses. Giving up.")
                return None # Indicate failure
    return response_data


def generate_two_entity_relations(document_content: str,
                                  entity1: str,
                                  entity2: str,
                                  system_message: str,
                                  openrouter_model: str):
    prompt = f"""
    ### Document Content:
    {document_content}
    ### Entities:
    - {entity1}
    - {entity2}
    """
    completion = get_llm_response(prompt,
                       openrouter_model,
                       system_message)
    return completion

def generate_three_entity_relations(document_content: str,
                                    entity1: str,
                                    entity2: str,
                                    entity3: str,
                                    system_message: str,
                                    openrouter_model: str):
    prompt = f"""
    ### Document Content:
    {document_content}
    ### Entities:
    - {entity1}
    - {entity2}
    - {entity3}
    """
    completion = get_llm_response(prompt,
                       openrouter_model,
                       system_message)
    return completion

def generate_synthetic_data_for_document(markdown_filepath: str, model_name: str):
    random.seed(42) # Ensure reproducibility
    set_openrouter_key() # Load API key from environment

    try:
        with open(markdown_filepath, 'r', encoding='utf-8') as f:
            document_content = f.read()
    except FileNotFoundError:
        print(f"Error: Markdown file not found at {markdown_filepath}")
        return
    except Exception as e:
        print(f"Error reading markdown file {markdown_filepath}: {str(e)}")
        return

    document_id = os.path.splitext(os.path.basename(markdown_filepath))[0]
    print(f"Generating synthetic data for document: {document_id}")

    sanitized_model_name = model_name.replace("/", "_").replace(":", "_")
    output_dir = f'data/dataset/raw/markdown_entigraph_{sanitized_model_name}/'
    # os.makedirs(output_dir, exist_ok=True) # jdump will create directory
    output_path = os.path.join(output_dir, f'{document_id}.json')

    output = None
    if os.path.exists(output_path):
        loaded_data = jload(output_path)
        # Check if loaded data is in the expected list format [entities_list, summary_str, relation1_str, ...]
        if isinstance(loaded_data, list) and len(loaded_data) >= 2 and \
           isinstance(loaded_data[0], list) and isinstance(loaded_data[1], str):
            output = loaded_data
            print(f"Entities and summary loaded from existing file: {output_path}")
        else:
            print(f"Output file {output_path} has unexpected format or is incomplete. Re-initializing.")
            # Fall through to regenerate by not setting 'output'

    if output is None: # If file didn't exist, or was invalid
        print("Generating entities and summary...")
        generated_data = generate_entities(
            document_content,
            SYSTEM_PROMPT_GENERATE_ENTITIES,
            model_name
        )

        if generated_data and 'entities' in generated_data and 'summary' in generated_data:
            entities = generated_data['entities']
            summary = generated_data['summary']
            output = [entities, summary]
            jdump(output, output_path)
            print(f"Entities and summary generated and saved to {output_path}")
        else:
            print(f"Failed to generate entities and summary for {document_id}. Skipping further processing for this document.")
            return # Critical step failed, cannot proceed for this document

    # At this point, output[0] is entities and output[1] is summary
    entities = output[0]

    # Generate two-entity relations
    pair_list = []
    for i in range(len(entities)):
        for j in range(i+1, len(entities)):
            pair_list.append((entities[i], entities[j]))

    # Check which relations are already generated
    # Current output: [entities_list, summary_str, relation1, relation2, ...]
    # Number of existing relations = len(output) - 2 (for entities and summary)
    # We need to generate relations for pairs not yet covered.
    # This simple check assumes relations are added sequentially and doesn't handle partial generation of pair_list well.
    # For robustness, one might store relations as a dict keyed by sorted entity pairs, or similar.
    # However, sticking to the current script's implied logic: regenerate if not all are present.
    # A simple way to manage this is to only add new relations if they are not already in some form.
    # The original script appends, so we can assume we need to generate for all pairs,
    # and if the script is re-run, it will append duplicates unless managed.
    # For simplicity and to match the original script's behavior of appending, we'll just generate.
    # To avoid duplicates if re-run, the loading logic would need to be more robust.
    # The current script saves after *each* relation, so it can resume.

    print(f"Generating relations for {len(pair_list)} pairs of entities...")
    # Start generating relations from where we left off (if applicable)
    # The current output list has entities, summary, and then relations.
    # So, items from index 2 onwards are relations.
    # This is a simplification; a more robust approach would check *which* pairs are missing.
    # However, the original code regenerates/appends.

    # Let's refine: only generate relations not already accounted for by simple count.
    # This assumes the order of pairs in pair_list is consistent.
    num_existing_two_entity_relations = 0
    # Count how many two-entity relations we might have (this is an approximation)
    # The script mixes two-entity and three-entity relations in the same list.
    # This makes it hard to determine resumption point without more structure in 'output'.
    # For now, let's assume the script runs to completion for pairs, then for triples.
    # If we want to resume, we'd need a clear marker or separate lists for pair/triple relations.

    # Given the tqdm, it implies a full pass. Let's stick to that for now.
    # If the script is interrupted, it will have saved some relations.
    # Upon restart, it will re-generate entities/summary (if needed) and then re-generate all relations,
    # appending them again. This will lead to duplicates.

    # To improve: if output exists, and entities/summary are valid:
    # output = [entities, summary] # Start fresh for relations for this run
    # This ensures that if the script is re-run, it doesn't endlessly append old relations.
    # This is a change from the original script's behavior if it was intended to append across multiple partial runs.
    # Let's assume a clean generation of relations per run after entities/summary are set.

    current_relations = output[2:] # Get existing relations
    output = output[:2] # Reset output to just [entities, summary] before adding relations from this run

    for entity1, entity2 in tqdm(pair_list, desc="Two-entity relations"):
        # Here you could add a check if a relation for (entity1, entity2) is already in current_relations
        # to avoid re-generating, but that requires a more complex check.
        # For now, re-generate and append.
        response = generate_two_entity_relations(
            document_content, entity1, entity2,
            SYSTEM_PROMPT_GENERATE_TWO_ENTITY_RELATIONS,
            model_name)
        if response:
            output.append(response)
        jdump(output, output_path) # Save after each relation

    # Generate three-entity relations
    triple_list = []
    if len(entities) >= 3:
        for i in range(len(entities)):
            for j in range(i+1, len(entities)):
                for k in range(j+1, len(entities)):
                    triple_list.append((entities[i], entities[j], entities[k]))
        random.shuffle(triple_list) # Shuffle for variety if processing is partial

        print(f"Generating relations for {len(triple_list)} triples of entities...")
        for entity1, entity2, entity3 in tqdm(triple_list, desc="Three-entity relations"):
            response = generate_three_entity_relations(
                document_content, entity1, entity2, entity3,
                SYSTEM_PROMPT_GENERATE_THREE_ENTITY_RELATIONS,
                model_name)
            if response:
                output.append(response)
            jdump(output, output_path) # Save after each relation

    print(f"Finished generating all data for {document_id}. Saved to {output_path}")


if __name__ == '__main__':
    # The model name is hardcoded here. For Docker, this is fine.
    # It could also be passed as an environment variable or command-line argument if flexibility is needed.
    model_name_to_use = "deepseek/deepseek-chat-v3-0324:free" # Specify your model

    # Determine files to process
    markdown_files_to_process = []
    if len(sys.argv) > 1:
        # If command line arguments are provided (e.g., by Docker CMD or docker run override), use them as file paths.
        # These paths should be relative to the WORKDIR /app in the container.
        # Example: docker run myimage data/file1.md data/file2.md
        # sys.argv[0] is the script name, actual args start from sys.argv[1]
        markdown_files_to_process = [os.path.join("/app", f) if not f.startswith("/app") else f for f in sys.argv[1:]]
        # Ensure paths are absolute within the container context if they are relative.
        # The script itself expects paths (e.g. 'data/file1.md').
        # If Docker CMD is `python entigraph.py data/file1.md`, sys.argv[1] will be `data/file1.md`.
        # This path is relative to WORKDIR, so `os.path.exists` will check `/app/data/file1.md`. This is correct.
        markdown_files_to_process = sys.argv[1:]
        print(f"Processing files specified in command line: {markdown_files_to_process}")
    else:
        # Default behavior: search for markdown files in the 'data/' directory (relative to WORKDIR /app).
        # This means it will look in /app/data/ within the container.
        search_path = 'data/*.md' # This path is relative to the script's location in /app
        markdown_files_to_process = glob.glob(search_path)
        print(f"No files specified. Searching for markdown files in '{search_path}' (relative to /app)...")

    if not markdown_files_to_process:
        print("\nNo markdown files found to process.")
        print("Usage (inside container, paths relative to /app):")
        print("  - To process all files in /app/data/: python entigraph.py")
        print("  - To process specific files: python entigraph.py data/file1.md data/file2.md")
        sys.exit(1)

    print(f"Found {len(markdown_files_to_process)} file(s) to process: {markdown_files_to_process}")

    for markdown_file_path_arg in markdown_files_to_process:
        # Ensure the path used is correct for the script's execution context
        # If paths from glob are like 'data/foo.md', they are fine.
        # If paths from argv are like 'data/foo.md', they are also fine.
        # The script's open() calls will correctly resolve these relative to /app.

        # We must ensure the script can find `utils.io_utils` if it's not embedded.
        # The Dockerfile should `COPY utils /app/utils` if `utils` is a directory at the same level as Dockerfile.
        # And `utils` should have an `__init__.py` to be a package.
        # The `sys.path.append` line at the top of the original script `sys.path.append(os.path.dirname(os.path.dirname(__file__)))`
        # tries to add the parent of the script's directory to sys.path.
        # If script is /app/entigraph.py, its parent is /app, and its parent's parent is /.
        # This is likely not what was intended and might be for a specific local project structure.
        # For Docker, it's better to ensure modules are in WORKDIR or standard Python paths.
        # I've removed that sys.path modification for now and embedded jload/jdump.

        if not os.path.exists(markdown_file_path_arg):
            print(f"Warning: The file '{markdown_file_path_arg}' does not exist (checked path relative to /app). Skipping.")
            continue
        
        print(f"\n--- Starting processing for: {markdown_file_path_arg} ---")
        try:
            generate_synthetic_data_for_document(markdown_file_path_arg, model_name_to_use)
            print(f"--- Finished processing for: {markdown_file_path_arg} ---")
        except Exception as e:
            print(f"\n!!! An unexpected error occurred while processing {markdown_file_path_arg}: {e} !!!")
            # Optionally, log traceback: import traceback; traceback.print_exc()
            print("!!! Skipping to the next file. !!!")
            continue

    print("\nAll specified files processed.")

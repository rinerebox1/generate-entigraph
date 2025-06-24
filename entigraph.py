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
    "type": "json_schema",
    "json_schema": {
      "name": "extract_entities_and_summary",
      "strict": True,
      "schema": {
        "type": "object",
        "properties": {
          "entities": {
            "type": "array",
            "items": { "type": "string" },
            "description": "A list of salient entities (people, places, concepts, technologies, etc.) found in the document."
          },
          "summary": {
            "type": "string",
            "description": "A concise summary of the document, typically one or two sentences."
          }
        },
        "required": ["entities", "summary"],
        "additionalProperties": False
      }
    }
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
          response_format: Optional[Dict[str, Any]] = None): # boolから辞書型に変更
    """
    Sends a request to the OpenRouter API and returns the response.
    Accepts an optional response_format dictionary for structured outputs.
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("OpenRouter API key not set. Call set_openrouter_key() first.")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]

    # APIに渡す引数を準備
    api_kwargs = {
        "model": model,
        "messages": messages,
    }
    # response_formatが指定されている場合のみ引数に追加
    if response_format:
        api_kwargs["response_format"] = response_format

    completion = client.chat.completions.create(**api_kwargs)

    return completion.choices[0].message.content


def generate_entities(document_content: str,
                      system_message: str,
                      openrouter_model: str):
    prompt = f"""
    ### Document Content:
    {document_content}
    """
    can_read_entities = None
    response_data = None
    while not can_read_entities:
        try:
            # json_format=True の代わりに、定義したスキーマを渡す
            completion = get_llm_response(prompt,
                               openrouter_model,
                               system_message,
                               response_format=JSON_SCHEMA_FOR_ENTITIES)
            response_data = json.loads(completion)
            if 'entities' in response_data and 'summary' in response_data:
                 can_read_entities = response_data['entities']
            else:
                print(f"Invalid JSON response: {completion}. Missing 'entities' or 'summary'. Retrying...")
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON: {str(e)}. Response was: {completion}. Retrying...")
        except Exception as e:
            print(f"Failed to generate entities: {str(e)}. Retrying...")
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
    random.seed(42)
    set_openrouter_key()

    try:
        with open(markdown_filepath, 'r', encoding='utf-8') as f:
            document_content = f.read()
    except FileNotFoundError:
        print(f"Error: Markdown file not found at {markdown_filepath}")
        return
    except Exception as e:
        print(f"Error reading markdown file {markdown_filepath}: {str(e)}")
        return

    # Use filename (without extension) as document_id
    document_id = os.path.splitext(os.path.basename(markdown_filepath))[0]

    print(f"Generating synthetic data for document: {document_id}")

    # Sanitize model_name for file path
    sanitized_model_name = model_name.replace("/", "_").replace(":", "_")
    # Ensure dataset directory exists
    output_dir = f'data/dataset/raw/markdown_entigraph_{sanitized_model_name}/'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{document_id}.json')


    if os.path.exists(output_path):
        try:
            output = jload(output_path)
            if not isinstance(output, list) or (len(output) > 0 and not isinstance(output[0], list)):
                print(f"Output file {output_path} has unexpected format. Initializing anew.")
                output = [[]] # Ensure output starts as a list with an empty list for entities
        except json.JSONDecodeError:
            print(f"Could not decode JSON from {output_path}. Initializing anew.")
            output = [[]]
    else:
        output = [[]] # output[0] for entities, output[1] for summary, rest for relations

    # first check if entities are already generated
    # output should be like: [ [...entities...], "summary_string", "relation1", "relation2", ...]
    if isinstance(output[0], list) and len(output[0]) > 0 and len(output) > 1 and isinstance(output[1], str):
        entities = output[0]
        # summary = output[1] # Summary is already there
        print("Entities and summary loaded from existing file.")
    else:
        print("Generating entities and summary...")
        # The generate_entities function now returns a dict: {'entities': [...], 'summary': '...'}
        generated_data = generate_entities(
            document_content,
            SYSTEM_PROMPT_GENERATE_ENTITIES, # Use defined system prompt
            model_name)

        entities = generated_data['entities']
        summary = generated_data['summary']

        output = [entities, summary] # Initialize output with entities and summary
        jdump(output, output_path)
        print(f"Entities and summary generated and saved to {output_path}")

    pair_list = []
    # iterate over pairs of entities and generate relations
    for i in range(len(entities)):
        for j in range(i+1, len(entities)):
            pair = (entities[i], entities[j])
            pair_list.append(pair)

    print(f"Generating relations for {len(pair_list)} pairs of entities...")
    for entity1, entity2 in tqdm(pair_list):
        # Simple check if a response for this pair might already exist (crude, assumes order and no duplicates before this run)
        # A more robust check would involve inspecting the content of existing relations if format was guaranteed
        # For now, we regenerate to ensure completeness as per original logic, but save frequently
        response = generate_two_entity_relations(
            document_content, entity1, entity2,
            SYSTEM_PROMPT_GENERATE_TWO_ENTITY_RELATIONS, # Use defined system prompt
            model_name)
        if response:
            output.append(response)
        jdump(output, output_path) # Save after each relation
    
    # iterate over triples of entities and generate relations
    triple_list = []
    if len(entities) >= 3:
        for i in range(len(entities)):
            for j in range(i+1, len(entities)):
                for k in range(j+1, len(entities)):
                    triple = (entities[i], entities[j], entities[k])
                    triple_list.append(triple)
        random.shuffle(triple_list)

        print(f"Generating relations for {len(triple_list)} triples of entities...")
        for entity1, entity2, entity3 in tqdm(triple_list):
            response = generate_three_entity_relations(
                document_content, entity1, entity2, entity3,
                SYSTEM_PROMPT_GENERATE_THREE_ENTITY_RELATIONS, # Use defined system prompt
                model_name)
            if response:
                output.append(response)
            jdump(output, output_path) # Save after each relation


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

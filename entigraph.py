import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import json
from tqdm import tqdm
import glob

# from inference.devapi import gptqa # Removed this import
from utils.io_utils import jload, jdump
# from tasks.quality import QuALITY # Removed this import
# from utils.io_utils import set_openai_key # Removed this import
import random
from openai import OpenAI # Added for OpenRouter API interaction
import os # Added to access environment variables

# Placeholder for API key, will be set by set_openai_key
OPENROUTER_API_KEY = None

# Define system prompts directly in the script
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

def set_openai_key():
    """Sets the OpenRouter API key from the environment variable."""
    global OPENROUTER_API_KEY
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
    if not OPENROUTER_API_KEY:
        print("Warning: OPENROUTER_API_KEY environment variable not set.")

def gptqa(prompt: str,
          model: str,
          system_message: str,
          json_format: bool = False):
    """
    Sends a request to the OpenRouter API and returns the response.
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("OpenRouter API key not set. Call set_openai_key() first.")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]

    if json_format:
        response_format_config = {"type": "json_object"}
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format=response_format_config
        )
    else:
        completion = client.chat.completions.create(
            model=model,
            messages=messages
        )

    return completion.choices[0].message.content

def generate_entities(document_content: str,
                      system_message: str,
                      openai_model: str):
    prompt = f"""
    ### Document Content:
    {document_content}
    """
    can_read_entities = None
    response_data = None # Initialize response_data
    while not can_read_entities:
        try:
            completion = gptqa(prompt,
                               openai_model,
                               system_message,
                               json_format=True)
            response_data = json.loads(completion) # Store parsed JSON
            if 'entities' in response_data and 'summary' in response_data:
                 can_read_entities = response_data['entities'] # Check if entities key exists and is not None
            else:
                print(f"Invalid JSON response: {completion}. Missing 'entities' or 'summary'. Retrying...")
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON: {str(e)}. Response was: {completion}. Retrying...")
        except Exception as e:
            print(f"Failed to generate entities: {str(e)}. Retrying...")
    return response_data # Return the full parsed JSON object

def generate_two_entity_relations(document_content: str,
                                  entity1: str,
                                  entity2: str,
                                  system_message: str,
                                  openai_model: str):
    prompt = f"""
    ### Document Content:
    {document_content}
    ### Entities:
    - {entity1}
    - {entity2}
    """
    completion = gptqa(prompt,
                       openai_model,
                       system_message)
    return completion

def generate_three_entity_relations(document_content: str,
                                    entity1: str,
                                    entity2: str,
                                    entity3: str,
                                    system_message: str,
                                    openai_model: str):
    prompt = f"""
    ### Document Content:
    {document_content}
    ### Entities:
    - {entity1}
    - {entity2}
    - {entity3}
    """
    completion = gptqa(prompt,
                       openai_model,
                       system_message)
    return completion

def generate_synthetic_data_for_document(markdown_filepath: str, model_name: str):
    random.seed(42)
    set_openai_key()

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
    model_name = "deepseek/deepseek-chat-v3-0324:free" # Specify your model

    # 処理対象のファイルリストを決定する
    markdown_files = []
    if len(sys.argv) > 1:
        # コマンドライン引数が与えられた場合、それをファイルリストとする
        # (例: python entigraph.py file1.md file2.md)
        markdown_files = sys.argv[1:]
        print(f"Processing files specified in command line: {markdown_files}")
    else:
        # コマンドライン引数がない場合、'data/' ディレクトリの .md ファイルを検索
        search_path = 'data/*.md'
        markdown_files = glob.glob(search_path)
        print(f"No files specified. Searching for markdown files in '{search_path}'...")

    if not markdown_files:
        print("\nNo markdown files found to process.")
        print("Usage:")
        print("  - To process all files in data/: python entigraph.py")
        print("  - To process specific files: python entigraph.py data/file1.md data/file2.md")
        sys.exit(1)

    print(f"Found {len(markdown_files)} file(s) to process.")

    # 各ファイルに対して処理を実行
    for markdown_file_path in markdown_files:
        if not os.path.exists(markdown_file_path):
            print(f"Warning: The file '{markdown_file_path}' does not exist. Skipping.")
            continue
        
        print(f"\n--- Starting processing for: {markdown_file_path} ---")
        try:
            generate_synthetic_data_for_document(markdown_file_path, model_name)
            print(f"--- Finished processing for: {markdown_file_path} ---")
        except Exception as e:
            # 1つのファイルでエラーが起きても処理を止めず、次のファイルに進む
            print(f"\n!!! An unexpected error occurred while processing {markdown_file_path}: {e} !!!")
            print("!!! Skipping to the next file. !!!")
            continue

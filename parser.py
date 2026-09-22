import json
import ast
import ollama

from chunker import chunk_file, chunk_text
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from openai import OpenAI

import psycopg2
from pgvector.psycopg2 import register_vector


SYSTEM_PROMPT = """
You are a Principal Solution Architect and Technical Writer.

Analyze the supplied source code and generate engineering documentation.

Rules:
1. Only use information visible in the code.
2. Do not invent functionality.
3. If information is unknown, return "Not Available".
4. Focus on architecture, dependencies, business logic, risks, and operations.
5. Return valid JSON only.

Generate:

{
  "application_summary": "",
  "components": [],
  "functions": [],
  "classes": [],
  "dependencies": [],
  "data_flow": [],
  "business_rules": [],
  "exceptions": [],
  "operational_runbook": {
      "startup_steps": [],
      "shutdown_steps": [],
      "monitoring_points": [],
      "troubleshooting_steps": []
  },
  "architecture_description": "",
  "mermaid_diagram": ""
}

Return only valid JSON.
"""


def ingestion_pipeline(file_path):

    # Create PGVector connection
    # 1. Connect to PostgreSQL
    conn = psycopg2.connect(
        host="localhost",
        database="documind",
        user="documind_user",
        password="documind@123"
    )
    cur = conn.cursor()

    # 2. Register pgvector type in psycopg2
    register_vector(conn)

    # 3. Create VECTOR extension
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # 4. Create a table with a vector column (e.g., dimension 3)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS code (
            id SERIAL PRIMARY KEY,
            content TEXT,
            embedding vector(1024)
        );
    """)

    conn.commit()

    with open(file_path, encoding="utf-8") as f:
        source = f.read()

    chunks = []
    lines = source.splitlines()

    data_to_insert = []

    tree = ast.parse(source)

    metadata = {
        "functions": [],
        "classes": [],
        "imports": []
    }

    for node in ast.walk(tree):
        # start_line = node.lineno - 1
        # end_line = node.end_lineno
        # chunk_text = "\n".join(lines[start_line:end_line])
        # chunks.append(chunk_text)

        if isinstance(node, ast.FunctionDef):
            metadata["functions"].append(node.name)

            #chunking
            start_line = node.lineno - 1
            end_line = node.end_lineno
            chunk_text = "\n".join(lines[start_line:end_line])
            chunks.append(chunk_text)

        elif isinstance(node, ast.ClassDef):
            metadata["classes"].append(node.name)
            #chunking
            start_line = node.lineno - 1
            end_line = node.end_lineno
            chunk_text = "\n".join(lines[start_line:end_line])
            chunks.append(chunk_text)

        elif isinstance(node, ast.Import):
            for imp in node.names:
                metadata["imports"].append(imp.name)

    # Embedding Client

    # The client automatically picks up the OPENAI_API_KEY environment variable
    # client = OpenAI()
    # response = client.embeddings.create(
    #     model="text-embedding-3-small",
    #     input=""
    # )

    # chunking - AST
    print("\n\n ##### AST BASED CHUNKING ##### \n")
    for i, chunk in enumerate(chunks):
        print(f"CHUNK {i+1} = ")
        print(chunk)

        response = ollama.embeddings(
            model='mxbai-embed-large',
            prompt=chunk
        )
        print(f"\n\n CHUNK {i+1} EMBEDDING = ")
        print(response['embedding'])

        data_to_insert.append((chunk, response['embedding']))

        # print(f"\n\n CHUNK {i+1} EMBEDDING = ")
        # response = client.embeddings.create(
        #         model="text-embedding-3-small",
        #         input=chunk
        #     )
        # # Extract the float vector array
        # embedding_vector = response.data[0].embedding
        # print(f"Embedding length: {len(embedding_vector)}")
        # print(embedding_vector[:5])
    
    
    # 5. Bulk Insert the data directly as a Python list
    cur.executemany(
        "INSERT INTO code (content, embedding) VALUES (%s, %s);",
        data_to_insert
    )

    conn.commit()
    cur.close()
    conn.close()


    # chunking - tree-sitter
    print("\n\n ##### TREE-SITTER BASED CHUNKING ##### \n")
    chunks = chunk_file(file_path, "python")

    for chunk in chunks:
        print(f"Type: {chunk.node_type} | Lines: {chunk.start_line}-{chunk.end_line}")
        print(f"Parent Context: {chunk.parent_context or 'Module level'}")
        print(chunk.content)

        # Embedding didnt work for this chunking. Needs to check the below code.

        # response = ollama.embeddings(
        #     model='mxbai-embed-large',
        #     prompt=chunk
        # )
        # print("\n\n CHUNK EMBEDDING = ")
        # print(response['embedding'])

        # print("\n\n CHUNK EMBEDDING = ")
        # response = client.embeddings.create(
        #     model="text-embedding-3-small",
        #     input=chunk
        # )

        # # Extract the float vector array
        # embedding_vector = response.data[0].embedding
        # print(f"Embedding length: {len(embedding_vector)}")
        # print(embedding_vector[:5])

    # chunking - langchain text splitter
    code_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=150,
        chunk_overlap=20
    )
    chunks = code_splitter.split_text(source)

    print("\n\n ##### LANGCHAIN TEXT SPLITTER BASED CHUNKING ##### \n")

    for i, chunk in enumerate(chunks):
        print(f"CHUNK {i+1} \n")
        print(chunk)

        response = ollama.embeddings(
            model='mxbai-embed-large',
            prompt=chunk
        )
        print(f"\n\n CHUNK {i+1} EMBEDDING = ")
        print(response['embedding'])

        # print(f"\n\n CHUNK {i+1} EMBEDDING = ")
        # response = client.embeddings.create(
        #     model="text-embedding-3-small",
        #     input=chunk
        # )
        # # Extract the float vector array
        # embedding_vector = response.data[0].embedding
        # print(f"Embedding length: {len(embedding_vector)}")
        # print(embedding_vector[:5])
    
    return source, metadata


def generate_docs(source, metadata):

    prompt = f"""
    Metadata:
    {json.dumps(metadata, indent=2)}

    Source Code:
    {source}

    Return ONLY valid JSON.
    """

    response = ollama.chat(
        model="llama3.1:8b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]

def create_markdown(doc):

    md = f"""

    # Application Overview
    {doc.get('application_summary')}

    # Components
    {chr(10).join('- ' + x for x in doc.get('components'))}

    # Business Rules
    {chr(10).join('- ' + x for x in doc.get('business_rules'))}

    # Exceptions
    {chr(10).join('- ' + x for x in doc.get('exceptions'))}

    # Architecture
    {doc.get('architecture_description')}
    """

    return md

source, metadata = ingestion_pipeline("./input/discount.py")

# documentation = generate_docs(source, metadata)

# print(documentation)

# with open("output.json", "w") as f:
#     f.write(documentation)

# md_content = create_markdown(json.loads(documentation))

# print(md_content)

# with open("output.md", "w") as f:
#     f.write(md_content)

print("Documentation generated")
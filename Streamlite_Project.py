import os
import json
import time

import streamlit as st
import chromadb
from openai import OpenAI
from google import genai


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="GenAI/Data Engineer/Data Science Interview Assistance",
    page_icon="🚀",
    layout="wide",
)

st.title("🤖 GenAI/Data Engineer/Data Science Interview Assistance")

st.write(
    "Interview RAG application with embeddings, ChromaDB, "
    "chunking, similarity search, metadata filtering, "
    "prompt engineering and LLM generation."
)



OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = ""


if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
else:
    openai_client = None


if GEMINI_API_KEY:
    geminiai_client = genai.Client(api_key=GEMINI_API_KEY)
else:
    geminiai_client = None


# ============================================================
# SIDEBAR - EMBEDDING MODEL
# ============================================================

st.sidebar.header("⚙️ RAG Configuration")

embedding_provider = st.sidebar.selectbox(
    "Embedding Provider",
    ["OpenAI", "GeminiAI"],
)

if embedding_provider == "OpenAI":

    embedding_model = st.sidebar.selectbox(
        "OpenAI Embedding Model",
        [
            "text-embedding-3-small",
            "text-embedding-3-large",
        ],
    )

else:

    embedding_model = st.sidebar.selectbox(
        "Gemini Embedding Model",
        [
            "gemini-embedding-001",
        ],
    )


# ============================================================
# CHUNKING
# ============================================================

st.sidebar.subheader("📄 Chunking")

chunking_mode = st.sidebar.selectbox(
    "Chunking Mode",
    [
        "Q&A Document",
        "Character Chunking",
    ],
    help=(
        "Q&A Document keeps each interview question and answer "
        "as one retrieval document. Character Chunking splits "
        "the Q&A into smaller overlapping pieces."
    ),
)

chunk_size = st.sidebar.slider(
    "Chunk Size",
    min_value=200,
    max_value=2000,
    value=500,
    step=100,
    disabled=(chunking_mode == "Q&A Document"),
)

chunk_overlap = st.sidebar.slider(
    "Chunk Overlap",
    min_value=0,
    max_value=500,
    value=50,
    step=10,
    disabled=(chunking_mode == "Q&A Document"),
)

if chunking_mode == "Character Chunking" and chunk_overlap >= chunk_size:
    st.sidebar.error("Chunk overlap must be smaller than chunk size.")
    st.stop()


# ============================================================
# RETRIEVAL
# ============================================================

st.sidebar.subheader("🔎 Retrieval")

top_k = st.sidebar.slider(
    "Top-K",
    min_value=1,
    max_value=10,
    value=3,
)

similarity_threshold = st.sidebar.slider(
    "Distance Threshold",
    min_value=0.0,
    max_value=2.0,
    value=0.90,
    step=0.05,
    help=(
        "For Chroma cosine distance, smaller values generally mean "
        "closer vectors. Start around 0.90 and tune using the raw "
        "retrieval distances shown in the app."
    ),
)


# ============================================================
# METADATA FILTERING
# ============================================================

st.sidebar.subheader("🏷️ Metadata Filter")

filter_category = st.sidebar.selectbox(
    "Category",
    ["All", "GenAI", "Data Engineering", "Data Science"],
)

filter_difficulty = st.sidebar.selectbox(
    "Difficulty",
    ["All", "Easy", "Medium", "Hard"],
)


# ============================================================
# PROMPT ENGINEERING
# ============================================================

st.sidebar.subheader("📝 Prompt Engineering")

prompt_style = st.sidebar.selectbox(
    "Prompt Style",
    [
        "Basic",
        "Role Based",
        "Structured Interview",
        "Strict RAG",
    ],
)


# ============================================================
# LLM GENERATION MODEL
# ============================================================

st.sidebar.subheader("🧠 Generation Model")

llm_provider = st.sidebar.selectbox(
    "LLM Provider",
    [
        "OpenAI",
        "Gemini",
    ],
)

if llm_provider == "OpenAI":

    llm_model = st.sidebar.selectbox(
        "OpenAI LLM",
        [
            "gpt-5-mini",
            "gpt-5",
        ],
    )

else:

    llm_model = st.sidebar.selectbox(
        "Gemini LLM",
        [
            "gemini-2.5-flash",
        ],
    )


# ============================================================
# CHROMADB
# ============================================================

@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(
        path="./vector_db"
    )


chroma_client = get_chroma_client()


# ============================================================
# COLLECTION NAME
# Include embedding + chunk configuration so incompatible
# vector spaces/configurations are not mixed.
# ============================================================

def build_collection_name():

    provider_name = embedding_provider.lower()
    model_name = embedding_model.replace("-", "_").replace(".", "_")

    if chunking_mode == "Q&A Document":
        chunk_config = "qa_document"
    else:
        chunk_config = (
            f"chunks_{chunk_size}_overlap_{chunk_overlap}"
        )

    return (
        f"rag_{provider_name}_{model_name}_"
        f"{chunk_config}_cosine"
    )


collection_name = build_collection_name()


# ============================================================
# RESET VECTOR DB
# ============================================================

if st.sidebar.button("🗑️ Reset Current Vector Collection"):

    try:

        chroma_client.delete_collection(
            name=collection_name
        )

        st.sidebar.success(
            f"Deleted: {collection_name}"
        )

        st.rerun()

    except Exception as e:

        st.sidebar.warning(
            f"Collection was not found or could not be deleted: {e}"
        )


# ============================================================
# EMBEDDINGS
# ============================================================

def create_embeddings(texts):

    if not texts:
        return []

    if embedding_provider == "OpenAI":

        if openai_client is None:
            raise RuntimeError(
                "OPENAI_API_KEY is not configured. "
                "Set it as an environment variable or Streamlit secret."
            )

        response = openai_client.embeddings.create(
            model=embedding_model,
            input=texts,
        )

        return [
            item.embedding
            for item in response.data
        ]


    if embedding_provider == "GeminiAI":

        if geminiai_client is None:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured. "
                "Set it as an environment variable or Streamlit secret."
            )

        response = geminiai_client.models.embed_content(
            model=embedding_model,
            contents=texts,
        )

        return [
            item.values
            for item in response.embeddings
        ]

    raise ValueError(
        f"Unsupported embedding provider: {embedding_provider}"
    )


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    file_path = (
        "genai_data_engineer_data_science_10000_interview_qa.json"
    )

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            "The JSON file must contain a list of interview records."
        )

    return data


# ============================================================
# NORMALIZE DATA
# Preserve category/topic/difficulty/tags for Chroma metadata.
# ============================================================

def normalize_data(data):

    records = []

    for item in data:

        question = str(
            item.get("question", "")
        ).strip()

        answer = str(
            item.get("answer", "")
        ).strip()

        if not question or not answer:
            continue

        record_id = str(
            item.get(
                "id",
                f"qa_{len(records) + 1:05d}",
            )
        )

        tags = item.get("tags", [])

        if not isinstance(tags, list):
            tags = [str(tags)]

        records.append(
            {
                "id": record_id,
                "category": str(
                    item.get("category", "Unknown")
                ),
                "topic": str(
                    item.get("topic", "Unknown")
                ),
                "difficulty": str(
                    item.get("difficulty", "Unknown")
                ),
                "tags": tags,
                "question": question,
                "answer": answer,
            }
        )

    return records


# ============================================================
# CREATE RETRIEVAL DOCUMENTS
#
# Default:
#     1 Q&A = 1 Chroma document
#
# Optional:
#     Character chunking with overlap
# ============================================================

def create_documents(records):

    document_ids = []
    documents = []
    metadatas = []

    for record in records:

        text = (
            f"Question: {record['question']}\n"
            f"Answer: {record['answer']}"
        )

        if chunking_mode == "Q&A Document":

            chunks = [text]

        else:

            chunks = []

            start = 0

            while start < len(text):

                end = min(
                    start + chunk_size,
                    len(text),
                )

                chunk = text[start:end]

                if chunk.strip():
                    chunks.append(chunk)

                if end >= len(text):
                    break

                start = end - chunk_overlap

        for chunk_number, chunk in enumerate(chunks):

            if chunking_mode == "Q&A Document":

                document_id = record["id"]

            else:

                document_id = (
                    f"{record['id']}_chunk_{chunk_number}"
                )

            metadata = {
                "original_id": record["id"],
                "category": record["category"],
                "topic": record["topic"],
                "difficulty": record["difficulty"],
                "tags": ",".join(
                    str(tag)
                    for tag in record["tags"]
                ),
                "chunk_number": chunk_number,
                "chunking_mode": chunking_mode,
            }

            document_ids.append(document_id)
            documents.append(chunk)
            metadatas.append(metadata)

    return (
        document_ids,
        documents,
        metadatas,
    )


# ============================================================
# GET / CREATE COLLECTION
# ============================================================

def get_collection():

    return chroma_client.get_or_create_collection(
        name=collection_name,
        configuration={
            "hnsw": {
                "space": "cosine",
            }
        },
    )


# ============================================================
# LOAD DATA INTO CHROMADB
# ============================================================

def load_into_chroma():

    data = load_data()

    records = normalize_data(data)

    (
        document_ids,
        documents,
        metadatas,
    ) = create_documents(records)

    if not documents:
        raise ValueError(
            "No valid question/answer documents were found."
        )

    collection = get_collection()

    existing_count = collection.count()

    # Performance optimization:
    # don't re-embed an already-built collection.
    if existing_count > 0:

        return (
            collection,
            existing_count,
            0,
            len(records),
        )

    progress = st.progress(0)

    batch_size = 100

    total = len(documents)

    inserted = 0

    for start in range(
        0,
        total,
        batch_size,
    ):

        end = min(
            start + batch_size,
            total,
        )

        batch_documents = documents[start:end]
        batch_ids = document_ids[start:end]
        batch_metadata = metadatas[start:end]

        embeddings = create_embeddings(
            batch_documents
        )

        collection.upsert(
            ids=batch_ids,
            documents=batch_documents,
            metadatas=batch_metadata,
            embeddings=embeddings,
        )

        inserted += len(batch_documents)

        progress.progress(
            end / total
        )

    progress.empty()

    return (
        collection,
        total,
        inserted,
        len(records),
    )


# ============================================================
# PROMPT ENGINEERING
# ============================================================

def create_prompt(question, context):

    if prompt_style == "Basic":

        return f"""
Answer the question using the supplied context.

Context:
{context}

Question:
{question}

Answer:
"""


    if prompt_style == "Role Based":

        return f"""
You are an expert interviewer and technical mentor.

Answer the user's question using the supplied context.

Context:
{context}

User Question:
{question}

Give a clear, technically accurate interview-ready answer.
"""


    if prompt_style == "Structured Interview":

        return f"""
You are helping a candidate prepare for a technical interview.

Use the supplied context to answer the question.

Context:
{context}

Question:
{question}

Return the answer in this structure:

1. Short Definition
2. Detailed Explanation
3. Real-world Example
4. Interview Tip

Do not invent facts that are not supported by the context.
"""


    # Strict RAG

    return f"""
You are a Retrieval-Augmented Generation interview assistant.

STRICT RULES:

1. Use only the supplied context.
2. Do not invent information.
3. If the context does not contain enough information,
   say: "I don't have enough information from the retrieved context."
4. Give a concise, interview-ready answer.
5. Prefer the retrieved interview answer when it directly matches
   the question.

Retrieved Context:
{context}

Question:
{question}

Final Answer:
"""


# ============================================================
# LLM GENERATION
# ============================================================

def generate_answer(prompt):

    if llm_provider == "OpenAI":

        if openai_client is None:
            raise RuntimeError(
                "OPENAI_API_KEY is not configured."
            )

        response = openai_client.chat.completions.create(
            model=llm_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful technical interview "
                        "assistant."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            
        )

        return response.choices[0].message.content


    if llm_provider == "Gemini":

        if geminiai_client is None:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured."
            )

        response = geminiai_client.models.generate_content(
            model=llm_model,
            contents=prompt,
        )

        return response.text


    raise ValueError(
        f"Unsupported LLM provider: {llm_provider}"
    )


# ============================================================
# LOAD / BUILD VECTOR DB BUTTON
# ============================================================

if st.sidebar.button("🚀 Load / Build Vector DB"):

    try:

        start_time = time.time()

        (
            collection,
            count,
            inserted,
            record_count,
        ) = load_into_chroma()

        elapsed = (
            time.time() - start_time
        )

        if inserted > 0:

            st.sidebar.success(
                f"Vector DB built: {inserted:,} documents"
            )

        else:

            st.sidebar.success(
                f"Vector DB already exists: {count:,} documents"
            )

        st.sidebar.info(
            f"Source Q&A records: {record_count:,}"
        )

        st.sidebar.info(
            f"Processing time: {elapsed:.2f} seconds"
        )

        st.session_state["db_ready"] = True

    except Exception as e:

        st.sidebar.error(
            f"Vector DB error: {e}"
        )


# ============================================================
# USER QUESTION
# ============================================================

st.subheader("💬 Ask a Question")

question = st.text_input(
    "Enter your interview question"
)


# ============================================================
# SEARCH & GENERATE
# ============================================================

if st.button(
    "🔍 Search & Generate Answer"
):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

        st.stop()


    try:

        collection = get_collection()

        if collection.count() == 0:

            st.warning(
                "The vector database is empty. "
                "Click 'Load / Build Vector DB' first."
            )

            st.stop()


        # ====================================================
        # QUERY EMBEDDING
        # ====================================================

        start_embedding = time.time()

        query_embedding = create_embeddings(
            [question.strip()]
        )[0]

        embedding_time = (
            time.time() - start_embedding
        )


        # ====================================================
        # METADATA FILTER
        # ====================================================

        where_conditions = []

        if filter_category != "All":

            where_conditions.append(
                {
                    "category": filter_category
                }
            )

        if filter_difficulty != "All":

            where_conditions.append(
                {
                    "difficulty": filter_difficulty
                }
            )


        if len(where_conditions) == 0:

            where_filter = None

        elif len(where_conditions) == 1:

            where_filter = where_conditions[0]

        else:

            where_filter = {
                "$and": where_conditions
            }


        # ====================================================
        # CHROMA SIMILARITY SEARCH
        # ====================================================

        start_search = time.time()

        query_kwargs = {
            "query_embeddings": [
                query_embedding
            ],
            "n_results": top_k,
            "include": [
                "documents",
                "metadatas",
                "distances",
            ],
        }

        if where_filter is not None:

            query_kwargs["where"] = where_filter


        results = collection.query(
            **query_kwargs
        )

        search_time = (
            time.time() - start_search
        )


        documents = (
            results.get("documents", [[]])[0]
        )

        distances = (
            results.get("distances", [[]])[0]
        )

        metadatas = (
            results.get("metadatas", [[]])[0]
        )


        # ====================================================
        # RAW RETRIEVAL RESULTS
        # This is important while tuning similarity threshold.
        # ====================================================

        st.subheader(
            "🔎 Raw ChromaDB Results"
        )

        if not documents:

            st.warning(
                "ChromaDB returned no results."
            )

            st.stop()


        for i, (
            document,
            metadata,
            distance,
        ) in enumerate(
            zip(
                documents,
                metadatas,
                distances,
            )
        ):

            st.write(
                f"Result {i + 1} | "
                f"Cosine Distance: {distance:.4f}"
            )

            with st.expander(
                f"View Raw Result {i + 1}"
            ):

                st.write(document)

                st.caption(
                    f"Metadata: {metadata}"
                )


        # ====================================================
        # DISTANCE FILTERING
        # ====================================================

        filtered = []

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        ):

            if distance <= similarity_threshold:

                filtered.append(
                    (
                        document,
                        metadata,
                        distance,
                    )
                )


        st.subheader(
            "📚 Retrieved Chunks"
        )


        if not filtered:

            best_distance = min(
                distances
            )

            st.warning(
                "No results passed the distance threshold."
            )

            st.info(
                f"Best returned distance: "
                f"{best_distance:.4f}. "
                f"Current threshold: "
                f"{similarity_threshold:.2f}."
            )

            st.info(
                "Try increasing the threshold, "
                "or inspect the raw results above."
            )

            st.stop()


        for i, (
            document,
            metadata,
            distance,
        ) in enumerate(filtered):

            with st.expander(
                f"Chunk {i + 1} | "
                f"Distance: {distance:.4f}",
                expanded=(i == 0),
            ):

                st.write(document)

                st.caption(
                    f"Metadata: {metadata}"
                )


        # ====================================================
        # CONTEXT
        # ====================================================

        context_parts = []

        for i, (
            document,
            metadata,
            distance,
        ) in enumerate(filtered):

            context_parts.append(
                f"""
[Retrieved Document {i + 1}]
Distance: {distance:.4f}
Category: {metadata.get("category", "")}
Topic: {metadata.get("topic", "")}
Difficulty: {metadata.get("difficulty", "")}

{document}
"""
            )


        context = "\n\n".join(
            context_parts
        )


        # ====================================================
        # PROMPT
        # ====================================================

        prompt = create_prompt(
            question,
            context,
        )


        with st.expander(
            "📝 Final Prompt Sent to LLM"
        ):

            st.code(
                prompt,
                language="text",
            )


        # ====================================================
        # GENERATION
        # ====================================================

        start_generation = time.time()

        answer = generate_answer(
            prompt
        )

        generation_time = (
            time.time()
            - start_generation
        )


        # ====================================================
        # FINAL ANSWER
        # ====================================================

        st.subheader(
            "🤖 Interview Answer"
        )

        st.write(answer)


        # ====================================================
        # PERFORMANCE METRICS
        # ====================================================

        st.subheader(
            "📊 Performance Metrics"
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Embedding Time",
            f"{embedding_time:.3f}s",
        )

        col2.metric(
            "Search Time",
            f"{search_time:.3f}s",
        )

        col3.metric(
            "Generation Time",
            f"{generation_time:.3f}s",
        )

        col4.metric(
            "Retrieved",
            len(filtered),
        )

        st.caption(
            f"Collection: {collection_name}"
        )


    except Exception as e:

        st.error(
            f"Error: {e}"
        )

# skills_config.py
"""
Configuration of core skills, concept mappings, capability groups,
and production readiness keywords for the Candidate Intelligence Engine.
"""

# Core Concept categories mapping to specific skills in candidates' profiles
CORE_CONCEPTS = {
    "vector_databases": {
        "weight": 1.0,
        "keywords": [
            "milvus", "qdrant", "weaviate", "pinecone", "faiss",
            "vector database", "vector databases", "vector search",
            "opensearch", "elasticsearch", "chroma", "pgvector",
            "annoy", "nmslib", "vespa"
        ]
    },
    "retrieval_rag": {
        "weight": 1.0,
        "keywords": [
            "retrieval", "rag", "hybrid retrieval", "hybrid search",
            "dense retrieval", "bm25", "embeddings", "sentence-transformers",
            "sentence transformers", "bge", "e5", "retrieval-augmented generation",
            "retrieval augmented generation", "semantic search", "vector search",
            "dense passage retrieval", "dpr", "colbert", "splade"
        ]
    },
    "ranking_evaluation": {
        "weight": 1.0,
        "keywords": [
            "ranking", "re-ranking", "re-ranker", "learning-to-rank",
            "learning to rank", "ndcg", "mrr", "map", "evaluation framework",
            "evaluation frameworks", "ltr", "pointwise", "pairwise", "listwise",
            "relevance feedback", "click-through rate", "ctr", "precision@k"
        ]
    },
    "llm_dl": {
        "weight": 0.8,
        "keywords": [
            "llm", "llms", "fine-tuning", "fine tuning", "lora", "qlora",
            "peft", "transformers", "bert", "gpt", "gpt-4", "gpt-3.5",
            "llama", "mistral", "gemma", "qwen", "phi", "falcon",
            "pytorch", "tensorflow", "deep learning", "dl",
            "instruction tuning", "rlhf", "dpo", "sft"
        ]
    },
    "programming_ml": {
        "weight": 0.8,
        "keywords": [
            "python", "machine learning", "ml", "applied ml",
            "natural language processing", "nlp", "distributed systems",
            "inference optimization", "xgboost", "bentoml", "triton",
            "onnx", "quantization", "model serving", "mlops", "kubeflow",
            "airflow", "spark", "kafka", "redis", "fastapi"
        ]
    }
}

# ---------------------------------------------------------------------------
# Capability Groups for Career Relevance Scoring (6 groups)
# Each group represents a distinct recruiter-facing competency.
# A capability is "found" if ANY of its keywords appear in career descriptions.
# Keywords expanded with semantic equivalents to avoid missing synonymous terms.
# ---------------------------------------------------------------------------
CAPABILITIES = {
    "retrieval": [
        "retrieval", "semantic search", "vector search",
        "hybrid search", "dense retrieval", "information retrieval",
        "search engine", "search infrastructure", "search platform",
        "search systems", "search relevance", "document retrieval",
        "passage retrieval", "text retrieval"
    ],
    "ranking": [
        "ranking", "learning to rank", "relevance optimization",
        "re-ranking", "ranking pipeline", "ranking algorithm",
        "ranking system", "ranking engine", "search relevance",
        "feed ranking", "content ranking", "result ranking",
        "ltr", "pointwise", "pairwise", "listwise"
    ],
    "recommendation": [
        "recommendation", "recommendation system", "recommendation engine",
        "recommender", "collaborative filtering", "content-based filtering",
        "personalization", "content discovery", "user recommendations",
        "product recommendations", "item recommendations", "feed personalization",
        "user personalization", "personalised"
    ],
    "matching": [
        "matching", "candidate matching", "matching infrastructure",
        "entity matching", "entity resolution", "similarity search",
        "profile matching", "nearest neighbor", "semantic similarity",
        "candidate discovery", "approximate nearest neighbor",
        "ann search", "knn", "k-nn"
    ],
    "vector_database": [
        "milvus", "pinecone", "qdrant", "faiss",
        "weaviate", "chroma", "pgvector", "vespa",
        "vector database", "vector databases", "vector store",
        "opensearch", "elasticsearch", "annoy",
        "vector index", "embedding index"
    ],
    "llm_finetuning": [
        "llm", "large language model", "gpt", "llama",
        "mistral", "gemma", "qwen", "bert", "transformers",
        "fine tuning", "fine-tuning", "lora", "qlora",
        "peft", "instruction tuning", "rlhf", "dpo", "sft",
        "pretrained model", "foundation model", "language model"
    ]
}

# ---------------------------------------------------------------------------
# Production Readiness Keyword Dimensions
# Each dimension is treated as a single binary concept (found / not found).
# Synonyms within a dimension are grouped to avoid double-counting.
# Final production score = fraction of dimensions evidenced.
# ---------------------------------------------------------------------------
PRODUCTION_KEYWORDS = {
    "deployment": [
        "production", "deployed", "deployment", "live",
        "release", "shipped", "rollout", "launch", "productionized"
    ],
    "scale": [
        "latency", "real-time", "millions of requests", "scalable",
        "high throughput", "low latency", "at scale", "billion",
        "traffic", "qps", "tps", "large scale", "massive scale"
    ],
    "reliability": [
        "monitoring", "observability", "high availability",
        "uptime", "sla", "alerting", "incident", "reliability",
        "incident response", "fault tolerance"
    ],
    "engineering": [
        "ci/cd", "microservices", "automation",
        "devops", "infrastructure as code", "terraform",
        "docker", "kubernetes", "load balancing",
        "containerization", "orchestration"
    ]
}

# ---------------------------------------------------------------------------
# Technology display names for evidence extraction in explanations
# Maps lowercase keywords found in career text → display name for reasoning
# ---------------------------------------------------------------------------
TECH_DISPLAY = {
    "faiss": "FAISS",
    "milvus": "Milvus",
    "pinecone": "Pinecone",
    "qdrant": "Qdrant",
    "weaviate": "Weaviate",
    "chroma": "Chroma",
    "pgvector": "pgvector",
    "elasticsearch": "Elasticsearch",
    "opensearch": "OpenSearch",
    "vespa": "Vespa",
    "llama": "LLaMA",
    "mistral": "Mistral",
    "gemma": "Gemma",
    "gpt-4": "GPT-4",
    "gpt-3": "GPT-3",
    "bert": "BERT",
    "lora": "LoRA",
    "qlora": "QLoRA",
    "peft": "PEFT",
    "pytorch": "PyTorch",
    "tensorflow": "TensorFlow",
    "spark": "Apache Spark",
    "kafka": "Kafka",
    "redis": "Redis",
    "kubernetes": "Kubernetes",
    "docker": "Docker",
    "bm25": "BM25",
    "colbert": "ColBERT",
    "sentence-transformers": "Sentence-Transformers",
    "sentence transformers": "Sentence-Transformers",
}

# ---------------------------------------------------------------------------
# Capability action phrases used in evidence-based reasoning generation
# Maps capability name → list of description phrases (in priority order)
# ---------------------------------------------------------------------------
CAP_ACTIONS = {
    "retrieval": [
        "retrieval and semantic search systems",
        "hybrid retrieval pipelines",
        "search infrastructure",
    ],
    "ranking": [
        "ranking and relevance scoring systems",
        "learning-to-rank pipelines",
        "relevance optimization infrastructure",
    ],
    "recommendation": [
        "recommendation engines",
        "personalization and content discovery systems",
        "collaborative filtering infrastructure",
    ],
    "matching": [
        "candidate matching and similarity search systems",
        "entity resolution pipelines",
        "nearest-neighbor matching infrastructure",
    ],
    "vector_database": [
        "vector search and embedding-based retrieval",
        "vector database infrastructure",
        "dense embedding pipelines",
    ],
    "llm_finetuning": [
        "LLM fine-tuning workflows",
        "transformer model optimization (LoRA/PEFT)",
        "instruction-tuned LLM systems",
    ],
}

# --- Title Classification ---
AI_TITLE_KEYWORDS = [
    "ai engineer", "artificial intelligence engineer", "ml engineer",
    "machine learning engineer", "nlp engineer", "nlp developer",
    "applied ml scientist", "applied machine learning engineer",
    "data scientist", "deep learning engineer", "staff ai engineer",
    "senior ai engineer", "senior ml engineer", "lead ml engineer",
    "lead ai engineer", "applied ml engineer", "junior ml engineer",
    "ai research engineer", "research engineer", "nlp scientist",
    "recommendation systems engineer", "search engineer",
    "applied scientist", "ml scientist", "staff machine learning engineer",
    "senior machine learning engineer", "senior nlp engineer",
    "senior data scientist", "staff ml engineer"
]

GENERIC_TECH_TITLES = [
    "software engineer", "backend engineer", "platform engineer",
    "data engineer", "systems engineer",
    "technical lead", "tech lead", "senior engineer",
    "member of technical staff", "mts", "developer",
    "full stack developer", "fullstack developer",
    "software developer", "senior software engineer",
    "principal engineer", "founding engineer"
]

IRRELEVANT_TITLES = [
    "marketing manager", "operations manager", "accountant",
    "graphic designer", "hr manager", "sales executive",
    "customer support", "civil engineer", "mechanical engineer",
    "business analyst", "project manager", "scrum master",
    "content writer", "java developer", "android developer",
    "ios developer", "qa engineer", "test engineer",
    "devops engineer", "site reliability engineer"
]

# Text phrases in job descriptions to look for when identifying "hidden gem" candidates
TIER5_CAREER_KEYWORDS = [
    "recommendation system", "recommendation systems", "recommendation engine",
    "search engine", "search infrastructure", "information retrieval",
    "vector search", "semantic search", "retrieval engine", "ranking pipeline",
    "ranking algorithms", "candidate matching", "matching infrastructure",
    "retrieval and ranking", "rag pipeline", "rag systems"
]

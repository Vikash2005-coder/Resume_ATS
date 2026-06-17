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

# --- Capability Groups for Career Relevance Scoring ---
# Each group represents a distinct recruiter-facing competency.
# A capability is "found" if ANY of its keywords appear in career descriptions.
CAPABILITIES = {
    "retrieval": [
        "retrieval", "semantic search", "vector search",
        "hybrid search", "dense retrieval", "information retrieval",
        "search engine", "search infrastructure"
    ],
    "ranking": [
        "ranking", "learning to rank", "relevance optimization",
        "re-ranking", "ranking pipeline", "ranking algorithm",
        "ranking system", "ranking engine"
    ],
    "recommendation": [
        "recommendation", "recommendation system", "recommendation engine",
        "recommender", "collaborative filtering", "content-based filtering"
    ],
    "matching": [
        "matching", "candidate discovery", "matching infrastructure",
        "candidate matching", "entity matching", "similarity matching"
    ]
}

# --- Production Readiness Keyword Dimensions ---
PRODUCTION_KEYWORDS = {
    "deployment": [
        "production", "deployed", "deployment", "live",
        "release", "shipped", "rollout", "launch"
    ],
    "scale": [
        "latency", "real-time", "millions of requests", "scalable",
        "high throughput", "low latency", "at scale", "billion",
        "traffic", "qps", "tps"
    ],
    "reliability": [
        "monitoring", "observability", "high availability",
        "uptime", "sla", "alerting", "incident", "reliability"
    ],
    "engineering": [
        "ci/cd", "microservices", "automation",
        "devops", "infrastructure as code", "terraform",
        "docker", "kubernetes", "load balancing"
    ]
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

# Text phrases in job descriptions to look for when identifying Tier-5 "hidden gem" candidates
TIER5_CAREER_KEYWORDS = [
    "recommendation system", "recommendation systems", "recommendation engine",
    "search engine", "search infrastructure", "information retrieval",
    "vector search", "semantic search", "retrieval engine", "ranking pipeline",
    "ranking algorithms", "candidate matching", "matching infrastructure",
    "retrieval and ranking", "rag pipeline", "rag systems"
]

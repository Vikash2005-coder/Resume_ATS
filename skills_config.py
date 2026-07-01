# skills_config.py
"""
Configuration of core skills, concept mappings, capability groups,
and production readiness keywords for the Candidate Intelligence Engine.

v2 additions:
  - CAPABILITY_DESCRIPTIONS: rich text per capability for semantic dot-product matching
  - TITLE_LEVEL_MAP: seniority mapping for career progression scoring
  - TECH_GENERATIONS: tech era mapping for learning velocity scoring
  - OWNERSHIP_VERBS: high-evidence ownership verb list
  - DOMAIN_GROUPS: career domain groups for consistency scoring
  - PRODUCTION_KEYWORDS: expanded from 4 → 12 dimensions
"""

# ---------------------------------------------------------------------------
# Core Concept categories for skill matching
# ---------------------------------------------------------------------------
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
# Capability Groups — 6 recruiter-facing competencies
# Used for both keyword-based and semantic detection
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
# Capability Descriptions — rich natural-language text for semantic matching
# Each description is encoded by BGE-small at startup and dot-producted
# against cached candidate embeddings in Stage 2 (< 5ms for 1000 candidates).
# ---------------------------------------------------------------------------
CAPABILITY_DESCRIPTIONS = {
    "retrieval": (
        "Built semantic search and information retrieval systems using dense embeddings, "
        "BM25, hybrid search, FAISS, vector databases, and retrieval-augmented generation "
        "pipelines. Experience with sentence-transformers, BGE, E5, ColBERT, and SPLADE "
        "for high-quality passage retrieval at scale."
    ),
    "ranking": (
        "Designed learning-to-rank systems and relevance scoring pipelines, including "
        "NDCG and MRR optimization, cross-encoder re-ranking, and pointwise/pairwise/listwise "
        "training. Built search relevance feedback loops and A/B tested ranking algorithms "
        "in production search engines."
    ),
    "recommendation": (
        "Built recommendation engines using collaborative filtering, content-based filtering, "
        "personalization algorithms, and user behavior modeling. Developed feed ranking, "
        "item similarity, and real-time recommendation systems serving millions of users."
    ),
    "matching": (
        "Implemented candidate matching, entity resolution, and nearest-neighbor search "
        "systems. Built approximate ANN infrastructure using FAISS, HNSW, and IVF indexes. "
        "Designed semantic similarity and profile matching pipelines for talent discovery."
    ),
    "vector_database": (
        "Deployed and maintained large-scale vector databases including Milvus, Pinecone, "
        "Qdrant, Weaviate, Chroma, and pgvector. Managed embedding indexing, approximate "
        "nearest neighbor search, and high-throughput vector retrieval for production systems."
    ),
    "llm_finetuning": (
        "Fine-tuned large language models using LoRA, QLoRA, PEFT, RLHF, DPO, and "
        "instruction tuning on custom datasets. Deployed transformer models including LLaMA, "
        "Mistral, Gemma, and GPT for downstream NLP tasks. Optimized inference with "
        "quantization, ONNX, and speculative decoding."
    ),
}

# ---------------------------------------------------------------------------
# Production Readiness — 12 independent dimensions (expanded from 4)
# Each dimension is binary (found/not found); final score = fraction found.
# More dimensions = more granular continuous score.
# ---------------------------------------------------------------------------
PRODUCTION_KEYWORDS = {
    "deployment": [
        "production", "deployed", "deployment", "live",
        "release", "shipped", "rollout", "launch", "productionized",
        "went live", "in production"
    ],
    "scale": [
        "million", "billion", "at scale", "high throughput",
        "large scale", "massive scale", "real-time", "qps", "tps",
        "millions of users", "millions of requests"
    ],
    "latency": [
        "latency", "low latency", "p99", "p95", "millisecond",
        "ms latency", "sub-second", "fast inference", "response time",
        "tail latency", "slo"
    ],
    "reliability": [
        "reliability", "high availability", "uptime", "sla",
        "fault tolerance", "disaster recovery", "redundancy",
        "99.9", "five nines"
    ],
    "monitoring": [
        "monitoring", "observability", "alerting", "logging",
        "metrics", "dashboards", "datadog", "prometheus", "grafana",
        "splunk", "cloudwatch", "newrelic"
    ],
    "ci_cd": [
        "ci/cd", "continuous integration", "continuous deployment",
        "github actions", "jenkins", "gitlab ci", "automated testing",
        "pipeline", "cd pipeline", "ci pipeline"
    ],
    "cloud": [
        "aws", "gcp", "azure", "cloud", "ec2", "s3", "gcs", "lambda",
        "cloud run", "sagemaker", "vertex ai", "bedrock", "azure ml"
    ],
    "docker": [
        "docker", "container", "containerization", "dockerfile",
        "docker compose", "container registry"
    ],
    "kubernetes": [
        "kubernetes", "k8s", "helm", "pod", "deployment manifest",
        "orchestration", "kubectl", "eks", "gke", "aks"
    ],
    "mlops": [
        "mlops", "mlflow", "kubeflow", "model registry", "model serving",
        "feature store", "experiment tracking", "model monitoring",
        "weights and biases", "wandb", "dvc", "bentoml", "triton"
    ],
    "engineering": [
        "microservices", "rest api", "grpc", "service mesh",
        "load balancing", "infrastructure as code", "terraform",
        "api gateway", "message queue", "event-driven"
    ],
    "incidents": [
        "incident", "on-call", "postmortem", "root cause",
        "debugging production", "production issues", "production incident",
        "oncall", "pagerduty", "runbook"
    ],
}

# ---------------------------------------------------------------------------
# Title Level Map — for career progression scoring
# Higher number = more senior. Used to track upward trajectory.
# ---------------------------------------------------------------------------
TITLE_LEVEL_MAP = {
    # Level 1 — Intern / Entry
    "intern": 1, "trainee": 1, "junior": 1, "fresher": 1,
    "graduate engineer": 1, "associate engineer": 1,
    # Level 2 — Individual Contributor (IC)
    "engineer": 2, "developer": 2, "analyst": 2, "scientist": 2,
    "researcher": 2, "specialist": 2, "associate": 2,
    # Level 3 — Senior IC
    "senior": 3, "sr.": 3, "sr ": 3, "level ii": 3, "ii ": 3,
    "mid-level": 3, "sde ii": 3, "swe ii": 3,
    # Level 4 — Lead / Tech Lead
    "lead": 4, "tech lead": 4, "technical lead": 4, "level iii": 4,
    "expert": 4, "member of technical staff": 4, "mts": 4,
    "sde iii": 4, "swe iii": 4,
    # Level 5 — Staff / Principal
    "staff": 5, "principal": 5, "founding engineer": 5,
    "distinguished": 5, "architect": 5,
    # Level 6 — Director / VP / C-Suite
    "director": 6, "vp": 6, "head of": 6, "chief": 6,
    "cto": 6, "vpe": 6, "fellow": 6,
}

# ---------------------------------------------------------------------------
# Technology Generations — for learning velocity scoring
# Higher number = more modern / cutting-edge. Sorted by era.
# ---------------------------------------------------------------------------
TECH_GENERATIONS = {
    # Gen 1 — Foundational (2010–2015)
    "python": 1, "java": 1, "sql": 1, "r": 1, "scala": 1,
    "hadoop": 1, "hive": 1, "mapreduce": 1,
    "sklearn": 1, "scikit-learn": 1, "scikit": 1,
    "numpy": 1, "pandas": 1, "matplotlib": 1, "scipy": 1,
    # Gen 2 — Deep Learning & Big Data (2015–2019)
    "tensorflow": 2, "keras": 2, "pytorch": 2, "theano": 2,
    "cnn": 2, "rnn": 2, "lstm": 2, "gru": 2,
    "word2vec": 2, "glove": 2, "fasttext": 2,
    "xgboost": 2, "lightgbm": 2, "catboost": 2,
    "spark": 2, "kafka": 2, "flink": 2, "airflow": 2,
    "redis": 2, "celery": 2, "rabbitmq": 2,
    # Gen 3 — Transformers & Containers (2018–2022)
    "bert": 3, "gpt": 3, "gpt-2": 3, "transformers": 3,
    "attention": 3, "hugging face": 3, "huggingface": 3,
    "docker": 3, "kubernetes": 3, "k8s": 3,
    "mlflow": 3, "kubeflow": 3, "dvc": 3,
    "faiss": 3, "elasticsearch": 3, "opensearch": 3,
    "sentence-transformers": 3, "sentence transformers": 3,
    "fastapi": 3, "triton": 3, "onnx": 3,
    # Gen 4 — Vector DBs & LLMs (2021–2023)
    "pinecone": 4, "weaviate": 4, "qdrant": 4, "milvus": 4, "chroma": 4,
    "pgvector": 4, "vespa": 4,
    "llm": 4, "llama": 4, "gpt-4": 4, "gpt-3.5": 4,
    "mistral": 4, "gemma": 4, "falcon": 4, "phi": 4, "qwen": 4,
    "rag": 4, "retrieval augmented": 4,
    "lora": 4, "qlora": 4, "peft": 4, "rlhf": 4, "dpo": 4, "sft": 4,
    "langchain": 4, "llamaindex": 4, "llama index": 4, "llama-index": 4,
    "bge": 4, "e5": 4, "colbert": 4, "splade": 4,
    "vector database": 4, "vector search": 4, "hybrid search": 4,
    "wandb": 4, "weights and biases": 4,
    # Gen 5 — Agentic AI & Inference Optimisation (2023–present)
    "agentic": 5, "multi-agent": 5, "langgraph": 5,
    "function calling": 5, "tool use": 5, "crewai": 5, "autogen": 5,
    "vllm": 5, "tgi": 5, "text generation inference": 5,
    "speculative decoding": 5, "flash attention": 5, "flashattention": 5,
    "mixtral": 5, "deepseek": 5, "claude": 5, "gemini": 5,
}

# ---------------------------------------------------------------------------
# Ownership Verbs — Tiered evidence strength (Phase 4 refinement)
# HIGH: clear authorship/leadership (1.0 each)  
# MED:  direct execution (0.5 each)
# LOW:  passive/supporting (−0.15 penalty each — generic filler language)
# ---------------------------------------------------------------------------
OWNERSHIP_VERBS_HIGH = [
    "architected", "designed", "spearheaded", "pioneered",
    "built end-to-end", "built from scratch", "led the development",
    "founded", "owned end-to-end", "established", "authored",
    "drove the", "led the design", "created from scratch",
]
OWNERSHIP_VERBS_MED = [
    "built", "implemented", "developed", "deployed", "scaled",
    "optimized", "owned", "created", "launched", "shipped",
    "delivered", "led", "engineered", "rewrote", "migrated",
    "designed",
]
OWNERSHIP_VERBS_LOW = [
    "worked on", "helped with", "assisted", "contributed to",
    "participated in", "was part of", "involved in", "supported",
    "used", "leveraged",
]
# Keep the original flat list for backwards compat
OWNERSHIP_VERBS = OWNERSHIP_VERBS_HIGH + OWNERSHIP_VERBS_MED

# ---------------------------------------------------------------------------
# Project Complexity Keywords — 5 independent dimensions (Phase 4)
# Each dimension uses density scoring: more hits = higher sub-score.
# Measures how large / hard the engineering work was, not just whether shipped.
# ---------------------------------------------------------------------------
PROJECT_COMPLEXITY_KEYWORDS = {
    "scale": [
        "billion", "billions of", "millions of requests", "millions of users",
        "petabyte", "terabyte", "internet scale", "large-scale", "massive scale",
        "web-scale", "global scale", "at scale"
    ],
    "distributed": [
        "distributed system", "distributed systems", "distributed computing",
        "distributed training", "distributed inference", "microservices",
        "service mesh", "event-driven", "streaming pipeline", "real-time pipeline",
        "stream processing", "message queue"
    ],
    "production_ai": [
        "production llm", "llm serving", "model serving at scale",
        "vector search at scale", "inference pipeline",
        "ml platform", "ai platform", "ai infrastructure",
        "rag pipeline", "production rag", "embedding pipeline"
    ],
    "high_throughput": [
        "high throughput", "low latency", "sub-millisecond",
        "thousands of requests", "high qps", "qps", "tps",
        "high availability", "99.9", "five nines", "real-time serving"
    ],
    "engineering_depth": [
        "end-to-end", "from scratch", "ground up",
        "built the entire", "led the design", "architected the",
        "designed the system", "owned the", "built and deployed"
    ],
}

# ---------------------------------------------------------------------------
# JD-Aligned Technologies — Phase 4 velocity bonus
# Technologies directly mentioned or implied in the Senior AI Engineer JD.
# These get a 1.5x weight multiplier inside learning velocity scoring.
# ---------------------------------------------------------------------------
JD_ALIGNED_TECHS = {
    # Retrieval / RAG (core JD requirement)
    "rag", "retrieval augmented", "dense retrieval", "hybrid search",
    "sentence-transformers", "sentence transformers", "bge", "e5",
    "colbert", "splade", "dpr",
    # Vector Databases
    "pinecone", "weaviate", "qdrant", "milvus", "chroma",
    "pgvector", "faiss", "vespa", "opensearch",
    # Ranking & Evaluation
    "learning to rank", "ndcg", "mrr", "bm25", "re-ranking",
    # LLM Fine-tuning
    "lora", "qlora", "peft", "rlhf", "dpo", "sft",
    "instruction tuning", "fine-tuning",
    # Agentic / Modern infra
    "langgraph", "llamaindex", "llama-index", "langchain",
    "vllm", "tgi", "ray", "speculative decoding",
    # Evaluation frameworks
    "ragas", "trulens", "deepeval",
}

# ---------------------------------------------------------------------------
# Domain Groups — for career consistency scoring
# ---------------------------------------------------------------------------
DOMAIN_GROUPS = {
    "ai_ml": [
        "ml engineer", "machine learning engineer", "ai engineer",
        "data scientist", "nlp engineer", "research engineer",
        "applied scientist", "deep learning engineer", "ai researcher",
        "ml researcher", "applied ml", "data science", "nlp scientist",
        "recommendation systems engineer", "search engineer",
        "computer vision", "speech engineer",
    ],
    "software_engineering": [
        "software engineer", "backend engineer", "platform engineer",
        "systems engineer", "full stack", "fullstack", "developer",
        "swe", "sde", "technical lead", "tech lead", "founding engineer",
        "staff engineer", "principal engineer", "senior engineer",
        "infrastructure engineer", "site reliability", "sre",
    ],
    "data_engineering": [
        "data engineer", "analytics engineer", "etl developer",
        "data architect", "bi developer", "business intelligence",
        "data analyst", "database administrator", "dba",
    ],
    "product_research": [
        "product manager", "program manager", "researcher",
        "ux", "ui", "product designer", "product analyst",
        "technical program manager", "tpm",
    ],
    "unrelated": [
        "marketing", "sales", "hr", "human resources", "accountant",
        "graphic designer", "content writer", "operations", "customer support",
        "civil engineer", "mechanical engineer", "electrical engineer",
        "teacher", "professor", "biology", "chemistry", "medical",
    ],
}

# ---------------------------------------------------------------------------
# Technology display names — for evidence extraction in reasoning
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
    "vllm": "vLLM",
    "langchain": "LangChain",
    "llamaindex": "LlamaIndex",
    "wandb": "W&B",
    "mlflow": "MLflow",
}

# ---------------------------------------------------------------------------
# Capability action phrases — for evidence-based reasoning generation
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

# ---------------------------------------------------------------------------
# Title Classification
# ---------------------------------------------------------------------------
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

TIER5_CAREER_KEYWORDS = [
    "recommendation system", "recommendation systems", "recommendation engine",
    "search engine", "search infrastructure", "information retrieval",
    "vector search", "semantic search", "retrieval engine", "ranking pipeline",
    "ranking algorithms", "candidate matching", "matching infrastructure",
    "retrieval and ranking", "rag pipeline", "rag systems"
]

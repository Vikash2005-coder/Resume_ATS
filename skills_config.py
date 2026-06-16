# skills_config.py
"""
Configuration of core skills, nice-to-have skills, and concept mapping
to support semantic skill matching in the ranking engine.
"""

# Core Concept categories mapping to specific skills in candidates' profiles
CORE_CONCEPTS = {
    "vector_databases": {
        "weight": 1.0,
        "keywords": [
            "milvus", "qdrant", "weaviate", "pinecone", "faiss", 
            "vector database", "vector databases", "vector search",
            "opensearch", "elasticsearch"
        ]
    },
    "retrieval_rag": {
        "weight": 1.0,
        "keywords": [
            "retrieval", "rag", "hybrid retrieval", "hybrid search",
            "dense retrieval", "bm25", "embeddings", "sentence-transformers",
            "sentence transformers", "bge", "e5", "retrieval-augmented generation",
            "retrieval augmented generation"
        ]
    },
    "ranking_evaluation": {
        "weight": 1.0,
        "keywords": [
            "ranking", "re-ranking", "re-ranker", "learning-to-rank", 
            "learning to rank", "ndcg", "mrr", "map", "evaluation framework",
            "evaluation frameworks", "ltr"
        ]
    },
    "llm_dl": {
        "weight": 0.8,
        "keywords": [
            "llm", "llms", "fine-tuning", "fine tuning", "lora", "qlora", 
            "peft", "transformers", "bert", "gpt", "gpt-4", "gpt-3.5",
            "pytorch", "tensorflow", "deep learning", "dl"
        ]
    },
    "programming_ml": {
        "weight": 0.8,
        "keywords": [
            "python", "machine learning", "ml", "applied ml", 
            "natural language processing", "nlp", "distributed systems",
            "inference optimization", "xgboost", "bento2t", "bentoml"
        ]
    }
}

# Synonyms for checking title match and mapping generic software engineering titles
AI_TITLE_KEYWORDS = [
    "ai engineer", "artificial intelligence engineer", "ml engineer", 
    "machine learning engineer", "nlp engineer", "nlp developer", 
    "applied ml scientist", "applied machine learning engineer", 
    "data scientist", "deep learning engineer", "staff ai engineer",
    "senior ai engineer", "senior ml engineer", "lead ml engineer",
    "lead ai engineer"
]

GENERIC_TECH_TITLES = [
    "software engineer", "backend engineer", "platform engineer", 
    "data engineer", "search engineer", "systems engineer", 
    "technical lead", "tech lead", "senior engineer", 
    "member of technical staff", "mts"
]

IRRELEVANT_TITLES = [
    "marketing manager", "operations manager", "accountant", 
    "graphic designer", "hr manager", "sales executive", 
    "customer support", "civil engineer", "mechanical engineer",
    "business analyst", "project manager", "scrum master"
]

# Text phrases in job descriptions to look for when identifying Tier-5 "hidden gem" candidates
TIER5_CAREER_KEYWORDS = [
    "recommendation system", "recommendation systems", "recommendation engine",
    "search engine", "search infrastructure", "information retrieval", 
    "vector search", "semantic search", "retrieval engine", "ranking pipeline",
    "ranking algorithms", "candidate matching", "matching infrastructure", 
    "retrieval and ranking", "rag pipeline", "rag systems"
]

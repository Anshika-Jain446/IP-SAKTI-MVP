from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
from typing import List
import json
import os
import re
import logging
from functools import lru_cache

import ollama


# =========================================================
# CONVERSATION API
# =========================================================

try:
    from conversation_api import router as conversation_router
except Exception as e:
    conversation_router = None
    logging.warning(
        "Conversation router unavailable: %r",
        e
    )


# =========================================================
# OPTIONAL BGE EMBEDDING ENGINE
# =========================================================

try:
    from embeddings import (
        semantic_search,
        EMBEDDING_MODEL,
        EMBEDDINGS_AVAILABLE,
    )

except Exception as e:

    EMBEDDINGS_AVAILABLE = False
    EMBEDDING_MODEL = None

    logging.warning(
        "Embedding engine unavailable: %r",
        e
    )

    def semantic_search(
        query,
        documents,
        top_k=5,
        min_similarity=0.35
    ):
        return []


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=os.getenv(
        "LOG_LEVEL",
        "INFO"
    ).upper(),

    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    )
)

logger = logging.getLogger(
    "ip-sakti"
)


# =========================================================
# CONFIGURATION
# =========================================================

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen2.5:1.5b"
)

# Single source of truth for API version.
#
# Default intentionally matches the earlier 0.7.x build.
# If you want another version, set:
#
# APP_VERSION=1.0.0
#
APP_VERSION = os.getenv(
    "APP_VERSION",
    "0.7.0"
)

TOP_K = max(
    1,
    int(
        os.getenv(
            "TOP_K",
            "5"
        )
    )
)

MIN_SEMANTIC_SIMILARITY = float(
    os.getenv(
        "MIN_SEMANTIC_SIMILARITY",
        "0.35"
    )
)

MAX_EVIDENCE_TEXT = max(
    500,
    int(
        os.getenv(
            "MAX_EVIDENCE_TEXT",
            "1800"
        )
    )
)

MAX_QUERY_LENGTH = max(
    500,
    int(
        os.getenv(
            "MAX_QUERY_LENGTH",
            "1800"
        )
    )
)


# =========================================================
# APP
# =========================================================

app = FastAPI(

    title="IP-SAKTI MVP API",

    version=APP_VERSION,

    description=(
        "Evidence-first IP / Traditional Knowledge / "
        "Access and Benefit Sharing assessment prototype."
    )
)


if conversation_router is not None:

    app.include_router(
        conversation_router
    )


# =========================================================
# INPUT MODEL
# =========================================================

class ProductInput(BaseModel):

    product_name: str = Field(
        ...,
        min_length=1,
        max_length=200
    )

    ingredients: List[str] = Field(
        default_factory=list,
        max_length=50
    )

    purpose: str = Field(
        ...,
        min_length=1,
        max_length=1000
    )

    product_type: str = Field(
        ...,
        min_length=1,
        max_length=200
    )

    jurisdiction: str = Field(
        default="India",
        max_length=200
    )

    based_on_traditional_knowledge: str = Field(
        default="unknown",
        max_length=50
    )

    @field_validator(
        "product_name",
        "purpose",
        "product_type",
        "jurisdiction"
    )
    @classmethod
    def clean_text(cls, value):

        value = str(
            value
        ).strip()

        if not value:
            raise ValueError(
                "Field cannot be empty."
            )

        return value

    @field_validator(
        "ingredients"
    )
    @classmethod
    def clean_ingredients(
        cls,
        values
    ):

        cleaned = []
        seen = set()

        for value in values or []:

            value = str(
                value
            ).strip()

            if not value:
                continue

            normalized = normalize_text(
                value
            )

            if (
                normalized
                and normalized not in seen
            ):

                cleaned.append(
                    value
                )

                seen.add(
                    normalized
                )

        return cleaned


# =========================================================
# CORPUS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

CORPUS_PATH = os.path.join(
    BASE_DIR,
    "data",
    "corpus.json"
)


@lru_cache(maxsize=1)
def _load_corpus():

    if not os.path.exists(
        CORPUS_PATH
    ):

        logger.error(
            "Corpus not found: %s",
            CORPUS_PATH
        )

        return []

    try:

        with open(
            CORPUS_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        if not isinstance(
            data,
            list
        ):

            logger.error(
                "corpus.json must contain a list."
            )

            return []

        data = [
            item
            for item in data
            if isinstance(
                item,
                dict
            )
        ]

        logger.info(
            "Loaded %d corpus chunks.",
            len(data)
        )

        return data

    except Exception as e:

        logger.exception(
            "Corpus loading failed: %r",
            e
        )

        return []


def load_corpus(
    force_reload=False
):

    if force_reload:

        _load_corpus.cache_clear()

    return _load_corpus()


# =========================================================
# NORMALIZATION
# =========================================================

def normalize_text(text):

    if text is None:
        return ""

    text = str(
        text
    ).lower().strip()

    replacements = {

        "traditional knowledge":
            "traditional_knowledge",

        "traditional-knowledge":
            "traditional_knowledge",

        "access and benefit sharing":
            "access_benefit_sharing",

        "access and benefit-sharing":
            "access_benefit_sharing",

        "benefit-sharing":
            "benefit_sharing",

        "benefit sharing":
            "benefit_sharing",

        "biological diversity":
            "biodiversity",

        "bio-diversity":
            "biodiversity",

        "biological resource":
            "biological_resource",

        "biological-resources":
            "biological_resource",

        "genetic resource":
            "genetic_resource",

        "genetic-resources":
            "genetic_resource",

        "geographical indications":
            "geographical_indication",

        "geographical indication":
            "geographical_indication",

        "trade mark":
            "trademark",

        "trade-mark":
            "trademark",

        "intellectual property":
            "intellectual_property",

        "prior art":
            "prior_art",

        "inventive step":
            "inventive_step",

        "traditional use":
            "traditional_use",

        "traditional medicine":
            "traditional_medicine",

        "aloe-vera":
            "aloe_vera",

        "curcuma longa":
            "curcuma_longa",

        "aloe barbadensis":
            "aloe_barbadensis"
    }

    for old, new in replacements.items():

        text = text.replace(
            old,
            new
        )

    text = re.sub(
        r"[^a-z0-9_\s-]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# STOPWORDS
# =========================================================

STOPWORDS = {

    "the",
    "and",
    "for",
    "with",
    "from",
    "this",
    "that",
    "are",
    "was",
    "were",
    "has",
    "have",
    "into",
    "your",
    "user",
    "product",
    "purpose",
    "based",
    "india",
    "yes",
    "true",
    "false",
    "unknown",

    "skin",
    "care",
    "cream",
    "cosmetic",
    "formulation",
    "type",

    "information",
    "use",
    "used",
    "using",
    "may",
    "can",
    "including",
    "under",

    "its",
    "their",
    "also",
    "such"
}


# =========================================================
# TOKENIZATION
# =========================================================

def tokenize(text):

    normalized = normalize_text(
        text
    )

    words = re.findall(
        r"\b[a-z][a-z0-9_]{2,}\b",
        normalized
    )

    return {
        word
        for word in words
        if word not in STOPWORDS
    }


def phrase_in_text(
    text,
    phrase
):

    normalized_text = normalize_text(
        text
    )

    normalized_phrase = normalize_text(
        phrase
    )

    if not normalized_phrase:
        return False

    pattern = (
        r"(?<![a-z0-9_])"
        + re.escape(
            normalized_phrase
        )
        + r"(?![a-z0-9_])"
    )

    return bool(
        re.search(
            pattern,
            normalized_text
        )
    )


# =========================================================
# DOMAIN KNOWLEDGE
# =========================================================

STRONG_DOMAIN_TERMS = {

    "IP": {

        "patent",
        "invention",
        "inventive_step",
        "novelty",
        "prior_art",
        "trademark",
        "copyright",
        "design",
        "geographical_indication",
        "intellectual_property"
    },

    "TK": {

        "traditional_knowledge",
        "traditional_use",
        "ayurveda",
        "ayurvedic",
        "herbal",
        "indigenous",
        "folk",
        "medicinal",
        "traditional_medicine",
        "ethnobotanical"
    },

    "ABS": {

        "biodiversity",
        "biological_resource",
        "genetic_resource",
        "access_benefit_sharing",
        "benefit_sharing",
        "biological",
        "genetic"
    }
}


# =========================================================
# DOMAIN ROUTING SIGNALS
# =========================================================

DOMAIN_ROUTING_SIGNALS = {

    "IP": {

        "patent",
        "invention",
        "trademark",
        "copyright",
        "design",
        "geographical_indication",
        "intellectual_property",
        "prior_art",
        "novelty",
        "inventive_step"
    },

    "TK": {

        "traditional_knowledge",
        "traditional_use",
        "ayurveda",
        "ayurvedic",
        "traditional_medicine",
        "folk",
        "indigenous",
        "ethnobotanical"
    },

    "ABS": {

        "biodiversity",
        "biological_resource",
        "genetic_resource",
        "access_benefit_sharing",
        "benefit_sharing",
        "genetic"
    }
}


# =========================================================
# INGREDIENT ALIASES
# =========================================================

INGREDIENT_ALIASES = {

    "turmeric": {

        "turmeric",
        "curcuma",
        "curcuma_longa",
        "curcumin"
    },

    "curcuma": {

        "turmeric",
        "curcuma",
        "curcuma_longa",
        "curcumin"
    },

    "curcuma longa": {

        "turmeric",
        "curcuma",
        "curcuma_longa",
        "curcumin"
    },

    "aloe": {

        "aloe",
        "aloe_vera",
        "aloe_barbadensis",
        "barbadensis"
    },

    "aloe vera": {

        "aloe",
        "aloe_vera",
        "aloe_barbadensis",
        "barbadensis"
    }
}


def get_ingredient_aliases(
    ingredient
):

    normalized = normalize_text(
        ingredient
    )

    aliases = {
        normalized
    }

    if normalized in INGREDIENT_ALIASES:

        aliases.update(
            INGREDIENT_ALIASES[
                normalized
            ]
        )

    for key, values in INGREDIENT_ALIASES.items():

        if (
            phrase_in_text(
                normalized,
                key
            )
            or normalized in values
        ):

            aliases.update(
                values
            )

    return {
        normalize_text(
            value
        )
        for value in aliases
        if value
    }


# =========================================================
# DOCUMENT HELPERS
# =========================================================

def get_document_text(
    item
):

    text = item.get(
        "page_content",
        ""
    )

    if not text:

        text = item.get(
            "text",
            ""
        )

    return str(
        text or ""
    )


def get_document_id(
    item
):

    return str(
        item.get(
            "id",
            item.get(
                "chunk_id",
                ""
            )
        )
        or ""
    ).strip()


def get_document_domain(
    item
):

    metadata = item.get(
        "metadata",
        {}
    ) or {}

    domain = item.get(
        "domain",
        metadata.get(
            "domain",
            "UNKNOWN"
        )
    )

    return str(
        domain or "UNKNOWN"
    ).upper().strip()


def get_document_source(
    item
):

    metadata = item.get(
        "metadata",
        {}
    ) or {}

    return str(
        item.get(
            "source",
            metadata.get(
                "filename",
                "Unknown source"
            )
        )
        or "Unknown source"
    )


def get_document_page(
    item
):

    metadata = item.get(
        "metadata",
        {}
    ) or {}

    return item.get(
        "page",
        metadata.get(
            "page"
        )
    )


def get_document_chunk(
    item
):

    metadata = item.get(
        "metadata",
        {}
    ) or {}

    return item.get(
        "chunk",
        metadata.get(
            "chunk"
        )
    )


# =========================================================
# DOMAIN DETECTION
# =========================================================

def detect_domains(
    product
):

    text = normalize_text(
        " ".join([
            product.product_name,
            " ".join(
                product.ingredients
            ),
            product.purpose,
            product.product_type,
            product.jurisdiction
        ])
    )

    domains = []

    tk_answer = normalize_text(
        product.based_on_traditional_knowledge
    )

    if tk_answer in {
        "yes",
        "true",
        "y",
        "1"
    }:

        domains.append(
            "TK"
        )

    for domain, signals in (
        DOMAIN_ROUTING_SIGNALS.items()
    ):

        if any(
            phrase_in_text(
                text,
                signal
            )
            for signal in signals
        ):

            if domain not in domains:

                domains.append(
                    domain
                )

    # Known biological-resource ingredients
    # are routing signals ONLY.
    #
    # They do NOT constitute ABS evidence.
    ingredient_signals = {

        "turmeric",
        "curcuma",
        "curcuma_longa",
        "curcumin",
        "aloe",
        "aloe_vera",
        "aloe_barbadensis"
    }

    ingredient_text = normalize_text(
        " ".join(
            product.ingredients
        )
    )

    if any(
        phrase_in_text(
            ingredient_text,
            signal
        )
        for signal in ingredient_signals
    ):

        if "ABS" not in domains:

            domains.append(
                "ABS"
            )

    # IP is always a baseline
    # assessment domain.
    if "IP" not in domains:

        domains.append(
            "IP"
        )

    return domains


# =========================================================
# PRODUCT CLASSIFICATION
# =========================================================

def classify_product(
    product
):

    tk_value = normalize_text(
        product.based_on_traditional_knowledge
    )

    if tk_value in {
        "yes",
        "true",
        "y",
        "1"
    }:

        tk_status = "YES"

    elif tk_value in {
        "no",
        "false",
        "n",
        "0"
    }:

        tk_status = "NO"

    else:

        tk_status = "UNKNOWN"

    return {

        "product_type":
            product.product_type,

        "traditional_knowledge":
            product.based_on_traditional_knowledge,

        "traditional_knowledge_status":
            tk_status,

        "jurisdiction":
            product.jurisdiction,

        "classification_note":
            (
                "Prototype classification based on "
                "user-provided information. This is a "
                "routing signal, not a legal determination."
            )
    }


# =========================================================
# SEARCH QUERY
# =========================================================

def build_search_query(
    product,
    domains
):

    parts = [

        product.product_name,

        " ".join(
            product.ingredients
        ),

        product.purpose,

        product.product_type
    ]

    if "TK" in domains:

        parts.append(
            "traditional knowledge traditional use"
        )

    if "ABS" in domains:

        parts.append(
            "biological resource biodiversity "
            "access benefit sharing"
        )

    if "IP" in domains:

        parts.append(
            "intellectual property patent trademark"
        )

    query = " ".join(
        str(part).strip()
        for part in parts
        if str(part).strip()
    )

    return query[
        :MAX_QUERY_LENGTH
    ]


# =========================================================
# INGREDIENT MATCHING
# =========================================================

def find_ingredient_matches(
    text,
    ingredients
):

    normalized_text = normalize_text(
        text
    )

    matches = set()

    for ingredient in ingredients:

        aliases = get_ingredient_aliases(
            ingredient
        )

        for alias in aliases:

            if phrase_in_text(
                normalized_text,
                alias
            ):

                matches.add(
                    alias
                )

    return sorted(
        matches
    )


# =========================================================
# DOMAIN MATCHING
# =========================================================

def find_domain_matches(
    text,
    domains
):

    normalized_text = normalize_text(
        text
    )

    result = {}

    for domain in domains:

        matched = []

        for term in STRONG_DOMAIN_TERMS.get(
            domain,
            set()
        ):

            if phrase_in_text(
                normalized_text,
                term
            ):

                matched.append(
                    term
                )

        result[
            domain
        ] = sorted(
            set(
                matched
            )
        )

    return result


def get_supported_domains_from_item(
    item,
    detected_domains
):
    """
    CRITICAL DISTINCTION:

    document domain metadata != actual evidence.

    A corpus item tagged IP is not evidence for IP
    unless the text itself contains relevant IP terms.

    Ingredient matches are intentionally NOT treated
    as domain evidence.
    """

    domain_matches = item.get(
        "domain_matches",
        {}
    ) or {}

    supported = []

    for domain in detected_domains:

        terms = domain_matches.get(
            domain,
            []
        )

        if terms:

            supported.append(
                domain
            )

    return supported


# =========================================================
# KEYWORD SCORING
# =========================================================

def calculate_keyword_score(
    query_tokens,
    document_text,
    document_domain,
    domains,
    ingredients
):

    normalized_text = normalize_text(
        document_text
    )

    document_tokens = tokenize(
        normalized_text
    )

    matched_terms = (
        query_tokens.intersection(
            document_tokens
        )
    )

    score = min(
        len(
            matched_terms
        ) * 0.45,
        5.0
    )

    # Metadata alignment contributes to ranking,
    # but NOT to evidence validation.
    domain_match = (
        document_domain in domains
    )

    if domain_match:

        score += 2.5

    ingredient_matches = (
        find_ingredient_matches(
            normalized_text,
            ingredients
        )
    )

    # Retrieval signal only.
    score += min(
        len(
            ingredient_matches
        ) * 4.0,
        12.0
    )

    domain_matches = (
        find_domain_matches(
            normalized_text,
            domains
        )
    )

    for domain in domains:

        score += min(
            len(
                domain_matches.get(
                    domain,
                    []
                )
            ) * 1.25,
            5.0
        )

    return {

        "score":
            round(
                score,
                4
            ),

        "matched_terms":
            sorted(
                matched_terms
            )[:30],

        "ingredient_matches":
            ingredient_matches,

        "domain_matches":
            domain_matches,

        "domain_match":
            domain_match
    }


# =========================================================
# EVIDENCE QUALITY
# =========================================================

def calculate_evidence_quality(
    item
):

    ingredient_count = len(
        item.get(
            "ingredient_matches",
            []
        )
    )

    domain_match = bool(
        item.get(
            "domain_match",
            False
        )
    )

    domain_term_count = sum(
        len(values)
        for values in (
            item.get(
                "domain_matches",
                {}
            ) or {}
        ).values()
    )

    try:

        similarity = float(
            item.get(
                "similarity",
                0.0
            ) or 0.0
        )

    except (
        TypeError,
        ValueError
    ):

        similarity = 0.0

    quality = 0.0

    quality += min(
        ingredient_count * 0.20,
        0.40
    )

    quality += (
        0.25
        if domain_match
        else 0.0
    )

    quality += min(
        domain_term_count * 0.05,
        0.20
    )

    quality += (
        min(
            max(
                similarity,
                0.0
            ),
            1.0
        ) * 0.15
    )

    return round(
        min(
            quality,
            1.0
        ),
        4
    )


# =========================================================
# CORPUS SEARCH
# =========================================================

def search_corpus(
    query,
    domains,
    ingredients=None,
    top_k=5
):

    if ingredients is None:

        ingredients = []

    query = str(
        query or ""
    ).strip()

    if not query:

        return []

    domains = [
        str(domain).upper().strip()
        for domain in domains
        if str(domain).strip()
    ]

    corpus = load_corpus()

    if not corpus:

        logger.warning(
            "Empty corpus."
        )

        return []

    query_tokens = tokenize(
        query
    )

    # -----------------------------------------------------
    # 1. DOMAIN FILTER
    # -----------------------------------------------------

    candidates = []

    for item in corpus:

        text = get_document_text(
            item
        )

        if not text:
            continue

        document_domain = (
            get_document_domain(
                item
            )
        )

        if (
            document_domain
            in domains
        ):

            candidates.append(
                item
            )

    logger.info(
        "Domain-aligned candidates: %d",
        len(candidates)
    )

    # -----------------------------------------------------
    # 2. KEYWORD SEARCH
    # -----------------------------------------------------

    keyword_results = {}

    for index, item in enumerate(
        candidates
    ):

        text = get_document_text(
            item
        )

        document_domain = (
            get_document_domain(
                item
            )
        )

        scoring = calculate_keyword_score(

            query_tokens=query_tokens,

            document_text=text,

            document_domain=document_domain,

            domains=domains,

            ingredients=ingredients
        )

        has_ingredient_evidence = bool(
            scoring[
                "ingredient_matches"
            ]
        )

        has_domain_evidence = any(

            scoring[
                "domain_matches"
            ].get(
                domain,
                []
            )

            for domain in domains
        )

        # HARD RELEVANCE GATE.
        #
        # Generic keyword overlap alone
        # is NOT enough.
        if not (
            has_ingredient_evidence
            or (
                scoring[
                    "domain_match"
                ]
                and has_domain_evidence
            )
        ):

            continue

        item_id = get_document_id(
            item
        )

        if not item_id:

            item_id = (
                f"{get_document_source(item)}-"
                f"{get_document_page(item)}-"
                f"{get_document_chunk(item)}-"
                f"{index}"
            )

        keyword_results[
            item_id
        ] = {

            "id":
                item_id,

            "keyword_score":
                scoring[
                    "score"
                ],

            "matched_terms":
                scoring[
                    "matched_terms"
                ],

            "ingredient_matches":
                scoring[
                    "ingredient_matches"
                ],

            "domain_matches":
                scoring[
                    "domain_matches"
                ],

            "domain_match":
                scoring[
                    "domain_match"
                ],

            "domain":
                document_domain,

            "source":
                get_document_source(
                    item
                ),

            "page":
                get_document_page(
                    item
                ),

            "chunk":
                get_document_chunk(
                    item
                ),

            "text":
                text[
                    :MAX_EVIDENCE_TEXT
                ]
        }

    logger.info(
        "Keyword evidence candidates: %d",
        len(
            keyword_results
        )
    )

    # -----------------------------------------------------
    # 3. BGE SEMANTIC SEARCH
    # -----------------------------------------------------

    semantic_results = []

    if (
        EMBEDDINGS_AVAILABLE
        and candidates
    ):

        semantic_documents = []

        for index, item in enumerate(
            candidates
        ):

            item_id = get_document_id(
                item
            )

            if not item_id:

                item_id = (
                    f"{get_document_source(item)}-"
                    f"{get_document_page(item)}-"
                    f"{get_document_chunk(item)}-"
                    f"{index}"
                )

            semantic_documents.append({

                "id":
                    item_id,

                "domain":
                    get_document_domain(
                        item
                    ),

                "source":
                    get_document_source(
                        item
                    ),

                "page":
                    get_document_page(
                        item
                    ),

                "chunk":
                    get_document_chunk(
                        item
                    ),

                "page_content":
                    get_document_text(
                        item
                    )
            })

        try:

            semantic_results = semantic_search(

                query=query,

                documents=semantic_documents,

                top_k=max(
                    top_k * 3,
                    10
                ),

                min_similarity=(
                    MIN_SEMANTIC_SIMILARITY
                )
            )

            logger.info(
                "BGE candidates: %d",
                len(
                    semantic_results
                )
            )

        except Exception as e:

            logger.exception(
                "BGE search failed: %r",
                e
            )

            semantic_results = []

    semantic_by_id = {}

    for item in semantic_results:

        item_id = str(
            item.get(
                "id",
                ""
            )
        ).strip()

        if item_id:

            semantic_by_id[
                item_id
            ] = item

    # -----------------------------------------------------
    # 4. MERGE
    # -----------------------------------------------------

    merged = {}

    all_ids = (
        set(
            keyword_results.keys()
        )
        |
        set(
            semantic_by_id.keys()
        )
    )

    for item_id in all_ids:

        keyword_item = (
            keyword_results.get(
                item_id
            )
        )

        semantic_item = (
            semantic_by_id.get(
                item_id
            )
        )

        if keyword_item:

            result = dict(
                keyword_item
            )

        elif semantic_item:

            result = {

                "id":
                    item_id,

                "keyword_score":
                    0.0,

                "matched_terms":
                    [],

                "ingredient_matches":
                    [],

                "domain_matches":
                    {},

                "domain_match":
                    True,

                "domain":
                    str(
                        semantic_item.get(
                            "domain",
                            "UNKNOWN"
                        )
                    ).upper(),

                "source":
                    semantic_item.get(
                        "source",
                        "Unknown source"
                    ),

                "page":
                    semantic_item.get(
                        "page"
                    ),

                "chunk":
                    semantic_item.get(
                        "chunk"
                    ),

                "text":
                    str(
                        semantic_item.get(
                            "page_content",
                            semantic_item.get(
                                "text",
                                ""
                            )
                        )
                    )[
                        :MAX_EVIDENCE_TEXT
                    ]
            }

        else:

            continue

        try:

            similarity = float(
                semantic_item.get(
                    "similarity",
                    0.0
                )
            ) if semantic_item else 0.0

        except (
            TypeError,
            ValueError
        ):

            similarity = 0.0

        result[
            "similarity"
        ] = (
            round(
                similarity,
                4
            )
            if semantic_item
            else None
        )

        # Recalculate evidence from FINAL text.
        result[
            "ingredient_matches"
        ] = sorted(
            set(
                result.get(
                    "ingredient_matches",
                    []
                )
            )
            |
            set(
                find_ingredient_matches(
                    result.get(
                        "text",
                        ""
                    ),
                    ingredients
                )
            )
        )

        result[
            "domain_matches"
        ] = find_domain_matches(

            result.get(
                "text",
                ""
            ),

            domains
        )

        has_ingredient_evidence = bool(
            result[
                "ingredient_matches"
            ]
        )

        has_domain_evidence = any(

            result[
                "domain_matches"
            ].get(
                domain,
                []
            )

            for domain in domains
        )

        domain_match = (
            result.get(
                "domain",
                "UNKNOWN"
            )
            in domains
        )

        # -------------------------------------------------
        # CRITICAL EVIDENCE GATE
        #
        # Semantic similarity by itself is NOT evidence.
        #
        # Metadata domain alignment by itself is NOT
        # evidence.
        # -------------------------------------------------

        if not (
            has_ingredient_evidence
            or (
                domain_match
                and has_domain_evidence
            )
        ):

            continue

        result[
            "domain_match"
        ] = domain_match

        keyword_score = float(
            result.get(
                "keyword_score",
                0.0
            )
        )

        semantic_scaled = (
            similarity * 10.0
        )

        ingredient_bonus = min(
            len(
                result[
                    "ingredient_matches"
                ]
            ) * 2.5,
            7.5
        )

        domain_bonus = (
            2.5
            if domain_match
            else 0.0
        )

        if (
            semantic_item
            and keyword_item
        ):

            combined_score = (

                0.55
                * semantic_scaled

                +

                0.25
                * min(
                    keyword_score,
                    10.0
                )

                +

                ingredient_bonus

                +

                domain_bonus
            )

            retrieval_type = (
                "HYBRID"
            )

        elif semantic_item:

            combined_score = (

                0.75
                * semantic_scaled

                +

                ingredient_bonus

                +

                domain_bonus
            )

            retrieval_type = (
                "BGE_SEMANTIC"
            )

        else:

            combined_score = (

                keyword_score

                +

                ingredient_bonus

                +

                domain_bonus
            )

            retrieval_type = (
                "KEYWORD"
            )

        result[
            "score"
        ] = round(
            combined_score,
            4
        )

        result[
            "retrieval_type"
        ] = retrieval_type

        result[
            "evidence_quality"
        ] = calculate_evidence_quality(
            result
        )

        evidence_basis = []

        if result[
            "ingredient_matches"
        ]:

            evidence_basis.append(
                "ingredient_match"
            )

        if any(

            result[
                "domain_matches"
            ].get(
                domain,
                []
            )

            for domain in domains
        ):

            evidence_basis.append(
                "domain_term_match"
            )

        if semantic_item:

            evidence_basis.append(
                "semantic_similarity"
            )

        if result[
            "matched_terms"
        ]:

            evidence_basis.append(
                "keyword_overlap"
            )

        result[
            "evidence_basis"
        ] = evidence_basis

        # NEW:
        # Explicitly record which detected domains
        # this evidence actually supports.
        result[
            "supported_domains"
        ] = get_supported_domains_from_item(
            result,
            domains
        )

        merged[
            item_id
        ] = result

    # -----------------------------------------------------
    # 5. RANK
    # -----------------------------------------------------

    results = list(
        merged.values()
    )

    results.sort(

        key=lambda item: (

            float(
                item.get(
                    "score",
                    0.0
                )
            ),

            float(
                item.get(
                    "evidence_quality",
                    0.0
                )
            ),

            len(
                item.get(
                    "ingredient_matches",
                    []
                )
            ),

            float(
                item.get(
                    "similarity",
                    0.0
                )
                or 0.0
            )
        ),

        reverse=True
    )

    # -----------------------------------------------------
    # 6. SOURCE DIVERSITY
    # -----------------------------------------------------

    selected = []

    source_counts = {}

    for item in results:

        source = item.get(
            "source",
            "Unknown source"
        )

        if source_counts.get(
            source,
            0
        ) >= 3:

            continue

        selected.append(
            item
        )

        source_counts[
            source
        ] = (
            source_counts.get(
                source,
                0
            )
            + 1
        )

        if len(
            selected
        ) >= top_k:

            break

    # Fill remaining slots.
    if len(
        selected
    ) < top_k:

        selected_ids = {
            item[
                "id"
            ]
            for item in selected
        }

        for item in results:

            if item[
                "id"
            ] in selected_ids:

                continue

            selected.append(
                item
            )

            if len(
                selected
            ) >= top_k:

                break

    return selected[
        :top_k
    ]


# =========================================================
# DOMAIN EVIDENCE SUMMARY
# =========================================================

def build_domain_evidence_summary(
    evidence,
    domains
):

    summary = {}

    for domain in domains:

        domain_items = []

        for item in evidence:

            domain_terms = (
                item.get(
                    "domain_matches",
                    {}
                ).get(
                    domain,
                    []
                )
            )

            if domain_terms:

                domain_items.append(
                    item
                )

        sources = sorted(
            set(
                item.get(
                    "source",
                    "Unknown source"
                )
                for item in domain_items
            )
        )

        terms = sorted(
            set(
                term
                for item in domain_items
                for term in item.get(
                    "domain_matches",
                    {}
                ).get(
                    domain,
                    []
                )
            )
        )

        ingredient_items = [
            item
            for item in evidence
            if item.get(
                "ingredient_matches"
            )
        ]

        summary[
            domain
        ] = {

            "supported":
                bool(domain_items),

            "evidence_count":
                len(domain_items),

            "sources":
                sources,

            "matched_domain_terms":
                terms,

            "ingredient_evidence_count":
                len(ingredient_items)
                if domain == "ABS"
                else 0
        }

    return summary


# =========================================================
# EVIDENCE VALIDATION
# =========================================================

def validate_evidence(
    evidence,
    domains
):

    domain_summary = (
        build_domain_evidence_summary(
            evidence,
            domains
        )
    )

    supported_domains = [
        domain
        for domain in domains
        if domain_summary.get(
            domain,
            {}
        ).get(
            "supported",
            False
        )
    ]

    unsupported_domains = [
        domain
        for domain in domains
        if domain not in supported_domains
    ]

    strong_items = 0
    ingredient_items = 0

    # IMPORTANT:
    # This is now actual domain evidence,
    # not metadata alignment.
    domain_evidence_items = 0

    for item in evidence:

        if float(
            item.get(
                "score",
                0.0
            )
        ) >= 8.0:

            strong_items += 1

        if item.get(
            "ingredient_matches"
        ):

            ingredient_items += 1

        if item.get(
            "supported_domains"
        ):

            domain_evidence_items += 1

    if not evidence:

        status = "UNSUPPORTED"

        message = (
            "No sufficiently relevant evidence "
            "was found for the detected domain(s) "
            "in the current corpus."
        )

    elif not supported_domains:

        status = "UNSUPPORTED"

        message = (
            "Retrieved material did not provide "
            "direct domain evidence for the detected "
            "domain(s)."
        )

    elif unsupported_domains:

        # NEW:
        # Mixed-domain result.
        #
        # Example:
        # detected = IP, TK, ABS
        # supported = IP
        # unsupported = TK, ABS
        status = "PARTIAL"

        message = (
            "Evidence was found for only a subset "
            "of the detected domain(s). Unsupported "
            "domains must not be inferred from the "
            "available evidence."
        )

    elif (
        strong_items >= 1
        and domain_evidence_items >= 1
    ):

        status = "EVIDENCE_FOUND"

        message = (
            "Relevant domain-specific evidence was "
            "found. This is retrieval evidence only "
            "and is not a legal conclusion."
        )

    else:

        status = "WEAK"

        message = (
            "Some domain-specific evidence was found, "
            "but manual verification is required."
        )

    return {

        "status":
            status,

        "message":
            message,

        "domains":
            domains,

        "supported_domains":
            supported_domains,

        "unsupported_domains":
            unsupported_domains,

        "evidence_count":
            len(evidence),

        "strong_evidence_count":
            strong_items,

        "ingredient_evidence_count":
            ingredient_items,

        # FIXED SEMANTICS:
        # This count is now actual textual domain evidence.
        "domain_aligned_count":
            domain_evidence_items,

        "domain_evidence_count":
            domain_evidence_items,

        "domain_evidence":
            domain_summary
    }


# =========================================================
# LLM JSON HELPERS
# =========================================================

def parse_json_object(
    content
):

    if isinstance(
        content,
        dict
    ):

        return content

    if not isinstance(
        content,
        str
    ):

        return None

    content = content.strip()

    try:

        parsed = json.loads(
            content
        )

        if isinstance(
            parsed,
            dict
        ):

            return parsed

    except json.JSONDecodeError:

        pass

    match = re.search(
        r"\{.*\}",
        content,
        flags=re.DOTALL
    )

    if match:

        try:

            parsed = json.loads(
                match.group(
                    0
                )
            )

            if isinstance(
                parsed,
                dict
            ):

                return parsed

        except json.JSONDecodeError:

            pass

    return None


def normalize_llm_analysis(
    data
):

    if not isinstance(
        data,
        dict
    ):

        data = {}

    domain_analysis = data.get(
        "domain_analysis",
        {}
    )

    if not isinstance(
        domain_analysis,
        dict
    ):

        domain_analysis = {}

    risks = data.get(
        "risks",
        []
    )

    if not isinstance(
        risks,
        list
    ):

        risks = [
            str(risks)
        ]

    verification = data.get(
        "recommended_verification",
        []
    )

    if not isinstance(
        verification,
        list
    ):

        verification = [
            str(verification)
        ]

    return {

        "summary":
            str(
                data.get(
                    "summary",
                    "Insufficient evidence."
                )
            ),

        "domain_analysis": {

            "IP":
                str(
                    domain_analysis.get(
                        "IP",
                        "Insufficient evidence."
                    )
                ),

            "TK":
                str(
                    domain_analysis.get(
                        "TK",
                        "Insufficient evidence."
                    )
                ),

            "ABS":
                str(
                    domain_analysis.get(
                        "ABS",
                        "Insufficient evidence."
                    )
                )
        },

        "evidence_interpretation":
            str(
                data.get(
                    "evidence_interpretation",
                    "Insufficient evidence."
                )
            ),

        "risks": [
            str(value)
            for value in risks[:10]
        ],

        "recommended_verification": [
            str(value)
            for value in verification[:10]
        ],

        "limitations":
            str(
                data.get(
                    "limitations",
                    "Insufficient evidence."
                )
            )
    }


# =========================================================
# LLM LEGAL REFERENCE GUARD
# =========================================================

def extract_legal_references(
    text
):

    if not text:
        return set()

    patterns = [

        r"\b[A-Z][A-Za-z ]+\s+Act(?:,\s*\d{4})?",

        r"\bSection\s+\d+[A-Za-z]?\b",

        r"\bSections\s+[\d,\sand-]+\b",

        r"\bRegulation\s+\d+[A-Za-z]?\b",

        r"\bArticle\s+\d+[A-Za-z]?\b",

        r"\bRule\s+\d+[A-Za-z]?\b"
    ]

    matches = set()

    for pattern in patterns:

        for match in re.findall(
            pattern,
            text,
            flags=re.IGNORECASE
        ):

            matches.add(
                normalize_text(
                    match
                )
            )

    return matches


def collect_evidence_legal_references(
    evidence
):

    references = set()

    for item in evidence:

        text = item.get(
            "text",
            ""
        )

        references.update(
            extract_legal_references(
                text
            )
        )

    return references


def sanitize_unsupported_legal_references(
    analysis,
    evidence
):

    if not isinstance(
        analysis,
        dict
    ):

        return analysis

    evidence_refs = (
        collect_evidence_legal_references(
            evidence
        )
    )

    if not evidence_refs:

        evidence_refs = set()

    text_fields = [
        "summary",
        "evidence_interpretation",
        "limitations"
    ]

    for field in text_fields:

        value = analysis.get(
            field,
            ""
        )

        if not isinstance(
            value,
            str
        ):

            continue

        refs = extract_legal_references(
            value
        )

        unsupported = (
            refs - evidence_refs
        )

        if unsupported:

            analysis[field] = (
                "Insufficient evidence. "
                "The model introduced a legal reference "
                "that was not present in the retrieved "
                "evidence."
            )

    risks = analysis.get(
        "risks",
        []
    )

    cleaned_risks = []

    for risk in risks:

        risk = str(
            risk
        )

        refs = extract_legal_references(
            risk
        )

        if refs - evidence_refs:

            cleaned_risks.append(
                "Insufficient evidence."
            )

        else:

            cleaned_risks.append(
                risk
            )

    analysis[
        "risks"
    ] = cleaned_risks[:10]

    verification = analysis.get(
        "recommended_verification",
        []
    )

    cleaned_verification = []

    for recommendation in verification:

        recommendation = str(
            recommendation
        )

        refs = extract_legal_references(
            recommendation
        )

        if refs - evidence_refs:

            cleaned_verification.append(
                "Verify the applicable legal framework "
                "using appropriate authoritative sources."
            )

        else:

            cleaned_verification.append(
                recommendation
            )

    analysis[
        "recommended_verification"
    ] = cleaned_verification[:10]

    return analysis


# =========================================================
# LLM REASONING
# =========================================================

def generate_llm_reasoning(
    product,
    domains,
    evidence,
    validation
):

    if not evidence:

        return {

            "status":
                "NOT_RUN",

            "message":
                (
                    "LLM reasoning was not run because "
                    "no sufficiently relevant evidence "
                    "was retrieved."
                )
        }

    domain_summary = (
        validation.get(
            "domain_evidence",
            {}
        )
    )

    supported_domains = (
        validation.get(
            "supported_domains",
            []
        )
    )

    unsupported_domains = (
        validation.get(
            "unsupported_domains",
            []
        )
    )

    evidence_blocks = []

    for index, item in enumerate(
        evidence[:5],
        start=1
    ):

        evidence_blocks.append(
            f"""
EVIDENCE {index}

Source:
{item.get("source", "Unknown")}

Domain metadata:
{item.get("domain", "UNKNOWN")}

Supported domains from TEXT:
{", ".join(item.get("supported_domains", [])) or "None"}

Retrieval type:
{item.get("retrieval_type", "UNKNOWN")}

Similarity:
{item.get("similarity", "N/A")}

Retrieval score:
{item.get("score", "N/A")}

Evidence quality:
{item.get("evidence_quality", "N/A")}

Page:
{item.get("page", "N/A")}

Chunk:
{item.get("chunk", "N/A")}

Matched terms:
{", ".join(item.get("matched_terms", []))}

Ingredient matches:
{", ".join(item.get("ingredient_matches", []))}

Domain matches:
{json.dumps(item.get("domain_matches", {}))}

Evidence basis:
{", ".join(item.get("evidence_basis", []))}

Evidence text:
{item.get("text", "")}
"""
        )

    evidence_text = "\n".join(
        evidence_blocks
    )

    prompt = f"""
You are the evidence-grounded reasoning engine
for IP-SAKTI.

IP-SAKTI is an Indian IP / Traditional Knowledge /
Access and Benefit Sharing assessment prototype.

You are NOT a lawyer.

Do not provide legal advice.

Do not invent:
- laws
- sections
- regulations
- cases
- government decisions
- databases
- facts
- obligations
- legal mechanisms

=========================================================
ABSOLUTE GROUNDING RULE
=========================================================

You may reason ONLY from:

1. Product information
2. Retrieved evidence below
3. Explicit evidence metadata below

If evidence does not establish something,
say exactly:

"Insufficient evidence."

Do NOT use your general knowledge to fill gaps.

=========================================================
CRITICAL DOMAIN RULE
=========================================================

A document's DOMAIN METADATA is NOT sufficient evidence.

For example:

Domain metadata = IP

does NOT mean:

"IP evidence exists."

IP evidence exists only when the evidence TEXT contains
relevant IP terms shown in "Domain matches".

Likewise:

TK metadata != TK evidence

ABS metadata != ABS evidence

Ingredient match != ABS obligation

Ingredient match != TK evidence

=========================================================
DETECTED DOMAINS
=========================================================

{", ".join(domains)}

Domains with direct evidence:

{", ".join(supported_domains) or "None"}

Domains WITHOUT direct evidence:

{", ".join(unsupported_domains) or "None"}

Domain evidence summary:

{json.dumps(domain_summary, indent=2)}

=========================================================
PRODUCT
=========================================================

Product name:
{product.product_name}

Ingredients:
{", ".join(product.ingredients) or "None provided"}

Purpose:
{product.purpose}

Product type:
{product.product_type}

Jurisdiction:
{product.jurisdiction}

Traditional knowledge answer:
{product.based_on_traditional_knowledge}

=========================================================
RETRIEVED EVIDENCE
=========================================================

{evidence_text}

=========================================================
TASK
=========================================================

Analyze the evidence conservatively.

For EVERY detected domain:

- If that domain has direct evidence, summarize only
  what that evidence supports.

- If that domain has no direct evidence, write:
  "Insufficient evidence."

If only IP evidence exists while TK and ABS are detected,
DO NOT claim that TK or ABS are supported.

If an ingredient appears in evidence, treat that only as
ingredient evidence.

Do NOT infer an ABS obligation solely from an ingredient.

Do NOT infer traditional knowledge solely because something
is herbal, medicinal, indigenous, or an ingredient appears.

Do NOT infer:

- novelty
- patentability
- ownership
- infringement
- compliance
- liability
- legal obligation
- legal violation

unless directly supported by the retrieved evidence.

=========================================================
LEGAL REFERENCE RULE
=========================================================

You MUST NOT name an Act, Section, Regulation, Rule,
Article, case, or other specific legal authority unless
that exact legal reference appears in the retrieved
evidence text.

For example, if the evidence does not contain
"TKDL Act, 2005", you MUST NOT recommend it.

If a legal authority is not present in the evidence,
say:

"Insufficient evidence."

=========================================================
OUTPUT
=========================================================

Return valid JSON with exactly these fields:

{{
  "summary": "...",

  "domain_analysis": {{
    "IP": "...",
    "TK": "...",
    "ABS": "..."
  }},

  "evidence_interpretation": "...",

  "risks": [
    "...",
    "..."
  ],

  "recommended_verification": [
    "...",
    "...",
    "..."
  ],

  "limitations": "..."
}}

=========================================================
FINAL RULE
=========================================================

Evidence is more important than assumptions.

Never convert retrieval signals into legal conclusions.
"""

    try:

        response = ollama.chat(

            model=OLLAMA_MODEL,

            messages=[
                {
                    "role":
                        "user",

                    "content":
                        prompt
                }
            ],

            format="json"
        )

        content = (
            response
            .get(
                "message",
                {}
            )
            .get(
                "content",
                ""
            )
        )

        parsed = parse_json_object(
            content
        )

        if parsed is None:

            normalized = normalize_llm_analysis({

                "summary":
                    str(
                        content
                    ),

                "limitations":
                    (
                        "The local model returned "
                        "content that could not be "
                        "parsed as structured JSON."
                    )
            })

            normalized = (
                sanitize_unsupported_legal_references(
                    normalized,
                    evidence
                )
            )

            return {

                "status":
                    "SUCCESS",

                "model":
                    OLLAMA_MODEL,

                "analysis":
                    normalized
            }

        normalized = normalize_llm_analysis(
            parsed
        )

        # SECOND LINE OF DEFENSE:
        # deterministically remove unsupported
        # legal references from model output.
        normalized = (
            sanitize_unsupported_legal_references(
                normalized,
                evidence
            )
        )

        return {

            "status":
                "SUCCESS",

            "model":
                OLLAMA_MODEL,

            "analysis":
                normalized
        }

    except Exception as e:

        logger.exception(
            "Ollama error: %r",
            e
        )

        return {

            "status":
                "ERROR",

            "model":
                OLLAMA_MODEL,

            "message":
                (
                    "Local Ollama model could not be reached. "
                    "Retrieval evidence remains available."
                ),

            "error":
                str(e)
        }


# =========================================================
# CONFIDENCE
# =========================================================

def calculate_confidence(
    evidence,
    validation
):

    # IMPORTANT:
    # This is retrieval/evidence confidence.
    # It is NOT legal confidence.

    if not evidence:

        return {

            "level":
                "LOW",

            "score":
                0.20,

            "basis": [
                "No relevant evidence was retrieved."
            ],

            "warning":
                (
                    "This score does not represent "
                    "legal certainty."
                )
        }

    supported_domains = validation.get(
        "supported_domains",
        []
    )

    strongest = evidence[
        0
    ]

    strongest_score = float(
        strongest.get(
            "score",
            0.0
        )
    )

    similarity = strongest.get(
        "similarity"
    )

    ingredient_match = bool(
        strongest.get(
            "ingredient_matches"
        )
    )

    domain_evidence = bool(
        strongest.get(
            "supported_domains"
        )
    )

    evidence_quality = float(
        strongest.get(
            "evidence_quality",
            0.0
        )
    )

    status = validation.get(
        "status"
    )

    # No domain actually supported.
    if not supported_domains:

        level = "LOW"
        score = 0.20

    # Mixed result:
    # some domains supported, some unsupported.
    elif status == "PARTIAL":

        if (
            strongest_score >= 12
            and domain_evidence
            and evidence_quality >= 0.60
        ):

            level = "MEDIUM"
            score = 0.65

        else:

            level = "LOW"
            score = 0.45

    elif status == "UNSUPPORTED":

        level = "LOW"
        score = 0.25

    elif (
        strongest_score >= 12
        and domain_evidence
        and ingredient_match
        and evidence_quality >= 0.60
    ):

        level = "HIGH"
        score = 0.85

    elif (
        strongest_score >= 8
        and domain_evidence
        and evidence_quality >= 0.40
    ):

        level = "MEDIUM"
        score = 0.70

    elif (
        similarity is not None
        and float(
            similarity or 0.0
        ) >= 0.75
        and domain_evidence
        and evidence_quality >= 0.35
    ):

        level = "MEDIUM"
        score = 0.65

    else:

        level = "LOW"
        score = 0.45

    basis = []

    if strongest.get(
        "domain"
    ) in supported_domains:

        basis.append(
            "domain-specific textual evidence"
        )

    if ingredient_match:

        basis.append(
            "ingredient evidence"
        )

    if similarity is not None:

        basis.append(
            "semantic retrieval"
        )

    if strongest.get(
        "matched_terms"
    ):

        basis.append(
            "keyword overlap"
        )

    if validation.get(
        "status"
    ) == "PARTIAL":

        basis.append(
            "partial domain coverage"
        )

    return {

        "level":
            level,

        "score":
            score,

        "basis":
            basis,

        "supported_domains":
            supported_domains,

        "warning":
            (
                "This is retrieval/evidence confidence, "
                "not legal certainty."
            )
    }


# =========================================================
# ACTION PLAN
# =========================================================

def build_action_plan(
    domains,
    evidence,
    validation
):

    actions = []

    status = validation.get(
        "status"
    )

    supported_domains = validation.get(
        "supported_domains",
        []
    )

    unsupported_domains = validation.get(
        "unsupported_domains",
        []
    )

    if status == "UNSUPPORTED":

        actions.append(
            "Do not rely on an unsupported conclusion."
        )

        actions.append(
            "Expand the authoritative corpus with "
            "relevant TK/ABS/IP source material."
        )

    elif status == "PARTIAL":

        actions.append(
            "Use only the supported-domain evidence "
            "for interpretation."
        )

        actions.append(
            "Do not infer conclusions for unsupported "
            "domains."
        )

        actions.append(
            "Expand or verify the corpus for: "
            + ", ".join(
                unsupported_domains
            )
        )

    else:

        actions.append(
            "Open and review the original retrieved "
            "source documents."
        )

        actions.append(
            "Compare the LLM interpretation with "
            "the underlying evidence."
        )

    if "TK" in domains:

        if "TK" in supported_domains:

            actions.append(
                "Verify the retrieved traditional-knowledge "
                "evidence against the original source."
            )

        else:

            actions.append(
                "No direct TK evidence was established. "
                "Verify whether the ingredients, formulation, "
                "or stated use are actually associated with "
                "documented traditional knowledge."
            )

    if "ABS" in domains:

        if "ABS" in supported_domains:

            actions.append(
                "Verify the retrieved biological-resource "
                "and access/benefit-sharing evidence using "
                "authoritative sources."
            )

        else:

            actions.append(
                "No direct ABS evidence was established. "
                "An ingredient match alone is not sufficient."
            )

    if "IP" in domains:

        if "IP" in supported_domains:

            actions.append(
                "Identify the specific IP issue supported "
                "by the retrieved evidence before drawing "
                "a conclusion."
            )

        else:

            actions.append(
                "No direct IP evidence was established. "
                "Verify the relevant IP question separately."
            )

    actions.append(
        "Verify material conclusions with an "
        "appropriate IP/TK/ABS facilitator or "
        "qualified professional."
    )

    return actions


# =========================================================
# RESPONSE SANITIZATION
# =========================================================

def sanitize_evidence(
    evidence
):

    output = []

    for item in evidence:

        output.append({

            "id":
                item.get(
                    "id"
                ),

            "domain":
                item.get(
                    "domain"
                ),

            "source":
                item.get(
                    "source"
                ),

            "page":
                item.get(
                    "page"
                ),

            "chunk":
                item.get(
                    "chunk"
                ),

            "retrieval_type":
                item.get(
                    "retrieval_type"
                ),

            "similarity":
                item.get(
                    "similarity"
                ),

            "score":
                item.get(
                    "score"
                ),

            "evidence_quality":
                item.get(
                    "evidence_quality"
                ),

            "evidence_basis":
                item.get(
                    "evidence_basis",
                    []
                ),

            "matched_terms":
                item.get(
                    "matched_terms",
                    []
                ),

            "ingredient_matches":
                item.get(
                    "ingredient_matches",
                    []
                ),

            "domain_matches":
                item.get(
                    "domain_matches",
                    {}
                ),

            # NEW
            "supported_domains":
                item.get(
                    "supported_domains",
                    []
                ),

            "text":
                item.get(
                    "text",
                    ""
                )
        })

    return output


# =========================================================
# MAIN ANALYZE ENDPOINT
# =========================================================

@app.post(
    "/api/analyze"
)
def analyze_product(
    product: ProductInput
):

    logger.info(
        "=" * 70
    )

    logger.info(
        "IP-SAKTI ANALYSIS"
    )

    logger.info(
        "=" * 70
    )

    # -----------------------------------------------------
    # CLASSIFICATION
    # -----------------------------------------------------

    classification = classify_product(
        product
    )

    # -----------------------------------------------------
    # DOMAIN DETECTION
    # -----------------------------------------------------

    domains = detect_domains(
        product
    )

    # -----------------------------------------------------
    # SEARCH QUERY
    # -----------------------------------------------------

    query = build_search_query(
        product,
        domains
    )

    # -----------------------------------------------------
    # EVIDENCE RETRIEVAL
    # -----------------------------------------------------

    evidence = search_corpus(

        query=query,

        domains=domains,

        ingredients=product.ingredients,

        top_k=TOP_K
    )

    logger.info(
        "Product: %s",
        product.product_name
    )

    logger.info(
        "Ingredients: %s",
        product.ingredients
    )

    logger.info(
        "Domains: %s",
        domains
    )

    logger.info(
        "Evidence found: %d",
        len(evidence)
    )

    logger.info(
        "BGE available: %s",
        EMBEDDINGS_AVAILABLE
    )

    logger.info(
        "Embedding model: %s",
        EMBEDDING_MODEL
    )

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    validation = validate_evidence(
        evidence,
        domains
    )

    logger.info(
        "Validation status: %s",
        validation.get(
            "status"
        )
    )

    logger.info(
        "Supported domains: %s",
        validation.get(
            "supported_domains"
        )
    )

    logger.info(
        "Unsupported domains: %s",
        validation.get(
            "unsupported_domains"
        )
    )

    # -----------------------------------------------------
    # LLM REASONING
    # -----------------------------------------------------

    llm_reasoning = (
        generate_llm_reasoning(
            product,
            domains,
            evidence,
            validation
        )
    )

    # -----------------------------------------------------
    # CONFIDENCE
    # -----------------------------------------------------

    confidence = calculate_confidence(
        evidence,
        validation
    )

    # -----------------------------------------------------
    # SOURCES
    # -----------------------------------------------------

    sources = []

    for item in evidence:

        source = item.get(
            "source",
            "Unknown source"
        )

        if source not in sources:

            sources.append(
                source
            )

    # -----------------------------------------------------
    # REASONING METADATA
    # -----------------------------------------------------

    if evidence:

        retrieval_methods = sorted(
            set(
                item.get(
                    "retrieval_type",
                    "UNKNOWN"
                )
                for item in evidence
            )
        )

        reasoning = {

            "summary":
                (
                    "The product was routed to the "
                    "detected domains and evidence was "
                    "retrieved using domain filtering, "
                    "ingredient matching, keyword scoring "
                    "and BGE semantic retrieval when available."
                ),

            "domains_considered":
                domains,

            "supported_domains":
                validation.get(
                    "supported_domains",
                    []
                ),

            "unsupported_domains":
                validation.get(
                    "unsupported_domains",
                    []
                ),

            "search_query":
                query,

            "evidence_count":
                len(evidence),

            "sources":
                sources,

            "retrieval_methods":
                retrieval_methods,

            "embedding_model":
                EMBEDDING_MODEL,

            "note":
                (
                    "Domain metadata is not treated as "
                    "evidence. Direct textual domain matches "
                    "are required for domain support. "
                    "This prototype does not establish "
                    "legal compliance, legal violation, "
                    "legal liability, patentability, "
                    "ownership, or infringement."
                )
        }

    else:

        reasoning = {

            "summary":
                (
                    "The system could not establish "
                    "sufficient domain-aligned evidence "
                    "from the current corpus."
                ),

            "domains_considered":
                domains,

            "supported_domains":
                [],

            "unsupported_domains":
                domains,

            "search_query":
                query,

            "evidence_count":
                0,

            "retrieval_methods": [

                "Domain filtering",

                "Ingredient matching",

                "Keyword retrieval",

                "BGE semantic retrieval"
            ],

            "limitation":
                (
                    "The system abstains instead of "
                    "returning unrelated documents."
                )
        }

    # -----------------------------------------------------
    # ACTION PLAN
    # -----------------------------------------------------

    action_plan = build_action_plan(

        domains=domains,

        evidence=evidence,

        validation=validation
    )

    # -----------------------------------------------------
    # FINAL RESPONSE
    # -----------------------------------------------------

    return {

        "app_version":
            APP_VERSION,

        "product":
            product.model_dump(),

        "classification":
            classification,

        "domains":
            domains,

        "search_query":
            query,

        "evidence":
            sanitize_evidence(
                evidence
            ),

        "reasoning":
            reasoning,

        "llm_reasoning":
            llm_reasoning,

        "validation":
            validation,

        "confidence":
            confidence,

        "action_plan":
            action_plan
    }


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():

    corpus = load_corpus()

    return {

        "message":
            "IP-SAKTI MVP API is running",

        "version":
            APP_VERSION,

        "docs":
            "/docs",

        "conversation":
            (
                "/api/conversation/start"
                if conversation_router
                is not None
                else None
            ),

        "corpus_chunks":
            len(corpus),

        "llm":
            OLLAMA_MODEL,

        "bge_embeddings":
            EMBEDDINGS_AVAILABLE,

        "embedding_model":
            EMBEDDING_MODEL
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get(
    "/api/health"
)
def health():

    corpus = load_corpus()

    return {

        "status":
            "ok",

        "version":
            APP_VERSION,

        "corpus_available":
            bool(corpus),

        "corpus_chunks":
            len(corpus),

        "ollama_model":
            OLLAMA_MODEL,

        "bge_embeddings":
            EMBEDDINGS_AVAILABLE,

        "embedding_model":
            EMBEDDING_MODEL,

        "conversation_router":
            conversation_router
            is not None
    }


# =========================================================
# DEVELOPMENT CORPUS RELOAD
# =========================================================

@app.post(
    "/api/admin/reload-corpus"
)
def reload_corpus():

    """
    Development helper.

    For production, protect this endpoint
    with authentication or remove it.
    """

    corpus = load_corpus(
        force_reload=True
    )

    return {

        "status":
            "reloaded",

        "corpus_chunks":
            len(corpus)
    }


# =========================================================
# LOCAL START
# =========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        "main:app",

        host="127.0.0.1",

        port=8000,

        reload=True
    )
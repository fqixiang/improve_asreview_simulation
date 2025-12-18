from asreview.learner import ActiveLearningCycleData

model_configurations =  {
        "elas_u4": ActiveLearningCycleData(
            querier="max",
            classifier="svm",
            classifier_param={"loss": "squared_hinge", "C": 0.11},
            balancer="balanced",
            balancer_param={"ratio": 9.8},
            feature_extractor="tfidf",
            feature_extractor_param={
                "ngram_range": (1, 2),
                "sublinear_tf": True,
                "min_df": 1,
                "max_df": 0.95,
            },
        ),
        "elas_u3": ActiveLearningCycleData(
            querier="max",
            classifier="nb",
            classifier_param={"alpha": 3.822},
            balancer="balanced",
            balancer_param={"ratio": 1.2},
            feature_extractor="tfidf",
            feature_extractor_param={"stop_words": "english"},
        ),
        "elas_l2": ActiveLearningCycleData(
            querier="max",
            classifier="svm",
            classifier_param={"loss": "squared_hinge", "C": 0.106, "max_iter": 5000},
            balancer="balanced",
            balancer_param={"ratio": 9.707},
            feature_extractor="multilingual-e5-large",
            feature_extractor_param={"normalize": True},
        ),
        "elas_h3": ActiveLearningCycleData(
            querier="max",
            classifier="svm",
            classifier_param={"loss": "squared_hinge", "C": 0.067, "max_iter": 5000},
            balancer="balanced",
            balancer_param={"ratio": 9.724},
            feature_extractor="mxbai",
            feature_extractor_param={"normalize": True},
        )
    }


def pad_labels(labels, n_priors, total_n_records):
    """Pad labels to match the dataset size."""
    return labels + [0] * (total_n_records - len(labels) - n_priors)


def n_query_extreme(results, n_records):
    """Determine the number of queries to make based on dataset size and current results."""
    if n_records >= 10000:
        if len(results) >= 10000:
            return 10**5  # finish the run
        if len(results) >= 1000:
            return 1000
        elif len(results) >= 100:
            return 25
        else:
            return 1
    else:
        if len(results) >= 1000:
            return 100
        elif len(results) >= 100:
            return 5
        else:
            return 1


def get_abstract_length(row):
    """Calculate abstract length based on language.
    
    For space-separated languages (English, etc.), count words.
    For non-space languages (Chinese, Japanese, etc.), count characters.
    
    Args:
        row: DataFrame row containing 'abstract' and optionally 'language' columns
        
    Returns:
        int: Length of abstract (word count or character count)
    """
    import pandas as pd
    
    abstract = str(row.get("abstract", ""))
    if pd.isna(abstract) or abstract == "nan" or abstract == "":
        return 0
    
    # Check language if available (synergy benchmark)
    language = row.get("language", "en")
    # Languages without spaces: Chinese, Japanese, Thai, Korean, Lao, Burmese, Khmer
    non_space_languages = ["zh", "ja", "th", "ko", "lo", "my", "km"]
    
    if language in non_space_languages:
        # Count characters (excluding whitespace)
        return len(abstract.replace(" ", ""))
    else:
        # Count words (space-separated tokens)
        return len(abstract.split())

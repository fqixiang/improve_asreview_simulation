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
erDiagram
    CASE {
      uuid case_uuid PK
      varchar cohort "Cohort: TCGA or HLCC"
    }
    MUTATION {
      uuid mutation_uuid PK
      uuid case_uuid FK
      varchar gene "Gene checked for mutation"
      varchar mutated "Mutation status: yes or no"
      varchar variant_calling "Variant calling: all or oncogenic only"
      int stratification_fold "Stratification fold over cases within a cohort"
    }
    WSI {
      uuid case_uuid PK
      uuid wsi_uuid FK
      varchar tissue_preservation "Tissue preservation: formalin fixed or fresh frozen"
      varchar wsi_artifact "Path to slides"
      bool quality_control_reject "Visual quality control"
    }
    TISSUE_BOUNDARY {
      uuid tissue_boundary_uuid PK
      uuid wsi_uuid FK
      uuid tile_extractor_uuid FK
      bytea tissue_boundary_thumbnail "Base64 encoded PNG"
    }
    TILE {
      uuid tile_uuid PK
      uuid wsi_uuid FK
      uuid extractor_uuid FK
      int x "Horizontal tile position from top left"
      int y "Vertical tile position from top left"
      bytea tile "JPEG encoded raw image, use with PIL.Image.open(io.BytesIO(x))"
    }
    TILE_EXTRACTOR {
      uuid tile_extractor_uuid PK
      varchar normalization "Normalization: none or reinhard"
      float mpp "Micrometer per pixel used for extraction"
      int tile_size "Height and width of image tiles"
    }
    EMBEDDER {
      uuid embedder_uuid PK
      varchar name "Model name: imgnet res34, ctranspath, or uni"
      varchar repository_url "Repository URL for replication"
      varchar version_tag "Version tag for model versions"
    }
    EMBEDDING {
      uuid embedding_uuid PK
      uuid tile_uuid FK
      uuid wsi_uuid FK "Partitioning column for fast retrieval"
      uuid embedder_uuid FK
      bytea embedding "Raw vector, used with numpy.load(io.BytesIO(x))"
    }
    EXPERIMENT {
      uuid experiment_uuid PK
      uuid embedder_uuid FK
      varchar gene "Gene checked for mutation"
      varchar variant_calling "Variant calling application"
      int dev_fold "Cross-validation fold for early stopping"
      int test_fold "Cross-validation fold withheld from training"
      varchar train_cohort "Which cohort to use for training"
      varchar holdout_chohrt "Which cohort to use for validation"
      int seed "Initialization seed for model and batch sampling"
      varchar status "Either of 'pending, processing, done'"
    }
    EXPLANATION {
      uuid explanation_uuid PK
      uuid experiment_uuid FK
      uuid model_uuid
      uuid embedding_uuid
      float attention_weights "Logit from the attention MIL head"
    }
    MODEL_CHECKPOINTS {
      uuid experiment_uuid PK
      bytea weights "Raw tensor, used with torch.load(io.BytesIO(x))"
    }
    PREDICTION {
      uuid prediction_uuid PK
      uuid experiment_uuid FK
      uuid wsi_uuid FK
      float prediction "Model output logit"
      int label "True mutation state"
      float logit "Predicted mutation state"
      str split "Slide set: train/dev/test/validation"
    }
    
    CASE ||--o{ MUTATION: "has"
    CASE ||--o{ WSI: "has"
    WSI ||--o{ TISSUE_BOUNDARY: "has"
    WSI ||--o{ TILE: "has"
    TILE ||--o{ EMBEDDING: "has"
    TILE ||--o{ PREDICTION: "has"
    TILE ||--|{ TILE_EXTRACTOR: "uses"
    EMBEDDER ||--o{ EMBEDDING: "produces"
    EMBEDDER ||--o{ EXPERIMENT: "participates in"
    EMBEDDING ||--|{ TILE_EXTRACTOR: "uses"
    EXPERIMENT ||--o{ EXPLANATION: "generates"
    EXPERIMENT ||--|{ MODEL_CHECKPOINTS: "uses"
    PREDICTION ||--o{ EXPLANATION: "explains"

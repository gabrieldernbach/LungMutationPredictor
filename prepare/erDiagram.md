erDiagram
    CASE {
      uuid case_uuid PK
      varchar cohort
    }
    MUTATION {
      uuid mutation_uuid PK
      uuid case_uuid FK
      varchar protein
      varchar status
      varchar variant_calling
    }
    CROSS_VALIDATION {
      uuid validation_uuid PK
      uuid case_uuid FK
      int validation_fold
      varchar target_protein
      varchar variant_calling
    }
    WSI {
      uuid case_uuid PK
      uuid wsi_uuid FK
      varchar tissue_preservation
      varchar wsi_artefact
    }
    TISSUE_BOUNDARY {
      uuid tissue_boundary_uuid PK
      uuid wsi_uuid FK
      uuid tile_extractor_uuid FK
      bytea tissue_boundary_thumbnail
    }
    TILE {
      uuid tile_uuid PK
      uuid wsi_uuid FK
      uuid extractor_uuid FK
      int x
      int y
      bytea img
    }
    TILE_EXTRACTOR {
      uuid tile_extractor_uuid PK
      varchar normalization
      float mpp
      int patch_size
    }
    EMBEDDER {
      uuid embedder_uuid PK
      varchar name
      varchar repository_url
      varchar version_tag
    }
    EMBEDDING {
      uuid embedding_uuid PK
      uuid wsi_uuid FK
      uuid tile_extractor_uuid FK
      uuid embedder_uuid FK
      bytea embedding_table
    }
    EXPERIMENT {
      uuid experiment_uuid PK
      uuid embedder_uuid FK
      varchar marker
      varchar variant_calling
      int dev_fold
      int test_fold
      int seed
    }
    EXPLANATION {
      uuid explanation_uuid PK
      uuid experiment_uuid FK
      uuid model_uuid
      uuid embedding_uuid
      float attention_weights
    }
    MODEL_CHECKPOINTS {
      uuid experiment_uuid PK
      bytea weights
    }
    PREDICTION {
      uuid prediction_uuid PK
      uuid experiment_uuid FK
      uuid wsi_uuid FK
      uuid tile_uuid
      int label
      float logit
      str split
    }
    
    CASE ||--o{ MUTATION: "has"
    CASE ||--o{ CROSS_VALIDATION: "has"
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
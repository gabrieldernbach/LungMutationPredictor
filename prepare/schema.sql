CREATE TABLE MUTATION (
    mutation_uuid UUID PRIMARY KEY,
    case_uuid UUID,
    marker VARCHAR(255),
    state VARCHAR(255),
    variant_calling VARCHAR(255),
    fold INT
);

CREATE TABLE WSI (
    case_uuid UUID PRIMARY KEY,
    wsi_uuid UUID REFERENCES WSI(wsi_uuid),
    tissue_preservation VARCHAR(255),
    wsi_artefact VARCHAR(255)
);

CREATE TABLE TISSUE_BOUNDARY (
    tissue_boundary_uuid UUID PRIMARY KEY,
    wsi_uuid UUID REFERENCES WSI(wsi_uuid),
    tile_extractor_uuid UUID REFERENCES TILE_EXTRACTOR(tile_extractor_uuid),
    tissue_boundary_thumbnail BYTEA
);

CREATE TABLE TILE (
    tile_uuid UUID PRIMARY KEY,
    wsi_uuid UUID REFERENCES WSI(wsi_uuid),
    extractor_uuid UUID REFERENCES TILE_EXTRACTOR(tile_extractor_uuid),
    x INT,
    y INT,
    img BYTEA
);

CREATE TABLE TILE_EXTRACTOR (
    tile_extractor_uuid UUID PRIMARY KEY,
    normalization VARCHAR(255),
    mpp FLOAT,
    patch_size INT
);

CREATE TABLE EMBEDDING (
    embedding_uuid UUID PRIMARY KEY,
    wsi_uuid UUID REFERENCES WSI(wsi_uuid),
    tile_extractor_uuid UUID REFERENCES TILE_EXTRACTOR(tile_extractor_uuid),
    embedder_uuid UUID REFERENCES EMBEDDER(embedder_uuid),
    embedding INT
);

CREATE TABLE EMBEDDER (
    embedder_uuid UUID PRIMARY KEY,
    name VARCHAR(255),
    repository_url VARCHAR(255),
    version_tag VARCHAR(255)
);

CREATE TABLE MODEL_TRAINING (
    model_uuid UUID PRIMARY KEY,
    embedder_uuid UUID REFERENCES EMBEDDER(embedder_uuid),
    target_protein VARCHAR(255),
    variant_calling VARCHAR(255),
    validation_fold INT,
    random_seed INT,
    model_artefact VARCHAR(255)
);

CREATE TABLE PREDICTION (
    prediction_uuid UUID PRIMARY KEY,
    model_uuid UUID REFERENCES MODEL_TRAINING(model_uuid),
    case_uuid UUID REFERENCES CASE(case_uuid),
    logit FLOAT
);

CREATE TABLE EXPLANATION (
    explanation_uuid UUID PRIMARY KEY,
    model_uuid UUID REFERENCES MODEL_TRAINING(model_uuid),
    tile_uuid UUID REFERENCES TILE(tile_uuid),
    logit FLOAT,
    attention_weight FLOAT
);
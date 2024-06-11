Repository for replicating the mutation prediction results in our publication 

**Dissecting AI-based mutation prediction in lung adenocarcinoma: a comprehensive real-world study**

> Molecular profiling of lung cancer is critical for identifying genetic alterations that predict response to targeted therapy. Deep learning has shown promise for predicting oncogenic mutations from tissue whole slide images, but existing studies often suffer from limited sample sizes, a focus on earlier stage patients with limited therapeutic applications, and insufficient analysis of robustness and generalizability.
This retrospective study systematically evaluates factors influencing the accuracy of mutation prediction using an  extensive dataset – the Heidelberg Lung Adenocarcinoma Cohort (HLCC) – comprising 2356 late-stage FFPE samples, and is validated on the publicly available TCGA-LUAD cohort.
Our results show that models trained on the sixfold larger HLCC cohort generalize well to the TCGA dataset for mutations in EGFR (AUC 0.76), STK11 (AUC 0.71) and TP53 (AUC 0.75). This supports the hypothesis that cohort sizes improve model robustness. Our benchmark highlights performance variations due to preprocessing and modeling choices, such as mutation variant calling, which can cause changes of up to 7% in EGFR prediction accuracy. 
Model explanations reveal that acinar and papillary growth patterns are key in  detecting EGFR mutations, while solid growth patterns and large nuclei are indicative of TP53 mutations. Despite the advancements, mutation prediction cannot replace comprehensive molecular profiling but can serve as a valuable enrichment tool for clinical trials and enhance  understanding of genotype-phenotype correlations.

Example heatmaps for predicting mutations in EGFR. Acinar and papillary growth patterns are targeted
![EGFR](EGFR.png)
Example heatmaps for predicting mutations in TP53. High attention patches often contain solid growth pattern.
![TP53](TP53.png)

The repository requires a table of cases with known mutational states and locations of their whole slide images. From this, the following structure of tables is derived, including:

Image tiling
Image embedding
Experiments
Training artifacts
Model predictions and their explanations (attention scores)
The entity relationship diagram below outlines how the data is related and stored.

Due to the vast size of the data—thousands of models each attending to millions of patches—cloud storage should be considered for storing and retrieving the data efficiently.
![mermaid-diagram-2024-06-11-173059](er_diagram.svg)

The order of execution is as follows
* `prepare/register_wsis.py`: This script sets up the required table summarizing the mutational state and the locations of the whole slide images.
* `slicing/app/main.py`: This script iterates over the slide images, parameterized to 224 square pixels at 0.5 mpp by default. Execution can be significantly sped up by using parallel instances. Multiple instances can be launched by setting parallelism in the Kubernetes deployment file `machine.yaml`.
* `embedding` package: This package contains ctranspath and uni embedder modules. One retrieval and upload node is launched, and multiple embedding services build a unified backbone.
* `training/gen_experiment.py`: This script defines which experiments to run and creates a declarative experiments table. These experiments can be executed by launching multiple `run.py` workers using the `16core.yaml`. Again, parallelism significantly reduces runtime.
* `evaluation`: This directory contains code for heatmapping and the generation of tile clusters.







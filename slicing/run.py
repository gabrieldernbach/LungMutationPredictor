from pathlib import Path

from projects.n20_10_hd_lung.ejc_review.utils import srun

if __name__ == "__main__":
    tag = "gabrieldernbach/histo:wsi_splitter",
    context = f"{Path(__file__).resolve().parent}"

    srun(f"docker build -t {tag} {context}")
    srun(f"docker push {tag}")
    srun(f"kubectl apply -f {context}/machine.yaml")

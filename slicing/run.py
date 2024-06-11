from pathlib import Path
import subprocess

if __name__ == "__main__":
    tag = "gabrieldernbach/histo:wsi_splitter",
    context = f"{Path(__file__).resolve().parent}"

    subprocess.run(f"docker build -t {tag} {context}".split())
    subprocess.run(f"docker push {tag}")
    subprocess.run(f"kubectl apply -f {context}/machine.yaml")

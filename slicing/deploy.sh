docker build -t gabrieldernbach/histo:wsi_splitter .
docker push gabrieldernbach/histo:wsi_splitter
kubectl apply -f /machine.yaml
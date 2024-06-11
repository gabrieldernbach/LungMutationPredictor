# build an push images
docker buildx build --platform linux/amd64,linux/arm64 -t gabrieldernbach/histo:schedule_embedding2 . --push
docker build --no-cache --platform linux/amd64 -t gabrieldernbach/histo:uni-embedder-service models/uni && docker push gabrieldernbach/histo:uni-embedder-service
docker build --no-cache --platform linux/amd64 -t gabrieldernbach/histo:ctranspath-embedder-service models/ctranspath && docker push gabrieldernbach/histo:ctranspath-embedder-service
docker build --no-cache --platform linux/amd64 -t eu.gcr.io/aignx-development/home_gabriel:rudolfv models/rudolfv && docker push eu.gcr.io/aignx-development/home_gabriel:rudolfv

# deploy
kubectl apply -f embed_with_ctranspath.yaml
kubectl apply -f embed_with_uni.yaml
kubectl apply -f embed_with_rudolfv.yaml

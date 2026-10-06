# 部署

## Docker

```bash
docker build -t beamforge:1.0.0 .
docker run -p 8899:8899 beamforge:1.0.0 serve --port 8899
```

## Kubernetes

```bash
kubectl apply -f deploy/k8s/deployment.yaml
kubectl apply -f deploy/k8s/service.yaml
```

## Helm

```bash
helm install beamforge deploy/helm
```

#!/usr/bin/env bash
# Runs the ragd-vs-Dify comparison as soon as its prerequisites exist.
set -u
cd "$(dirname "$0")/.."
log() { echo "[$(date +%H:%M:%S)] $*"; }
envval() { grep "^$1=" .env | cut -d= -f2- ; }
RAGD_DS=01a11067-68ea-7e69-9ba6-a4301295e431

log "waiting for SiliconFlow balance"
until [ "$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $(envval EMBEDDING_API_KEY)" \
  -H 'Content-Type: application/json' -d '{"model":"BAAI/bge-m3","input":["ping"]}' \
  https://api.siliconflow.cn/v1/embeddings)" = 200 ]; do sleep 60; done
log "SiliconFlow OK"

set -a; . ./.env; set +a
ROOT=$(echo "$RAG_API_KEYS" | cut -d, -f1)

if [ "$(wc -l < eval/questions.jsonl)" -lt 100 ]; then
  log "generating more questions"
  ./bin/rageval gen -docs eval/docs -n 90 -seed 7 -llm-model Qwen/Qwen2.5-7B-Instruct -out eval/questions-more.jsonl -docling http://localhost:5001
  python3 - <<'PY'
import json
qs=[json.loads(l) for l in open('eval/questions.jsonl')]+[json.loads(l) for l in open('eval/questions-more.jsonl')]
seen=set(); out=[]
for q in qs:
    k=q['query'].strip().lower()
    if k in seen: continue
    seen.add(k); q['id']='q%03d'%(len(out)+1); out.append(q)
with open('eval/questions.jsonl','w') as f:
    for q in out: f.write(json.dumps(q,ensure_ascii=False)+'\n')
print(len(out),'questions')
PY
fi

log "ragd-only run"
./bin/rageval run -set eval/questions.jsonl -k 5 \
  -target "name=ragd,kind=external,url=http://localhost:8080,key=$ROOT,knowledge=$RAGD_DS" \
  -json eval/report-ragd.json | tee eval/report-ragd.md

log "waiting for DIFY_DATASET_KEY in .env"
until grep -q '^DIFY_DATASET_KEY=dataset-.\+' .env; do sleep 10; done
set -a; . ./.env; set +a
if [ ! -f eval/.dify-dataset ]; then
  ./bin/rageval load -docs eval/docs -name study-eval -dify-url http://localhost/v1 -dify-key "$DIFY_DATASET_KEY" \
    | tee /tmp/dify-load.out
  grep '^dify knowledge id:' /tmp/dify-load.out | awk '{print $4}' > eval/.dify-dataset
fi
DIFY_DS=$(cat eval/.dify-dataset)

log "full comparison"
./bin/rageval run -set eval/questions.jsonl -k 5 \
  -target "name=ragd,kind=external,url=http://localhost:8080,key=$ROOT,knowledge=$RAGD_DS" \
  -target "name=dify-default,kind=dify,url=http://localhost/v1,key=$DIFY_DATASET_KEY,knowledge=$DIFY_DS,retrieval=eval/dify-default.json" \
  -target "name=dify-hybrid-rerank,kind=dify,url=http://localhost/v1,key=$DIFY_DATASET_KEY,knowledge=$DIFY_DS,retrieval=eval/dify-hybrid-rerank.json" \
  -json eval/report.json | tee eval/report.md
log "done"

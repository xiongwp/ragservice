# ragd：Go 实现的 RAG 服务（Dify 外部知识库）

一个独立的检索服务，负责建库、上传、索引和检索，通过 Dify 的「外部知识库 API」接入。Dify 继续负责应用和工作流。

- 存储：**MySQL 8**（库、文档、分段、任务队列、缓存、租户）+ **Qdrant**（向量）
- 文档解析：txt / md / html 在进程内完成；pdf / docx / pptx / xlsx 交给 **docling** sidecar
- 文本编码：UTF-8/16/32 直接识别；非 UTF-8 的老文件先看 BOM 和 HTML `<meta charset>`，再用统计方法识别（GB18030、Big5、Shift_JIS、EUC-JP、EUC-KR、windows-125x、ISO-8859-x、KOI8-R 等）；识别不出来时回退到 `RAG_TEXT_FALLBACK_ENCODING`（默认 gb18030）
- 模型：任何兼容 OpenAI 协议的 embedding、chat 接口，以及 Cohere/TEI 格式的 rerank 接口（硅基流动、OpenAI、自建 TEI/vLLM 等）

| 阶段 | 内容 |
|---|---|
| M1 跑通 | 建库、上传、索引流水线、向量检索、`/retrieval` |
| M2 效果 | 中文 BM25 全文检索 + RRF 融合、rerank、父子分段、元数据过滤、评测工具 `rageval` |
| M3 进阶 | Contextual Retrieval、query 改写（多 query / HyDE / 对话历史补全）、增量更新 |
| M4 生产 | 多租户与 API Key 角色、按租户限流、配额、管理界面、Dify 导入工具 `difyimport` |

## 快速开始

```bash
cp .env.example .env        # 至少填 RAG_API_KEYS 和 EMBEDDING_API_KEY
make up                     # mysql + qdrant + docling + ragd
```

打开 http://localhost:8080/ui/ ，用 `RAG_API_KEYS` 里的 root key 登录，可以建库、上传、看索引进度、做检索测试、管理 Key 和租户。

docling 镜像有好几个 GB。只处理 txt/md/html 的话，可以只起 `mysql qdrant ragd` 三个服务。

**没有模型 Key 时，可以用本地假模型把流程跑通**（只用来验证流程，不能用来评估检索效果）：

```bash
go run ./cmd/devmodels -addr :9900
```
然后设置 `EMBEDDING_BASE_URL=http://localhost:9900/v1 EMBEDDING_MODEL=dev-bigram RERANK_BASE_URL=http://localhost:9900/v1`。

## 接入 Dify

1. 在控制台的「API Keys」页为 Dify 建一个 **reader** Key。
2. 在 Dify 中：知识库 → 外部知识库 API → 添加。Endpoint 填 `http://<ragd>:8080`（Dify 会自己拼上 `/retrieval`），API Key 填上一步建的 Key。
3. 在 Dify 中：知识库 → 连接外部知识库，「外部知识库 ID」填 ragd 的知识库 ID（控制台里有复制按钮）。
4. 如果 Dify 也用 docker 部署、ragd 跑在宿主机上，地址用 `http://host.docker.internal:8080`。保存时提示连接失败的话，检查 Dify 的 `ssrf_proxy` 是否放行了这个地址。

### 从 Dify 迁移现有知识库

```bash
go run ./cmd/difyimport \
  -dify-url http://<dify>/v1 -dify-key dataset-xxx -dify-dataset <Dify 知识库 ID> \
  -rag-url http://localhost:8080 -rag-key <editor key> -create "公司制度"
```

导入时，上传来源的文档会直接下载原文件，用 ragd 的流程重新解析；其他来源（文本、Notion、网页）把 Dify 的分段按顺序拼接后导入。Dify 的文档元数据会一并带过来。重复运行是安全的：内容相同的文档会被跳过。加 `-dry-run` 只列出将要导入的内容，不做实际导入。

## 检索流程

```
query ─┬─(可选) LLM 改写：补全指代 / 多个改写问法 / HyDE 假设答案
       ├─ 向量检索 ×N ─┐   Qdrant，租户/知识库/元数据过滤在检索内完成
       └─ BM25 检索 ×N ─┴─ RRF 融合（k=60，只看排名）
                          → 回 MySQL 取子块、父块、文档（MySQL 是数据的唯一来源）
                          → 按父块去重 → (可选) cross-encoder rerank → top_k
```

- **返回的分数**始终在 0–1 之间，并且按返回顺序不升，所以 Dify 的 score_threshold 依然有意义：开了 rerank 就是 rerank 分数；混合检索时是归一化后的 RRF 分数；纯向量检索时是余弦相似度。
- **中文 BM25**：先用 gse（Go 版 jieba）分词保证精度，再加中文字二元组保证召回，覆盖「年假」「内网」这类词典里没有的业务词。分好的词变成 sparse 向量存进 Qdrant，IDF 由 Qdrant 在服务端计算。
- **元数据过滤**支持 Dify 的全部运算符。等值、范围、空值这类条件直接交给 Qdrant；子串类（contains / start with / end with）先在 MySQL 查出匹配的文档，再按文档 id 过滤。
- **出错时降级**：rerank 或 LLM 改写出错，只记一条日志，检索照常返回结果。
- 调试接口 `POST /v1/search` 可以一次查多个知识库、带对话历史，并返回每条结果的向量排名、关键词排名和 rerank 分数。控制台的「检索测试」页就是用的这个接口。

## 索引流程

```
上传 ─(同一 MySQL 事务：文档 + 原文件 + 任务)─► MySQL 任务队列（SKIP LOCKED，自动重试、租约回收）
  解析 → 清洗 → 结构感知切分（标题/表格/列表/代码块 → 按 token 合并，父 800 / 子 200）
  → (可选) Contextual Retrieval：LLM 为每个子块写一句上下文，结果缓存在 MySQL
  → 和旧分段按内容 hash 对比：没变的沿用原 id 和向量，只对变化的部分做 embedding
  → 写 Qdrant → 在一个 MySQL 事务里替换分段，同时登记「删除旧向量」任务
```

- **MySQL 和 Qdrant 的一致性**：新向量先写入，但对应的分段行还没提交，检索时会被忽略。分段行提交之后，旧向量没有了对应的行，同样会被忽略，随后由清理任务删除。删文档、删知识库也是在同一个事务里登记向量清理任务。
- **两层缓存**：embedding 按 `(模型, sha256)` 缓存，LLM 写的上下文按 `(模型, 标题+父块+子块)` 缓存。重建一个没改动的文档，不会再调用模型。
- **入库前校验向量**：维度、NaN/Inf、全零，以及 index 缺失或重复，任何一项不对都直接报错，不会写进库。
- **失败处理**：不支持的格式、模型鉴权错误、文档解析不出文字这类错误直接标为 failed，不再重试。网络错误和 429 会退避重试。

## 多租户与权限

| 角色 | 权限 |
|---|---|
| root | `RAG_API_KEYS` 里的 Key：管理租户；用 `X-Tenant-ID` 头可以切换到任意租户操作 |
| admin | 管理本租户的 API Key |
| editor | 建库、上传、修改、删除 |
| reader | 检索和只读（给 Dify 用这个角色） |

- 租户之间完全隔离：MySQL 查询和 Qdrant 过滤都会带上 tenant_id。
- 每个租户有独立的限流（rps）和配额（知识库数、文档数、存储量），0 表示不限制。
- Key 只存 sha256，明文只在创建时显示一次；吊销立即生效。

## API

| 方法 | 路径 | 角色 |
|---|---|---|
| POST | `/retrieval` | reader（Dify 协议） |
| GET | `/v1/whoami`、`/v1/usage` | reader |
| GET | `/v1/datasets`、`/v1/datasets/{id}`、`/v1/datasets/{id}/documents`、`/v1/documents/{id}` | reader |
| POST | `/v1/search` | reader |
| POST / DELETE | `/v1/datasets`、`/v1/datasets/{id}` | editor |
| POST | `/v1/datasets/{id}/documents`（multipart：`file`, `metadata`） | editor |
| PUT | `/v1/documents/{id}/file`（上传新版本，增量更新） | editor |
| PATCH | `/v1/documents/{id}` `{"metadata": {...}}`（不会重新做 embedding） | editor |
| POST | `/v1/documents/{id}/reindex`、`/v1/datasets/{id}/reindex` | editor |
| DELETE | `/v1/documents/{id}` | editor |
| GET / POST / DELETE | `/v1/keys`、`/v1/keys/{id}` | admin |
| GET / POST / PUT | `/v1/admin/tenants`、`/v1/admin/tenants/{id}` | root |
| GET | `/healthz`（检查 MySQL 和 Qdrant）、`/ui/` | 无需鉴权 |

建知识库时可以带 `chunk_config`：`{"parent_tokens": 800, "child_tokens": 200, "contextual": true}`。没给的父块/子块大小按当时的 `CHUNK_PARENT_TOKENS` / `CHUNK_CHILD_TOKENS` 补上并存进知识库，之后改这两个环境变量不影响已建的知识库（改了会让整库重新切片、重新向量化）。要求 `child_tokens >= 32` 且 `parent_tokens >= child_tokens`，索引时也按同样规则检查。

## 评测：和 Dify 原生知识库对比

```bash
go run ./cmd/rageval -set eval/questions.jsonl -k 5 \
  -target 'name=ragd,kind=external,url=http://localhost:8080,key=<reader key>,knowledge=<ragd 知识库 ID>' \
  -target 'name=dify,kind=dify,url=http://<dify>/v1,key=dataset-xxx,knowledge=<Dify 知识库 ID>' \
  -json report.json
```

- 问题集格式见 [eval/sample.jsonl](eval/sample.jsonl)：每行是一个问题，加上正确答案里应包含的短语（`expect`）或应命中的文件（`expect_docs`）。两边系统的分段 id 不同，所以是否命中按内容判断。
- 输出 Recall@k、Hit@k、MRR、nDCG@k、P50/P95 延迟，以及「只有一方答对」的问题数。每个问题的明细写在 `-json` 指定的文件里。
- Dify 那边使用该知识库在 Dify 里配置的检索设置，所以要把它的 Top K 设成大于等于 `-k`。
- 要得出可信的结论，至少需要从真实文档里整理 100–300 个问题。

## 配置

完整说明见 [.env.example](.env.example)。常用项：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `RAG_MYSQL_DSN` | — | `user:pass@tcp(host:3306)/db` |
| `QDRANT_HOST` / `QDRANT_GRPC_PORT` / `QDRANT_COLLECTION` | localhost / 6334 / rag_chunks | |
| `EMBEDDING_*` | 硅基流动 BAAI/bge-m3 | 模型必须输出 1024 维 |
| `RAG_SEARCH_MODE` | hybrid | 或 dense |
| `RERANK_BASE_URL` | 空（关闭） | |
| `LLM_BASE_URL` | 空（关闭） | 用于 Contextual Retrieval 和 query 改写 |
| `RAG_QUERY_REWRITE` | off | multi / hyde |
| `RAG_WORKERS` | 4 | 同时索引的文档数 |

## 开发与测试

```bash
make test         # 单元测试
make test-deps    # 起临时 MySQL 8.4（33306）和 Qdrant（56333/56334）
make test-e2e     # 加上集成测试和端到端测试（真实 MySQL / Qdrant / 队列，模型用假服务）
```

目录结构：

```
cmd/ragd         服务本体（迁移、API、索引 worker、控制台）
cmd/rageval      检索评测
cmd/difyimport   从 Dify 导入
cmd/devmodels    本地假模型（仅用于开发）
internal/
  api            HTTP 接口、鉴权、限流、控制台（ui/）
  auth           租户、API Key、角色
  chunk          清洗、结构感知切分
  contextual     Contextual Retrieval
  db             MySQL 连接与迁移（migrations/）
  embedding      embedding 客户端与缓存
  ingest         索引流水线
  lexical        中文 BM25 sparse 向量
  llm            chat 客户端
  parse          文档解析（含 docling 客户端）
  queue          MySQL 任务队列
  rerank         rerank 客户端
  retrieve       检索、融合、元数据过滤
  rewrite        query 改写
  store          MySQL 读写
  vector         Qdrant
```

## 已知限制

- **BM25 的平均文档长度用的是常数**（100 个词），没有全库统计。影响的是长度归一化，不影响 IDF。
- **向量维度固定为 1024**（bge-m3）。换别的维度的模型，需要新建一个 Qdrant 集合（换个 `QDRANT_COLLECTION` 名字）并重建索引。
- **原文件存在 MySQL 的 LONGBLOB 里**，适合单个文件在 50 MB 以内。量大以后建议改存对象存储。
- **限流是单实例内存计数**。多副本部署时，每个副本各自限流，所以实际上限是「副本数 × rps」。
- **管理界面把 Key 存在浏览器 localStorage 里**，只适合内网使用。
- **编码识别的盲区**：泰文（TIS-620/windows-874）、越南文（windows-1258）这类老式单字节编码识别不出来，内容很短的文件（只有一两句）也判断不准，这些情况都会用回退编码解码。另存为 UTF-8 就没有这个问题。

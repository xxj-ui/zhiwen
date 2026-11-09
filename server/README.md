# server/ 目录地图

后端只有一个包，按**职责**分 5 个目录。看一眼目录就知道框架长什么样：

```
server/
├── app.py        唯一装配点：FastAPI 实例、lifespan、异常处理、中间件、路由注册
├── main.py       CLI 入口（python -m server.main）：预览 / --build / --ask
│
├── core/         横切与基础设施（与业务无关）
│   config.py       参数集中读取（DSN、端口、top_k、切分、上限）
│   errors.py       错误码目录 + AppError
│   logging.py      日志格式与 request_id
│   timeutil.py     库内 UTC → 接口 +08:00
│   storage.py      原文件落盘、转正、静默删除
│
├── rag/          领域内核：纯算法，不碰数据库、不碰 HTTP
│   loaders.py      六种格式解析
│   splitter.py     句子聚合切分（chunk_index 文档级连续）
│   embedder.py     512 维向量化
│   keyword_search.py  分词与 tsquery 构造（打分交给 PG）
│   rrf.py          RRF 融合公式（纯函数）
│   llm.py          DeepSeek 接入（提示词 + 流式）
│
├── db/           数据访问：所有 SQL 只在这里
│   pool.py         连接池（min=1 / max=4 / timeout=5 s）
│   documents.py    表 documents
│   chunks.py       表 rag_chunks（向量检索 + 按文档写入）
│   conversations.py 表 conversations / messages（M2）
│
├── api/          HTTP 契约：路径、参数校验、响应组装
│   routes/         documents / chat / conversations / health
│   schemas.py      请求 / 响应模型
│   deps.py         容器：仓储 + 连接池 + 索引 + 模型状态位
│
└── services/     编排：顺序、事务边界、副作用、错误映射
    documents.py    上传 / 列表 / 删除
    retrieval.py    两路召回 + RRF 融合（供 chat 复用）
    chat.py         问答 SSE（M2）
    history.py      会话历史（M2）
    health.py       三项健康探测
```

## 依赖方向（单向，禁止回指）

```
app.py → api/ → services/ → db/ ─┐
                        └→ rag/ ─┴→ core/
```

- `rag/` **禁止** import `psycopg` / `fastapi` / `server.api` / `server.db` / `server.services`
- `db/` 只写 SQL，不做业务判断；事务由调用方传入 `conn`
- `services/` 不 import `fastapi`；编排、事务、错误分类都在这一层
- `api/` 只做参数校验与响应组装，不写 SQL、不做编排

## 一次上传请求的旅程

```
POST /api/documents/upload
  → api/routes/documents.py      取文件、检查数量、包信封
  → services/documents.py        校验 → 落盘 → 解析 → 切分 → 向量化 → 转正 → 事务写入
        ├→ core/storage.py       临时文件、rename、删旧文件
        ├→ rag/loaders.py        解析
        ├→ rag/splitter.py       切分
        ├→ rag/embedder.py       向量化
        └→ db/documents.py + db/chunks.py   同一事务：upsert 文档、全量替换片段、更新 chunk_count
  → api/schemas.py               组装响应
```

## 新代码放哪（四条口诀）

1. 有 `SELECT` / `INSERT` / `DELETE` → `db/`
2. 纯计算，给输入出结果 → `rag/`
3. 决定"先做什么后做什么、事务从哪到哪、失败算什么错" → `services/`
4. 只跟路径、参数、JSON 打交道 → `api/`

配置项一律进 `core/config.py`，不要散落在各模块里。

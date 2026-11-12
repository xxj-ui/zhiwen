# 知问 zhiwen

单机运行的个人知识库问答。上传文档，提问，得到带引用来源的流式回答。数据全部留在本机，只有生成模型走外部 API。

![问答界面](docs/images/qa-demo.png)

## 核心功能

| 能力 | 说明 |
| --- | --- |
| 上传解析 | PDF / Word / Markdown / TXT / Excel / CSV，扫描件自动 OCR |
| 混合检索 | 向量召回 + PostgreSQL 全文关键词，RRF 按名次融合 |
| 带引用回答 | 流式输出，每条结论标注来源片段；无依据时明确拒答 |
| 多轮追问 | 保留最近 5 轮上下文 |
| 文档管理 | 列表、删除、片段只读预览 |

## 设计与取舍

**向量库就用 pgvector，不引入独立服务**
单用户单机场景，8,935 块规模下 HNSW 查询 4.04 ms。独立向量库（Milvus / Qdrant）要多一套服务和一份运维成本，换不回这点时间；真到单机撑不住那天再换也不迟。

**混合检索用 RRF，不给分数调权重**
dense 与关键词的分数量纲不可比，线性加权要反复调参。RRF 只用名次融合，两路各取 top-20，零调参，且单路召回失效时另一路仍能兜住。

**扫描件 PDF 走本地 OCR，不调云 API**
RapidOCR + PP-OCRv4，依赖增量 26 MB，字符相似度 0.99，1.1~2.9 s/页；带文字层的 PDF 完全不受影响（0.09 s 走原路径）。代价是首次 0.6 s 引擎初始化。

## 架构

依赖单向流动，只有 `services` 同时接触算法层与仓储层：

```text
web/                  React + Vite + TS
  │  HTTP / SSE
server/api/           路由与请求校验（不写 SQL）
  │
server/services/      业务编排
  │
  ├─ server/rag/      纯算法：解析 / 切分 / 嵌入 / 关键词 / RRF / 生成
  └─ server/db + core 全部 SQL · 配置 / 错误 / 日志 / 存储
```

一次问答经过：问句 → 嵌入 → dense top-k + 关键词 top-k → RRF 融合 → 上下文拼装 → DeepSeek 流式 → SSE（`citations` → `delta`* → `done`）。

## 文档入库流程

每个文件走完八步才进知识库。解析按格式分两支：PDF 逐页判质量、不达标才 OCR；其他五种直接调对应解析库。任一步失败只影响这一个文件，旧数据与旧原文件保持不动。

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 200, "nodeSpacing": 24, "rankSpacing": 34, "padding": 10, "curve": "basis"}}}%%
flowchart TB

    subgraph ROW1["接收与解析 ①~③"]
        direction LR
        FE(["前端知识库页<br/>逐个文件提交"])
        S1["① 校验<br/>文件名 · 扩展名<br/>大小 · 文件数"]
        S2["② 中转落盘<br/>data/tmp<br/>超 50 MB 中断"]
        DISP["③ 解析<br/>按扩展名分发"]
        P["【PDF】逐页抽文字层<br/>不达标改走 OCR<br/>两路不过则该页跳过<br/>可用页不足整份拒绝"]
        O["【其他五种】<br/>按扩展名用 python 库解析<br/>编码按 UTF-8 → GB18030<br/>解不通则回退"]
        FE --> S1 --> S2 --> DISP
        DISP -->|"pdf"| P
        DISP -->|"其他"| O
    end

    subgraph ROW2[" "]
        direction LR
        S4["④ 切分<br/>size 500<br/>overlap 100"]
        S5["⑤ 向量化<br/>bge-small-zh-v1.5<br/>512 维"]
        S6["⑥ 新文件入库<br/>uploads ／ MinIO<br/>rename ／ put_object"]
        S7["⑦ 事务写入<br/>documents<br/>+ 替换 chunks"]
        S8["⑧ 删旧版本<br/>仅覆盖同名文档时<br/>回收旧原文件"]
        OK(["入库成功<br/>导入即可检索"])
        S4 --> S5 --> S6 --> S7 --> S8 --> OK
    end

    ROW1 -->|"产出的文档单元"| ROW2

    classDef step fill:#e0f2fe,stroke:#0284c7,color:#0c4a6e
    classDef pdf fill:#fde68a,stroke:#d97706,color:#78350f
    classDef oth fill:#e9d5ff,stroke:#7c3aed,color:#3b0764
    classDef ok fill:#dcfce7,stroke:#16a34a,color:#14532d
    class S1,S2,DISP,S4,S5,S6,S7,S8 step
    class P pdf
    class O oth
    class OK ok
```

图源：[docs/diagrams/知识库入库流程.mmd](docs/diagrams/知识库入库流程.mmd)。

## 提问检索流程

一次提问走完六步：先校验与拼检索 query，两路召回后按 RRF 融合取 top-5；有召回就先发引用编号再流式生成，无召回按固定话术回绝但仍记历史。生成失败或客户端断开则本轮不落库。

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 200, "nodeSpacing": 24, "rankSpacing": 34, "padding": 10, "curve": "basis"}}}%%
flowchart TB

    subgraph B1["提问与召回 ①~③"]
        direction LR
        FE(["前端问答页<br/>POST /api/chat"])
        V["① 校验<br/>非空 · ≤500 字<br/>续聊先验会话存在"]
        H["② 会话锁 + 取历史<br/>最近 5 轮 · ≤4000 字<br/>query = 上一轮问题<br/>+ 当前问题"]
        R["③ 两路召回 + RRF<br/>向量 top-20（HNSW）<br/>关键词 top-20（tsvector）<br/>融合取 top-5<br/>检索失败 → error 事件"]
        FE --> V --> H --> R
    end

    subgraph B2["回答与落库 ④~⑥"]
        direction LR
        Q{"④ 召回到片段？"}
        C["发 citations 事件<br/>编号此刻固定"]
        FB["发固定话术 delta<br/>不调大模型"]
        G["⑤ 生成回答<br/>流式发 delta<br/>首字节 15 s / 总 60 s"]
        ERR["error 事件<br/>本轮不落库"]
        SAVE["⑥ 整轮落库<br/>问题 + 回答<br/>+ 引用快照"]
        DONE(["done 事件<br/>返回 conversation_id"])
        Q -->|"有"| C --> G --> SAVE
        Q -->|"无"| FB --> SAVE
        G -.->|"超时 · 异常 · Key 缺失<br/>或客户端断开"| ERR
        SAVE --> DONE
    end

    B1 -->|"top-5 片段"| B2

    classDef step fill:#e0f2fe,stroke:#0284c7,color:#0c4a6e
    classDef dec fill:#fde68a,stroke:#d97706,color:#78350f
    classDef bad fill:#fee2e2,stroke:#dc2626,color:#7f1d1d
    classDef ok fill:#dcfce7,stroke:#16a34a,color:#14532d
    class V,H,R,C,FB,G,SAVE step
    class Q dec
    class ERR bad
    class DONE ok
```

图源：[docs/diagrams/提问检索流程.mmd](docs/diagrams/提问检索流程.mmd)。

## 实测数据

| 项 | 结果 | 说明 |
| --- | --- | --- |
| 检索（280 块） | **20.7 ms** | dense + PG 全文 + RRF（热态 n=15 中位）；8,680 块低 df 6.56 ms |
| 向量检索单项 | **4.04 ms** | HNSW，8,935 块 |
| SSE 首字节 | **1.64 s** | 含检索与 DeepSeek 首 token |
| 启动同步初始化 | **18.2~23.0 s** | 解析库 0.5~1.2 / 模型 8.0~11.3（BM25 构建已移除） |
| 扫描件 OCR | **1.09~2.07 s/页** | 120 dpi，字符相似度 0.99 |
| 检索质量 | **Recall@5 83.1%** · faithfulness 0.963 | 259 题，RAGAS + DeepSeek 判分 |

## 快速开始

```bash
cp .env.example .env         # 填 POSTGRES_PASSWORD / MINIO_ROOT_* / DEEPSEEK_API_KEY
docker compose up -d         # PostgreSQL(pgvector) + MinIO，自动建桶
pip install -r requirements.txt
python scripts/migrate.py apply
scripts/run_dev.sh           # 后端；预热约 20 s 后开始接收请求

cd web && npm install && npm run dev    # 另开终端，访问 http://127.0.0.1:5173
```

需要 Python 3.11+、PostgreSQL 15+（装 pgvector）、Node.js 18+。首次运行会下载 `BAAI/bge-small-zh-v1.5`（约 95 MB）到 `HF_HOME`。全部环境变量见 [.env.example](.env.example)。

## 界面

**知识库**：上传文档后自动解析、切分、向量化。

![知识库](docs/images/knowledge-base.png)

**片段预览**：点开任一文档，逐片段查看入库原文。

![片段预览](docs/images/preview.png)

## 技术栈

| 层 | 选型 |
| --- | --- |
| 后端 | Python 3.11+ / FastAPI / Uvicorn |
| 数据库 | PostgreSQL 17 + pgvector 0.8.5 |
| 嵌入 | `BAAI/bge-small-zh-v1.5`（CPU，512 维） |
| 检索 | dense + PostgreSQL `tsvector` + RRF |
| 生成 | DeepSeek（流式） |
| 前端 | React 18 + Vite + TypeScript |
| 存储 | MinIO（S3 兼容）/ 本地目录 |

## 项目结构

```text
server/             后端单包，分层见上方架构图（地图见 server/README.md）
web/                前端：问答 / 知识库两页
docs/               需求、SRS、设计 00~11、规范
scripts/            迁移、启动脚本、建表 SQL
evals/              评测数据集
tools/eval/         评测脚本（采集 / 指标 / RAGAS）
```

## 设计文档

按 SDLC 阶段推进，每个关键决策都有据可查：

- **需求** — [00 需求分析](docs/00.需求分析.md)（角色 / 场景 / 用例）、[SRS](docs/SRS.md)（FR / NFR / BR / AC + 接口与数据契约）
- **设计** — [01 功能模块](docs/design/01.功能模块梳理.md)、[02 整体架构](docs/design/02.整体架构设计.md)、[03 数据库](docs/design/03.数据库设计.md)、[04 API](docs/design/04.API接口设计.md)、[09 前端](docs/design/09.前端详细设计.md)、[10 多轮会话](docs/design/10.多轮会话设计.md)、[11 RAG 评测](docs/design/11.RAG评测设计.md)
- **过程** — [05 评审结果](docs/design/05.设计方案评审结果.md)、[06 遗留问题](docs/design/06.遗留问题清单.md)、[07 测试用例](docs/design/07.测试用例清单.md)、[08 性能基线](docs/design/08.性能基线实测.md)
- **规范** — [API 规则](docs/rules/api-rules.md)、[数据库规则](docs/rules/database-rules.md)

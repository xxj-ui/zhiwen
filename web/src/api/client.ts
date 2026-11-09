/**
 * 04 §3.2 统一信封的解包处；错误码→中文文案的映射也只在这里。
 * 组件不直接 fetch（02 §4.3）。
 */
export interface Envelope<T> {
  code: string
  message: string
  data: T | null
}

/** 错误码目录（04 §3.3，20 个业务码）→ 可直接展示的文案 */
export const ERROR_TEXT: Record<string, string> = {
  VALIDATION_ERROR: '参数校验失败',
  QUESTION_EMPTY: '问题不能为空',
  QUESTION_TOO_LONG: '问题不能超过 500 字',
  TOO_MANY_FILES: '每次只能上传 1 个文件',
  PAYLOAD_TOO_LARGE: '请求体过大',
  UNSUPPORTED_FILE_TYPE: '不支持的文件类型',
  FILE_TOO_LARGE: '文件超过 50 MB',
  EMPTY_FILE: '文件为空',
  INVALID_FILENAME: '非法文件名',
  PARSE_FAILED: '无法解析文件内容，可能是扫描件或空文档',
  TABLE_EMPTY: '表格为空或无有效行',
  EMBEDDING_FAILED: '向量化失败，请稍后重试',
  STORAGE_FAILED: '文件保存失败',
  DOCUMENT_NOT_FOUND: '文档不存在',
  CONVERSATION_NOT_FOUND: '会话不存在',
  DB_UNAVAILABLE: '数据库不可用，请检查数据库服务',
  LLM_KEY_MISSING: '未配置大模型 API Key',
  LLM_TIMEOUT: '大模型响应超时，请重试',
  LLM_ERROR: '大模型调用失败，请重试',
  INTERNAL_ERROR: '服务内部错误',
}

export class ApiError extends Error {
  constructor(
    readonly code: string,
    message: string,
    readonly status: number,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

const BASE = '/api'

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(BASE + path, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
  })
  const body = (await res.json().catch(() => null)) as Envelope<T> | null
  if (!res.ok || !body || body.code !== 'ok') {
    const code = body?.code ?? 'INTERNAL_ERROR'
    throw new ApiError(code, ERROR_TEXT[code] ?? body?.message ?? '请求失败', res.status)
  }
  return body.data as T
}

/**
 * 上传专用：multipart 不做 JSON header。
 * 一次请求提交整批（1~20 个，04 v0.6 / FR-1）：后端按提交顺序串行处理，
 * 并在**整批结束后重建一次关键词索引**再返回 —— 所以前端要批量提交，
 * 否则 N 个文件会走 N 次重复的入库流程。
 */
export async function uploadMany(files: File[]): Promise<UploadItem[]> {
  const form = new FormData()
  for (const f of files) form.append('files', f)
  const res = await fetch(BASE + '/documents/upload', { method: 'POST', body: form })
  const body = (await res.json().catch(() => null)) as Envelope<{ items: UploadItem[] }> | null
  if (!res.ok || !body || body.code !== 'ok' || !body.data) {
    const code = body?.code ?? 'INTERNAL_ERROR'
    throw new ApiError(code, ERROR_TEXT[code] ?? body?.message ?? '上传失败', res.status)
  }
  return body.data.items
}

export interface UploadItem {
  filename: string
  status: 'success' | 'failed'
  document_id: number | null
  chunks: number
  overwritten: boolean
  error: { code: string; message: string } | null
}

export interface DocumentItem {
  id: number
  filename: string
  file_type: string
  file_size: number
  chunk_count: number
  created_at: string
  updated_at: string
}

export interface CitationItem {
  n: number
  filename: string
  page: string | null
  snippet: string
  chunk_id: number
}

export interface DocumentChunkItem {
  chunk_id: number
  chunk_index: number
  page: string | null
  content: string
}

export interface DocumentBrief {
  id: number
  filename: string
  file_type: string
  file_size: number
  chunk_count: number
  updated_at: string
}

export interface DocumentChunks {
  document: DocumentBrief
  total: number
  offset: number
  limit: number
  items: DocumentChunkItem[]
}

export interface ConversationListItem {
  id: number
  title: string
  created_at: string
}

export interface ConversationDetail {
  id: number
  title: string
  created_at: string
  messages: { role: 'user' | 'assistant'; content: string; citations: CitationItem[] }[]
}

/** API-2 列表（无独立统计接口，统计由调用方现算：04 Q4） */
export async function listDocuments(): Promise<DocumentItem[]> {
  const data = await request<{ items: DocumentItem[] }>('/documents')
  return data.items
}

/** API-3 删除文档：成功返回 200 + data:null */
export async function deleteDocument(documentId: number): Promise<void> {
  await request<null>(`/documents/${documentId}`, { method: 'DELETE' })
}

/** API-9 文档片段预览（只读，按 chunk_index 顺序分页） */
export function listDocumentChunks(
  documentId: number,
  offset = 0,
  limit = 20,
): Promise<DocumentChunks> {
  return request<DocumentChunks>(`/documents/${documentId}/chunks?offset=${offset}&limit=${limit}`)
}

/** API-5 历史列表 */
export async function listConversations(limit = 50, offset = 0): Promise<ConversationListItem[]> {
  const data = await request<{ items: ConversationListItem[] }>(
    `/conversations?limit=${limit}&offset=${offset}`,
  )
  return data.items
}

/** API-6 历史详情 */
export function getConversation(conversationId: number): Promise<ConversationDetail> {
  return request<ConversationDetail>(`/conversations/${conversationId}`)
}

/** API-7 删除历史 */
export async function deleteConversation(conversationId: number): Promise<void> {
  await request<null>(`/conversations/${conversationId}`, { method: 'DELETE' })
}

export interface HealthData {
  status: 'ok' | 'degraded'
  database: 'ok' | 'error'
  embedding_model: 'loaded' | 'loading' | 'failed'
  warmup: 'idle' | 'warming' | 'ready' | 'failed'
  llm_key: 'configured' | 'missing'
}

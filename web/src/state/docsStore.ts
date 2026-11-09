/**
 * 文档域状态（02 S-07）：列表 + 上传队列
 * 上传口径：一次请求提交整批（1~20 个，04 v0.6 / FR-1）。
 * 后端整批处理完才返回（V003 后已无索引重建），仍保持批量提交以减少往返：
 * 逐个 POST 会让每个文件都触发一次全量重建（N 个文件 = N 次 20 s）。
 */
import { create } from 'zustand'
import { deleteDocument, listDocuments, uploadMany } from '../api/client'
import type { DocumentItem } from '../api/client'

export type QueueStatus = 'pending' | 'uploading' | 'success' | 'failed'

export interface QueueItem {
  id: number
  name: string
  size: number
  status: QueueStatus
  message?: string
}

interface DocsState {
  items: DocumentItem[]
  loading: boolean
  queue: QueueItem[]
  busy: boolean
  load: () => Promise<void>
  uploadFiles: (files: File[]) => Promise<void>
  remove: (id: number) => Promise<void>
  clearQueue: () => void
}

let seq = 0

export const useDocsStore = create<DocsState>((set, get) => ({
  items: [],
  loading: false,
  queue: [],
  busy: false,

  async load() {
    set({ loading: true })
    try {
      set({ items: await listDocuments() })
    } finally {
      set({ loading: false })
    }
  },

  async uploadFiles(files: File[]) {
    const accepted = files.slice(0, 20)          // 单次最多 20 个，多的忽略
    const queued: QueueItem[] = accepted.map((f) => ({
      id: ++seq, name: f.name, size: f.size, status: 'pending' as const,
    }))
    const ids = new Set(queued.map((q) => q.id))
    set((s) => ({
      queue: [...queued, ...s.queue].map((q) =>
        ids.has(q.id) ? { ...q, status: 'uploading' as const } : q,
      ),
      busy: true,
    }))

    try {
      // 整批一次请求：后端串行处理并在整批结束后重建一次关键词索引，
      // 返回的 items[] 与提交顺序一一对应
      const results = await uploadMany(accepted)
      const succeeded: number[] = []
      set((s) => ({
        queue: s.queue.map((q) => {
          const idx = queued.findIndex((x) => x.id === q.id)
          if (idx < 0) return q
          const res = results[idx]
          if (!res) return { ...q, status: 'failed' as const, message: '服务端未返回该文件的结果' }
          if (res.status === 'success') {
            succeeded.push(q.id)
            return {
              ...q,
              status: 'success' as const,
              message: `${res.chunks} 个片段${res.overwritten ? ' · 已覆盖同名文档' : ''}`,
            }
          }
          return { ...q, status: 'failed' as const, message: res.error?.message ?? '上传失败' }
        }),
      }))
      // 成功行给 3 s 回执（能看到"N 个片段"），随后自动移出队列：
      // 队列只用来展示"还没落地的活儿"，成功的文档已经出现在下方列表里。
      // 失败行保留，用户要看原因、自己决定何时清。
      if (succeeded.length > 0) {
        setTimeout(() => {
          const done = new Set(succeeded)
          set((s) => ({ queue: s.queue.filter((q) => !done.has(q.id)) }))
        }, 3000)
      }
    } catch (err) {
      const message = (err as Error).message
      set((s) => ({
        queue: s.queue.map((q) => (ids.has(q.id) ? { ...q, status: 'failed' as const, message } : q)),
      }))
    }

    set({ busy: false })
    await get().load()
  },

  async remove(id: number) {
    await deleteDocument(id)
    await get().load()
  },

  clearQueue() {
    // 上传中清列表 ≠ 取消请求：请求仍会在服务端跑完并入库，会让用户误以为已取消。
    // 上传期间由 UploadQueue 按钮 disabled + 这里兜底，双双拦住。
    if (get().busy) return
    set({ queue: [] })
  },
}))

export const selectDocStats = (s: DocsState) => ({
  documentCount: s.items.length,
  chunkCount: s.items.reduce((sum, d) => sum + d.chunk_count, 0),
})

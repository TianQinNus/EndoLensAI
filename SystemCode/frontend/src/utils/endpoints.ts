import type {
  AnalysisStatus,
  BackendFrame,
  BackendKeyFrame,
  BackendReport,
  BackendVideo,
} from './dataQueries'

export const apiBaseUrl = (
  import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000/api/v1'
).replace(/\/$/, '')

export function assetUrl(path: string | null): string | undefined {
  if (!path) return undefined
  return path.startsWith('http')
    ? path
    : `${apiBaseUrl.replace(/\/api\/v1$/, '')}${path}`
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${apiBaseUrl}${path}`, {
    ...init,
    headers: { Accept: 'application/json', ...init?.headers },
  })
  if (!response.ok) {
    const detail = await response.text()
    throw new Error(detail || `Request failed (${response.status})`)
  }
  return response.json() as Promise<T>
}

export function uploadVideo(file: File): Promise<BackendVideo> {
  const form = new FormData()
  form.append('file', file)
  return request<BackendVideo>('/videos', { method: 'POST', body: form })
}

export function createAnalysis(videoId: string): Promise<{ task_id: string }> {
  return request<{ task_id: string }>('/analyses', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      video_id: videoId,
      sampling_mode: 'interval',
      interval_seconds: 1,
      model_version: 'unet-b-expanded-seed42-r98-v1',
    }),
  })
}

export const getAnalysisStatus = (taskId: string) =>
  request<AnalysisStatus>(`/analyses/${taskId}`)

export const getAnalysisFrames = (taskId: string) =>
  request<{ items: BackendFrame[] }>(`/analyses/${taskId}/frames?limit=200`)

export const getKeyFrames = (taskId: string) =>
  request<{ keyframes: BackendKeyFrame[] }>(`/analyses/${taskId}/keyframes`)

export const getReport = (taskId: string) =>
  request<BackendReport>(`/analyses/${taskId}/report`)

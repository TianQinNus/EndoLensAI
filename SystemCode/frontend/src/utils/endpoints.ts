import type { KeyFrame, PolypDetection, VideoMetadata } from './dataQueries'

const baseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${baseUrl}${path}`, { ...init, headers: { Accept: 'application/json', ...init?.headers } })
  if (!response.ok) throw new Error(`Request failed (${response.status})`)
  return response.json() as Promise<T>
}

export const getVideo = (videoId: string) => request<VideoMetadata>(`/videos/${videoId}`)
export const getDetections = (videoId: string) => request<PolypDetection[]>(`/videos/${videoId}/detections`)
export const getKeyFrames = (videoId: string) => request<KeyFrame[]>(`/videos/${videoId}/keyframes`)
export const generateReport = (videoId: string) => request<Blob>(`/videos/${videoId}/report`, { method: 'POST' })
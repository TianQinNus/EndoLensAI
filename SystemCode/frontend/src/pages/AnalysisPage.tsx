import { useCallback, useEffect, useState } from 'react'
import { DetectionPanel } from '../components/DetectionPanel'
import { KeyFrames } from '../components/KeyFrames'
import { VideoPlayer } from '../components/VideoPlayer'
import { VideoTimeline } from '../components/VideoTimeline'
import {
  assetUrl,
  createAnalysis,
  getAnalysisFrames,
  getAnalysisStatus,
  getKeyFrames,
  uploadVideo,
} from '../utils/endpoints'
import type {
  AnalysisStatus,
  BackendFrame,
  KeyFrame,
  PolypDetection,
  VideoMetadata,
} from '../utils/dataQueries'

function toVideo(value: Awaited<ReturnType<typeof uploadVideo>>): VideoMetadata {
  return {
    id: value.video_id,
    name: value.filename,
    duration: value.duration_seconds,
    fps: value.fps ?? 0,
    streamUrl: assetUrl(value.video_url) ?? value.video_url,
    demo: false,
  }
}

function toDetection(frame: BackendFrame): PolypDetection {
  return {
    id: frame.frame_id,
    frameIndex: frame.frame_index,
    timestamp: frame.timestamp_seconds,
    confidence: frame.foreground_score ?? 0,
    contour: frame.contours[0]?.map(([x, y]) => ({ x, y })) ?? [],
    status: 'Candidate Polyp',
    maskUrl: assetUrl(frame.mask_url),
    sourceUrl: assetUrl(frame.source_frame_url),
  }
}

function toKeyFrame(frame: {
  frame_id: string
  timestamp_seconds: number
  thumbnail_url: string | null
}): KeyFrame {
  return {
    id: frame.frame_id,
    timestamp: frame.timestamp_seconds,
    label: formatTime(frame.timestamp_seconds),
    thumbnailUrl: assetUrl(frame.thumbnail_url),
  }
}

function AnalysisPage() {
  const [video, setVideo] = useState<VideoMetadata>()
  const [task, setTask] = useState<AnalysisStatus>()
  const [detections, setDetections] = useState<PolypDetection[]>([])
  const [frames, setFrames] = useState<KeyFrame[]>([])
  const [selectedId, setSelectedId] = useState('')
  const [currentTime, setCurrentTime] = useState(0)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const refreshResults = useCallback(async (taskId: string) => {
    const [framePage, keyFramePage] = await Promise.all([
      getAnalysisFrames(taskId),
      getKeyFrames(taskId),
    ])
    setDetections(
      framePage.items
        .filter((frame) => frame.status === 'completed' && frame.is_candidate)
        .map(toDetection),
    )
    setFrames(keyFramePage.keyframes.map(toKeyFrame))
  }, [])

  useEffect(() => {
    if (!task || ['completed', 'failed', 'cancelled'].includes(task.status)) return
    const timer = window.setInterval(() => {
      void getAnalysisStatus(task.task_id)
        .then((nextTask) => {
          setTask(nextTask)
          if (nextTask.status === 'completed') {
            void refreshResults(nextTask.task_id).catch((reason: Error) =>
              setError(reason.message),
            )
          }
        })
        .catch((reason: Error) => setError(reason.message))
    }, 1000)
    return () => window.clearInterval(timer)
  }, [refreshResults, task])

  const handleUpload = async (file: File) => {
    setBusy(true)
    setError('')
    try {
      const uploaded = await uploadVideo(file)
      const created = await createAnalysis(uploaded.video_id)
      setVideo(toVideo(uploaded))
      setTask(await getAnalysisStatus(created.task_id))
      setDetections([])
      setFrames([])
      setSelectedId('')
      setCurrentTime(0)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Unable to start analysis')
    } finally {
      setBusy(false)
    }
  }

  const handleSelect = useCallback((detection: PolypDetection) => {
    setSelectedId(detection.id)
    setCurrentTime(detection.timestamp)
  }, [])

  const selected = detections.find((item) => item.id === selectedId) ?? detections[0]

  return (
    <div>
      <section className="mb-2 flex flex-wrap items-center gap-3 rounded-md border border-slate-200 bg-white p-3 shadow-sm">
        <label className="cursor-pointer rounded bg-[#1677f2] px-3 py-2 text-xs font-medium text-white">
          {busy ? 'Uploading…' : 'Upload H264 MP4'}
          <input
            className="hidden"
            type="file"
            accept=".mp4,video/mp4"
            disabled={busy}
            onChange={(event) => {
              const file = event.target.files?.[0]
              if (file) void handleUpload(file)
              event.target.value = ''
            }}
          />
        </label>
        {video && <span className="text-xs text-slate-600">{video.name}</span>}
        {task && (
          <span className="rounded bg-slate-100 px-2 py-1 text-xs text-slate-700">
            {task.status} ·{' '}
            {task.progress.percent == null
              ? task.phase
              : `${task.progress.percent.toFixed(0)}%`}
          </span>
        )}
        {error && <span className="text-xs text-red-600">{error}</span>}
      </section>
      {!video ? (
        <div className="rounded-md border border-dashed border-slate-300 bg-white p-12 text-center text-sm text-slate-500">
          Upload a recorded H264 MP4 to begin analysis.
        </div>
      ) : (
        <>
          <div className="grid gap-[9px] lg:grid-cols-[minmax(450px,2fr)_minmax(330px,1fr)]">
            <VideoPlayer
              video={video}
              detection={selected}
              currentTime={currentTime}
              onTimeChange={setCurrentTime}
            />
            <DetectionPanel
              detections={detections}
              selectedId={selected?.id ?? ''}
              onSelect={handleSelect}
            />
          </div>
          <VideoTimeline
            currentTime={currentTime}
            detections={detections}
            onSeek={setCurrentTime}
            duration={video.duration}
          />
          <KeyFrames frames={frames} selectedTime={currentTime} onSelect={setCurrentTime} />
        </>
      )}
    </div>
  )
}

function formatTime(seconds: number) {
  return `${Math.floor(seconds / 60).toString().padStart(2, '0')}:${Math.floor(
    seconds % 60,
  )
    .toString()
    .padStart(2, '0')}`
}

export default AnalysisPage

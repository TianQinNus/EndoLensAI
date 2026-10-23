import { useCallback, useEffect, useState } from 'react'
import { DetectionPanel } from '../components/DetectionPanel'
import { KeyFrames } from '../components/KeyFrames'
import { VideoPlayer } from '../components/VideoPlayer'
import { VideoTimeline } from '../components/VideoTimeline'
import { getMockDetections, getMockKeyFrames, getMockVideo, type KeyFrame, type PolypDetection, type VideoMetadata } from '../utils/dataQueries'

function AnalysisPage() {
  const [video, setVideo] = useState<VideoMetadata>()
  const [detections, setDetections] = useState<PolypDetection[]>([])
  const [frames, setFrames] = useState<KeyFrame[]>([])
  const [selectedId, setSelectedId] = useState('polyp-001')
  const [currentTime, setCurrentTime] = useState(252)
  useEffect(() => { void Promise.all([getMockVideo(), getMockDetections('case-001'), getMockKeyFrames('case-001')]).then(([metadata, items, keyFrames]) => { setVideo(metadata); setDetections(items); setFrames(keyFrames) }) }, [])
  const handleSelect = useCallback((detection: PolypDetection) => { setSelectedId(detection.id); setCurrentTime(detection.timestamp) }, [])
  if (!video) return <div className="rounded-md border border-slate-200 bg-white p-[13px] shadow-sm">Loading demo analysis…</div>
  const selected = detections.find((item) => item.id === selectedId)
  return <div><div className="grid gap-[9px] lg:grid-cols-[minmax(450px,2fr)_minmax(330px,1fr)]"><VideoPlayer video={video} detection={selected} currentTime={currentTime} onTimeChange={setCurrentTime} /><DetectionPanel detections={detections} selectedId={selectedId} onSelect={handleSelect} /></div><VideoTimeline currentTime={currentTime} detections={detections} onSeek={setCurrentTime} /><KeyFrames frames={frames} selectedTime={currentTime} onSelect={setCurrentTime} /></div>
}
export default AnalysisPage

import { useEffect, useRef, useState } from 'react'
import type { PolypDetection, VideoMetadata } from '../utils/dataQueries'
interface VideoPlayerProps { video: VideoMetadata; detection?: PolypDetection; currentTime: number; onTimeChange: (time: number) => void }
export function VideoPlayer({ video, detection, currentTime, onTimeChange }: VideoPlayerProps) {
  const [playing, setPlaying] = useState(false)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  useEffect(() => { const timer = playing ? window.setInterval(() => onTimeChange(Math.min(video.duration, currentTime + 1)), 1000) : undefined; return () => { if (timer) window.clearInterval(timer) } }, [playing, currentTime, onTimeChange, video.duration])
  useEffect(() => {
    const canvas = canvasRef.current
    const context = canvas?.getContext('2d')
    if (!canvas || !context) return
    canvas.width = 800
    canvas.height = 450
    context.clearRect(0, 0, canvas.width, canvas.height)
    if (!detection) return
    context.strokeStyle = '#15ed84'
    context.lineWidth = 3
    context.beginPath()
    detection.contour.forEach((point, index) => index === 0 ? context.moveTo(point.x, point.y) : context.lineTo(point.x, point.y))
    context.closePath()
    context.stroke()
  }, [detection, currentTime])
  return <section className="overflow-hidden rounded-md bg-[#0b1724] p-[9px]"><div className="relative aspect-[1.68] overflow-hidden rounded-md bg-black"><video className="absolute inset-0 h-full w-full object-contain" src={video.streamUrl} muted playsInline aria-label={video.name} /><canvas ref={canvasRef} className="pointer-events-none absolute inset-0 z-[6] h-full w-full" aria-label="Detection contour overlay" /><div className="absolute bottom-3 left-3 z-[5] rounded bg-[rgba(7,17,29,.72)] px-[11px] py-[7px] font-mono text-[11px] leading-[1.5] text-white">Frame: {detection?.frameIndex ?? '—'}<br />Time: {formatTime(currentTime)}<br />AI result, not a diagnosis</div>{detection?.maskUrl && <img className="pointer-events-none absolute inset-0 z-[4] h-full w-full object-contain opacity-35 mix-blend-screen" src={detection.maskUrl} alt="AI segmentation mask" />}</div><div className="flex items-center gap-3.5 px-2 pt-[9px] text-xs text-[#e9f1f8]"><button className="bg-transparent p-0 text-[21px] text-white" onClick={() => setPlaying((value) => !value)} aria-label={playing ? 'Pause' : 'Play'}>{playing ? 'Ⅱ' : '▶'}</button><span>{formatTime(currentTime)} / {formatTime(video.duration)}</span><input className="h-[5px] flex-1 accent-[#2e83f8]" type="range" min="0" max={video.duration} value={currentTime} onChange={(event) => onTimeChange(Number(event.target.value))} aria-label="Video progress" /><span>{video.fps ? `${video.fps.toFixed(1)} FPS` : 'FPS —'}</span></div></section>
}
function formatTime(seconds: number) { return `${Math.floor(seconds / 60).toString().padStart(2, '0')}:${Math.floor(seconds % 60).toString().padStart(2, '0')}` }

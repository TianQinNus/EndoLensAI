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
  return <section className="overflow-hidden rounded-md bg-[#0b1724] p-[9px]"><div className="relative aspect-[1.68] overflow-hidden rounded-md bg-[radial-gradient(ellipse_at_67%_52%,#ef684e_0,#ba3c2e_25%,#7f291f_43%,#301916_67%,#070b0f_100%)]"><div className="absolute inset-[-15%] rotate-[-12deg] bg-[repeating-radial-gradient(ellipse_at_55%_48%,transparent_0_22px,rgba(255,205,182,.18)_24px_26px,transparent_29px_51px)] opacity-70" /><div className="absolute inset-0 bg-[radial-gradient(ellipse_at_35%_75%,rgba(255,193,161,.35),transparent_31%),radial-gradient(ellipse_at_85%_20%,rgba(255,214,193,.3),transparent_22%)] mix-blend-screen" /><video className="absolute inset-0 h-full w-full opacity-[.02]" muted playsInline aria-label="Demo colonoscopy video" /><canvas ref={canvasRef} className="pointer-events-none absolute inset-0 z-[6] h-full w-full" aria-label="Detection contour overlay" /><div className="absolute left-2.5 top-2.5 z-[3] h-[82px] w-[86px] rounded-md border border-white/15 bg-[rgba(30,20,16,.75)] text-center text-[58px] leading-[78px] text-[#dbe5dd]">◎</div><div className="absolute left-[49%] top-[47%] z-[4] h-40 w-[175px] rotate-[-10deg] rounded-[47%_52%_45%_48%] border-2 border-[#15ed84] shadow-[0_0_3px_rgba(0,255,135,.8)]" /><div className="absolute left-[83%] top-[62%] z-[4] h-[45px] w-[43px] rounded-full border-2 border-[#15ed84]" /><span className="absolute left-[54%] top-[34%] z-[5] rounded bg-[#14e879] px-1.5 py-0.5 text-xs font-bold text-[#052c1a]">Polyp {detection?.confidence.toFixed(2) ?? '0.92'}</span><span className="absolute left-[84%] top-[57%] z-[5] rounded bg-[#14e879] px-1.5 py-0.5 text-xs font-bold text-[#052c1a]">0.78</span><div className="absolute bottom-3 left-3 z-[5] rounded bg-[rgba(31,18,15,.72)] px-[11px] py-[7px] font-mono text-[11px] leading-[1.5] text-white">Frame: {detection?.frameIndex ?? 842} / 2,418<br />Time: {formatTime(currentTime)}<br />DEMO DATA</div><div className="absolute right-2 top-2 z-[7] flex flex-col gap-1.5">{['⌕', '⌕', '⛶', '▧'].map((icon) => <button key={icon} className="h-[34px] w-[34px] rounded-md border border-[#415267] bg-[rgba(7,17,29,.75)] text-lg text-white" type="button">{icon}</button>)}</div></div><div className="flex items-center gap-3.5 px-2 pt-[9px] text-xs text-[#e9f1f8]"><button className="bg-transparent p-0 text-[21px] text-white" onClick={() => setPlaying((value) => !value)} aria-label={playing ? 'Pause' : 'Play'}>{playing ? 'Ⅱ' : '▶'}</button><span>{formatTime(currentTime)} / 12:30</span><input className="h-[5px] flex-1 accent-[#2e83f8]" type="range" min="0" max={video.duration} value={currentTime} onChange={(event) => onTimeChange(Number(event.target.value))} aria-label="Video progress" /><span>🔊</span><span>1.0x</span><span>⛶</span></div></section>
}
function formatTime(seconds: number) { return `${Math.floor(seconds / 60).toString().padStart(2, '0')}:${Math.floor(seconds % 60).toString().padStart(2, '0')}` }

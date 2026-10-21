import {useEffect, useRef} from "react";
import {assetUrl, type FrameResult} from "./endpoints"; // adjust import after merge

// Exact saved RGB frame, not an arbitrarily seeked HTML-video frame.
// Canvas bitmap stays in native frame coordinates. CSS changes only display size.
export function FrameOverlay({frame}: {frame: FrameResult}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    let cancelled = false;
    const target = canvas.current;
    if (!target) return;
    target.width = frame.width; target.height = frame.height;
    const ctx = target.getContext("2d")!; ctx.clearRect(0,0,target.width,target.height);
    const load = (url: string) => new Promise<HTMLImageElement>((resolve,reject) => {
      const image = new Image(); image.crossOrigin = "anonymous";
      image.onload = () => resolve(image); image.onerror = reject; image.src = assetUrl(url);
    });
    if (!frame.source_frame_url) return;
    (async () => {
      const source = await load(frame.source_frame_url!);
      if (cancelled) return; ctx.drawImage(source,0,0);
      if (frame.status !== "completed" || !frame.mask_url) return;
      const mask = await load(frame.edited_mask_url ?? frame.mask_url);
      if (cancelled) return;
      const layer = document.createElement("canvas"); layer.width=frame.width; layer.height=frame.height;
      const lc = layer.getContext("2d")!; lc.drawImage(mask,0,0);
      const rgba=lc.getImageData(0,0,layer.width,layer.height);
      for (let i=0;i<rgba.data.length;i+=4) {
        const foreground=rgba.data[i]>127;
        rgba.data[i]=0; rgba.data[i+1]=240; rgba.data[i+2]=110; rgba.data[i+3]=foreground?85:0;
      }
      lc.putImageData(rgba,0,0); ctx.drawImage(layer,0,0);
    })().catch(error => {if (!cancelled) console.error("Frame overlay failed",error);});
    return () => {cancelled=true;};
  },[frame.frame_id,frame.mask_url,frame.edited_mask_url,frame.source_frame_url,frame.width,frame.height,frame.status]);
  return <figure style={{margin:0}}>
    <figcaption>{frame.edited_mask_url ? "Human-reviewed mask" : "Original AI mask"}</figcaption>
    <canvas ref={canvas} aria-label="Saved frame with segmentation overlay"
      style={{display:"block",width:"100%",height:"auto",maxWidth:"100%"}}/>
  </figure>;
}

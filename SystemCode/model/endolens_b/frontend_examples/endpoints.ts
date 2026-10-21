// Merge these functions into existing src/utils/endpoints.ts; do not replace UI.
export const MODEL_VERSION = "unet-b-expanded-seed42-r98-v1";
const BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000/api/v1";
export type FrameResult = {
  frame_id: string; frame_index: number; timestamp_seconds: number;
  width: number; height: number; status: "completed" | "failed";
  is_candidate: boolean | null; max_region_area_ratio: number | null;
  source_frame_url: string | null; thumbnail_url: string | null; mask_url: string | null;
  contours: number[][][]; foreground_score: number | null;
  review_status: "unreviewed" | "reviewed" | "rejected";
  revision_id: string | null; edited_mask_url?: string;
};
async function request(path: string, init?: RequestInit) {
  const response = await fetch(`${BASE}${path}`, init);
  if (!response.ok) throw new Error(`${response.status}: ${await response.text()}`);
  return response.json();
}
export function assetUrl(path: string): string {
  return new URL(path, new URL(BASE, window.location.origin).origin).href;
}
export async function uploadVideo(file: File) {
  const body = new FormData(); body.append("file", file);
  return request("/videos", { method: "POST", body }); // no manual multipart boundary
}
export function startAnalysis(videoId: string, intervalSeconds = 1) {
  return request("/analyses", { method: "POST", headers: {"Content-Type":"application/json"},
    body: JSON.stringify({video_id: videoId, sampling_mode: "interval",
      interval_seconds: intervalSeconds, model_version: MODEL_VERSION}) });
}
export const getAnalysis = (taskId: string, signal?: AbortSignal) => request(`/analyses/${taskId}`, {signal});
export const cancelAnalysis = (taskId: string) => request(`/analyses/${taskId}/cancel`, {method:"POST"});
export const getKeyframes = (taskId: string) => request(`/analyses/${taskId}/keyframes`);
export const getReport = (taskId: string) => request(`/analyses/${taskId}/report`);
export function getFrames(taskId: string, after: string | null = null, candidateOnly = false, signal?: AbortSignal) {
  const query = new URLSearchParams({limit:"100",candidate_only:String(candidateOnly)});
  if (after !== null) query.set("after", after);
  return request(`/analyses/${taskId}/frames?${query}`, {signal});
}
export function saveReview(frame: FrameResult, mask: Blob, notes = "") {
  const body = new FormData(); body.append("edited_mask", mask, "mask.png");
  body.append("review_status", "reviewed"); body.append("notes", notes);
  if (frame.revision_id) body.append("base_revision_id", frame.revision_id);
  return request(`/frames/${frame.frame_id}/review`, {method:"PUT",body});
}
export function exportCsv(taskId: string) {
  return request(`/analyses/${taskId}/exports`, {method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({format:"csv",include_reviewed_masks:true})});
}
export async function followAnalysis(taskId: string,
  onUpdate: (status: any, frames: FrameResult[]) => void, signal: AbortSignal) {
  let cursor: string | null = null;
  const saved = new Map<string,FrameResult>();
  try {
  while (!signal.aborted) {
    const status = await getAnalysis(taskId,signal);
    while (true) {
      const page = await getFrames(taskId,cursor,false,signal);
      page.items.forEach((frame: FrameResult) => saved.set(frame.frame_id,frame));
      if (page.last_cursor !== null) cursor = page.last_cursor;
      if (!page.has_more) break;
    }
    if (signal.aborted) return;
    onUpdate(status,[...saved.values()].sort((a,b) => a.frame_index-b.frame_index));
    if (["completed","failed","cancelled"].includes(status.status)) return;
    await new Promise<void>(resolve => {
      const done = () => {clearTimeout(timer);signal.removeEventListener("abort",done);resolve();};
      const timer = setTimeout(done,1500);signal.addEventListener("abort",done,{once:true});
    });
  }
  } catch (error) {
    if (!signal.aborted) throw error;
  }
}

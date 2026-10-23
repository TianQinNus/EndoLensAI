export interface Point { x: number; y: number }
export interface PolypDetection { id: string; frameIndex: number; timestamp: number; confidence: number; contour: Point[]; status: 'Candidate Polyp' }
export interface KeyFrame { id: string; timestamp: number; label: string; detectionId?: string }
export interface VideoMetadata { id: string; name: string; duration: number; fps: number; demo: boolean }

export const mockVideo: VideoMetadata = { id: 'case-001', name: 'Demo Case', duration: 750, fps: 28.7, demo: true }
export const mockDetections: PolypDetection[] = [
  { id: 'polyp-001', frameIndex: 842, timestamp: 252, confidence: .92, status: 'Candidate Polyp', contour: [{ x: 390, y: 220 }, { x: 470, y: 190 }, { x: 560, y: 230 }, { x: 590, y: 330 }, { x: 510, y: 390 }, { x: 410, y: 350 }] },
  { id: 'polyp-002', frameIndex: 856, timestamp: 268, confidence: .78, status: 'Candidate Polyp', contour: [{ x: 650, y: 280 }, { x: 700, y: 260 }, { x: 735, y: 310 }, { x: 700, y: 350 }, { x: 655, y: 330 }] },
  { id: 'polyp-003', frameIndex: 1089, timestamp: 363, confidence: .65, status: 'Candidate Polyp', contour: [{ x: 420, y: 210 }, { x: 480, y: 205 }, { x: 520, y: 260 }, { x: 470, y: 315 }, { x: 415, y: 275 }] },
]
export const mockKeyFrames: KeyFrame[] = [
  { id: 'frame-1', timestamp: 236, label: '03:56' }, { id: 'frame-2', timestamp: 248, label: '04:08', detectionId: 'polyp-001' },
  { id: 'frame-3', timestamp: 252, label: '04:12', detectionId: 'polyp-001' }, { id: 'frame-4', timestamp: 268, label: '04:28', detectionId: 'polyp-002' },
  { id: 'frame-5', timestamp: 315, label: '05:15' }, { id: 'frame-6', timestamp: 363, label: '06:03', detectionId: 'polyp-003' },
  { id: 'frame-7', timestamp: 441, label: '07:21' }, { id: 'frame-8', timestamp: 526, label: '08:46' },
]
export const mockReport = { title: 'Colonoscopy Analysis Report', caseName: mockVideo.name, generatedAt: 'Demo output', summary: '3 candidate regions flagged for review', model: 'E0 reference / E4 candidate comparison' }
export const getMockVideo = async (): Promise<VideoMetadata> => mockVideo
export const getMockDetections = async (videoId: string): Promise<PolypDetection[]> => { void videoId; return mockDetections }
export const getMockKeyFrames = async (videoId: string): Promise<KeyFrame[]> => { void videoId; return mockKeyFrames }

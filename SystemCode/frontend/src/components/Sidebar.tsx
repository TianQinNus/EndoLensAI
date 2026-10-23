import type { AppPage } from '../App'
interface SidebarProps { page: AppPage; onNavigate: (page: AppPage) => void }
export function Sidebar({ page, onNavigate }: SidebarProps) {
  const itemClass = 'flex w-full items-center gap-2.5 rounded-md border-0 px-[11px] py-2.5 text-left text-xs text-[#b9c9dc] transition-colors hover:bg-[#203852]'
  return <aside className="flex w-[134px] flex-none flex-col bg-gradient-to-b from-[#10243a] to-[#102236] px-[7px] py-3.5 max-[850px]:w-[66px] max-[850px]:basis-[66px]"><button className={`${itemClass} ${page === 'analysis' ? 'bg-[#1677f2] text-white shadow-[0_4px_12px_rgba(0,0,0,.18)]' : ''}`} onClick={() => onNavigate('analysis')}><span className="w-4 text-center text-[17px]">▶</span><span className="max-[850px]:hidden">Live Analysis</span></button><button className={`${itemClass} ${page === 'reports' ? 'bg-[#1677f2] text-white shadow-[0_4px_12px_rgba(0,0,0,.18)]' : ''}`} onClick={() => onNavigate('reports')}><span className="w-4 text-center text-[17px]">▤</span><span className="max-[850px]:hidden">Reports</span></button></aside>
}

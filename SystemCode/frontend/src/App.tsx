import { useState } from 'react'
import { Header } from './components/Header'
import { Sidebar } from './components/Sidebar'
import AnalysisPage from './pages/AnalysisPage'
import ReportsPage from './pages/ReportsPage'

export type AppPage = 'analysis' | 'reports'

function App() {
  const [page, setPage] = useState<AppPage>('analysis')

  return (
    <div className="min-h-screen bg-slate-50">
      <Header />
      <div className="flex min-h-[calc(100vh-50px)]">
        <Sidebar page={page} onNavigate={setPage} />
        <main className="min-w-0 flex-1 px-2.5 pb-[18px] pt-[9px]">
          {page === 'analysis' ? <AnalysisPage /> : <ReportsPage />}
        </main>
      </div>
    </div>
  )
}

export default App

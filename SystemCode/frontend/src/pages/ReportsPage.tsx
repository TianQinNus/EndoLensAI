function ReportsPage() {
  return (
    <div className="mx-auto max-w-[1100px]">
      <section className="mb-2 rounded-lg bg-gradient-to-r from-[#112a47] to-[#1d5f9b] p-5 text-white">
        <h1 className="mb-1 text-[22px]">Analysis Reports</h1>
        <p className="text-xs text-[#c8e0f9]">
          Reports are generated from completed backend analysis tasks.
        </p>
      </section>
      <section className="rounded-md border border-slate-200 bg-white p-4 text-sm text-slate-600 shadow-sm">
        Complete an analysis on the Live Analysis page to view its report and
        export metadata from the backend API.
      </section>
    </div>
  )
}

export default ReportsPage

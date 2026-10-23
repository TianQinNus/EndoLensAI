interface ReportCardProps { label: string; value: string; detail?: string }
export function ReportCard({ label, value, detail }: ReportCardProps) { return <div className="rounded-md border border-slate-200 bg-white p-4 shadow-sm"><div className="text-[28px] font-bold text-[#1b6fda]">{value}</div><div className="text-[11px] text-[#66758a]">{label}</div>{detail && <small>{detail}</small>}</div> }

import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'CRESIP — Coastal Reservoir Intelligence Platform',
  description: 'Real-time satellite fill analysis for 2,333 coastal reservoirs across 82 countries',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  )
}

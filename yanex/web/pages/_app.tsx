import type { AppProps } from 'next/app'
import { Inter } from 'next/font/google'
import { Navbar } from '@/components/Navbar'
import '../styles/globals.css'

const inter = Inter({ subsets: ['latin'] })

export default function App({ Component, pageProps }: AppProps) {
  return (
    <div className={inter.className}>
      <div className="min-h-screen bg-gray-50">
        <Navbar />
        <main className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
          <Component {...pageProps} />
        </main>
      </div>
    </div>
  )
}

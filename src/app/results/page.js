'use client'
import { useSearchParams } from 'next/navigation'

export default function ResultsPage() {
    const searchParams = useSearchParams()
    const score = searchParams.get('score')

    return (
        <div style={{ padding: '2rem', fontFamily: 'Inter, sans-serif', textAlign: 'center' }}>
            <h1>Your predicted success score</h1>
            {score ? (
                <p style={{ fontSize: '2rem', fontWeight: 'bold' }}>{score}</p>
            ) : (
                <p>No score available.</p>
            )}
        </div>
    )
}

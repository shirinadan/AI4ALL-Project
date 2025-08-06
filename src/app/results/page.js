'use client'
import { useSearchParams } from 'next/navigation'
import '../globals.css'
import './page.css'
import '../layout.js'

export default function ResultsPage() {
    const searchParams = useSearchParams()
    const score = searchParams.get('score')

    return (
        <div style={{ padding: '2rem', fontFamily: 'Inter, sans-serif', textAlign: 'center' }}>
            <h1>Your predicted success score</h1>
            {score ? (
                <div className="score-wrapper">
                    <div className="outer-ring">
                        <div className="inner-circle">
                            <div className="score-number">{Number(score).toFixed(3)}</div>
                            <div className="score-label">Success Score</div>
                        </div>
                    </div>
                </div>
            ) : (
                <p className="no-score">No score available.</p>
            )}
            <button onClick={() => window.location.href = '/quiz'}>Predict Again</button>
        </div>
    )
}

'use client'               // ⚠️ this makes useState, useRouter, etc. available
import { useState } from 'react'
import { useRouter } from 'next/navigation'
import './page.css'

const QUESTIONS = [
  {
    text: "What industry is your startup in?",
    options: ["Technology", "Healthcare", "Finance", "E‑commerce"],
  },
  {
    text: "How many team members do you have?",
    options: ["1–3", "4–10", "11–20", "20+"],
  },
  {
    text: "What stage of funding are you at?",
    options: ["Pre‑seed", "Seed", "Series A", "Series B+"],
  },
]

export default function QuizPage() {
  const [step, setStep] = useState(0)
  const [answers, setAnswers] = useState([])
  const router = useRouter()

  const handleSelect = (opt) => {
    const next = [...answers, opt]
    setAnswers(next)

    if (step < QUESTIONS.length - 1) {
      setStep(step + 1)
    } else {
      console.log("Quiz answers:", next)
      router.push('/')     // or push('/results') once you build that page
    }
  }

  const percent = Math.round(((step + 1) / QUESTIONS.length) * 100)

  return (
    <div className="quiz-page">
      <div className="quiz-card">
        <div className="quiz-progress">
          <div className="quiz-progress-bar" style={{ width: `${percent}%` }} />
        </div>

        <h2 className="quiz-question">{QUESTIONS[step].text}</h2>

        <div className="quiz-options">
          {QUESTIONS[step].options.map((o) => (
            <button
              key={o}
              className="quiz-option"
              onClick={() => handleSelect(o)}
            >
              {o}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
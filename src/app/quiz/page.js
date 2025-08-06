'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import '../globals.css'
import '../layout.js'
import './page.css'

const FUNDING_TYPES = [
  'seed',
  'venture',
  'undisclosed',
  'convertible note',
  'debt financing',
  'angel',
  'post ipo equity',
  'post ipo debt',
  'secondary market',
  'product crowdfunding'
]

const ROUNDS = [
  'round A',
  'round B',
  'round C',
  'round D',
  'round E',
  'round F',
  'round G',
  'round H',
  'round I',
  'round J'
]

const MARKETS = [
  'Software',
  'Health Care',
  'Finance',
  'Education',
  'Energy',
  'Mobile',
  'Advertising',
  'Biotechnology'
]

const COUNTRIES = [
  'USA', 'CAN', 'GBR', 'IND', 'DEU', 'FRA', 'CHN', 'Unknown'
]

const STATES = [
  'CA', 'NY', 'TX', 'MA', 'WA', 'FL', 'IL', 'Unknown'
]

export default function BizLensQuiz() {
  const [currentStep, setCurrentStep] = useState(0)
  const [fundingTypes, setFundingTypes] = useState([])
  const [foundedYear, setFoundedYear] = useState('')
  const [roundDepth, setRoundDepth] = useState('')
  const [market, setMarket] = useState('')
  const [countryCode, setCountryCode] = useState('')
  const [stateCode, setStateCode] = useState('')

  const questions = [
    {
      title: "Which funding types apply?",
      subtitle: "(select all that apply)",
      type: "funding"
    },
    {
      title: "In what year was your startup founded?",
      type: "year"
    },
    {
      title: "What is the funding round depth?",
      type: "depth"
    },
    {
      title: "What is your primary market category?",
      type: "market"
    },
    {
      title: "Country code of headquarters?",
      type: "country"
    },
    {
      title: "State code of headquarters?",
      type: "state"
    }
  ]

  function toggleFunding(type) {
    setFundingTypes((prev) =>
      prev.includes(type) ? prev.filter((t) => t !== type) : [...prev, type]
    )
  }

  function handleNext() {
    if (currentStep < questions.length - 1) {
      setCurrentStep(currentStep + 1)
    } else {
      handleSubmit()
    }
  }

  const router = useRouter()

  async function handleSubmit() {
    const payload = {
      funding_types: fundingTypes,
      founded_year: Number(foundedYear),
      round_depth: Number(roundDepth),
      market,
      country_code: countryCode,
      state_code: stateCode
    }

    try {
      const res = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })

      const data = await res.json()
      if (res.ok && data && typeof data.prediction !== 'undefined') {
        router.push(`/results?score=${encodeURIComponent(data.prediction)}`)
      } else {
        console.error('Unexpected response:', data)
      }
    } catch (err) {
      console.error('Submission failed:', err)
    }
  }

  function canProceed() {
    switch (currentStep) {
      case 0: return fundingTypes.length > 0
      case 1: return foundedYear !== ''
      case 2: return roundDepth !== ''
      case 3: return market !== ''
      case 4: return countryCode !== ''
      case 5: return stateCode !== ''
      default: return false
    }
  }

  const currentQuestion = questions[currentStep]

  return (
    <div className="quiz-wrapper">
      <div className="quiz-container">
        <div style={{ height: '6px', background: 'rgba(255,255,255,0.2)', borderRadius: '3px', marginBottom: '1.5rem', width: '100%' }}>
          <div style={{
            width: `${((currentStep + 1) / questions.length) * 100}%`,
            height: '100%',
            borderRadius: '3px',
            background: 'white',
            transition: 'width 0.3s ease'
          }} />
        </div>

        <h2 className="question-title">{currentQuestion.title}</h2>
        {currentQuestion.subtitle && <p className="question-subtitle">{currentQuestion.subtitle}</p>}

        <div className="answers-container">
          {currentQuestion.type === 'funding' && (
            <div className="funding-grid">
              {FUNDING_TYPES.map((type) => (
                <button
                  key={type}
                  onClick={() => toggleFunding(type)}
                  className={`funding-button${fundingTypes.includes(type) ? ' selected' : ''}`}
                >
                  {type}
                </button>
              ))}
            </div>
          )}

          {currentQuestion.type === 'year' && (
            <input
              type="number"
              min="1900"
              max={new Date().getFullYear()}
              value={foundedYear}
              onChange={(e) => setFoundedYear(e.target.value)}
              placeholder="e.g. 2018"
              className="year-input"
            />
          )}

          {currentQuestion.type === 'depth' && (
            <input
              type="number"
              min="0"
              max="10"
              value={roundDepth}
              onChange={(e) => setRoundDepth(e.target.value)}
              placeholder="e.g. 2"
              className="year-input"
            />
          )}

          {currentQuestion.type === 'market' && (
            <select value={market} onChange={(e) => setMarket(e.target.value)} className="market-select">
              <option value="">— select one —</option>
              {MARKETS.map((cat) => <option key={cat} value={cat}>{cat}</option>)}
            </select>
          )}

          {currentQuestion.type === 'country' && (
            <select value={countryCode} onChange={(e) => setCountryCode(e.target.value)} className="market-select">
              <option value="">— select one —</option>
              {COUNTRIES.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          )}

          {currentQuestion.type === 'state' && (
            <select value={stateCode} onChange={(e) => setStateCode(e.target.value)} className="market-select">
              <option value="">— select one —</option>
              {STATES.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          )}
        </div>

        <div className="navigation">
          <button
            onClick={() => setCurrentStep(Math.max(0, currentStep - 1))}
            disabled={currentStep === 0}
            className={`nav-button${currentStep === 0 ? ' disabled' : ''}`}
          >
            Back
          </button>

          <span className="progress-text">{currentStep + 1} of {questions.length}</span>

          <button
            onClick={handleNext}
            disabled={!canProceed()}
            className={`next-button${!canProceed() ? ' disabled' : ''}`}
          >
            {currentStep === questions.length - 1 ? 'Submit' : 'Next'}
          </button>
        </div>
      </div>
    </div>
  )
}

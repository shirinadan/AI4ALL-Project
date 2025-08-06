'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import '../globals.css'
import '../layout.js'
import './page.css'


const INDUSTRIES = [
  'Technology',
  'Healthcare', 
  'Finance',
  'E-commerce',
  'Manufacturing',
  'Education',
  'Real Estate',
  'Retail',
  'Transportation',
  'Entertainment'
]

const FUNDING_TYPES = [
  'Seed',
  'Convertible Note',
  'Series A',
  'Series B',
  'Series C',
  'Series D',
  'Series E',
  'Debt Financing',
  'Grant',
  'Product Crowdfunding',
  'Equity Crowdfunding',
  'Private Equity',
  'Post‑IPO Equity',
  'Angel',
  'Undisclosed',
  'Venture',
]

const MARKET_CATS = [
  'Social Television',
  'Enterprise Search',
  'Reviews & Recommendations',
  'Biomass Power Generation',
  'Minerals',
  'Parenting',
  'Transaction Processing',
  'Hardware',
  'Auto',
  'Health Services Industry',
]

export default function BizLensQuiz() {
  const [currentStep, setCurrentStep] = useState(0)
  const [selectedIndustry, setSelectedIndustry] = useState('')
  const [fundingTypes, setFundingTypes] = useState([])
  const [foundedYear, setFoundedYear] = useState('')
  const [market, setMarket] = useState('')

  const questions = [
    {
      title: "What industry is your startup in?",
      type: "industry"
    },
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
      title: "What is your primary market category?",
      type: "market"
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
      console.log((currentStep + 1) / questions.length * 100 + '% completed')
    } else {
      console.log("got here")
      handleSubmit()
    }
  }

  const router = useRouter()

  async function handleSubmit() {
    const payload = {
      industry: selectedIndustry,
      fundingTypes,
      foundedYear,
      market
    }

    console.log(payload);

    try {
      const res = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })

      const data = await res.json()
      console.log("Prediction response:", data)

      if (res.ok && data && typeof data.success_score !== 'undefined') {
        router.push(`/results?score=${encodeURIComponent(data.success_score)}`)
      }
      else {
        console.error('Unexpected response:', data)
      }
    } catch (err) {
      console.error('Submission failed:', err)
    }
  }

  function canProceed() {
    switch (currentStep) {
      case 0: return selectedIndustry !== ''
      case 1: return fundingTypes.length > 0
      case 2: return foundedYear !== ''
      case 3: return market !== ''
      default: return false
    }
  }

  const currentQuestion = questions[currentStep]

  return (
    <div className="quiz-wrapper">
      <div className="quiz-container">
{/* Progress Bar */}
        <div style={{
          height: '6px',
          background: 'rgba(255,255,255,0.2)',
          borderRadius: '3px',
          // overflow: 'hidden',
          marginBottom: '1.5rem',
          width: '100%',
        }}>
          <div style={{
            width: `${((currentStep + 1) / questions.length) * 100}%`,
            height: '100%',
            borderRadius: '3px',
            background: 'rgb(255, 255, 255)',
            transition: 'width 0.3s ease'
          }} />
        </div>

        {/* Question */}
        <h2 className="question-title">
          {currentQuestion.title}
        </h2>

        {currentQuestion.subtitle && (
          <p className="question-subtitle">
            {currentQuestion.subtitle}
          </p>
        )}

        {/* Answer Options */}
        <div className="answers-container">
          {currentQuestion.type === 'industry' && INDUSTRIES.map((industry) => (
            <button
              key={industry}
              onClick={() => setSelectedIndustry(industry)}
              className={`answer-button${selectedIndustry === industry ? ' selected' : ''}`}
              onMouseEnter={(e) => {
                if (selectedIndustry !== industry) {
                  e.target.classList.add('hover-temp')
                }
              }}
              onMouseLeave={(e) => {
                if (selectedIndustry !== industry) {
                  e.target.classList.remove('hover-temp')
                }
              }}
            >
              {industry}
            </button>
          ))}

          {currentQuestion.type === 'funding' && (
            <div className="funding-grid">
              {FUNDING_TYPES.map((type) => (
                <button
                  key={type}
                  onClick={() => toggleFunding(type)}
                  className={`funding-button${fundingTypes.includes(type) ? ' selected' : ''}`}
                  onMouseEnter={(e) => {
                    if (!fundingTypes.includes(type)) {
                      e.target.classList.add('hover-temp')
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (!fundingTypes.includes(type)) {
                      e.target.classList.remove('hover-temp')
                    }
                  }}
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

          {currentQuestion.type === 'market' && (
            <select
              value={market}
              onChange={(e) => setMarket(e.target.value)}
              className={`market-select${market ? ' has-value' : ''}`}
            >
              <option value="">— select one —</option>
              {MARKET_CATS.map((cat) => (
                <option key={cat} value={cat}>
                  {cat}
                </option>
              ))}
            </select>
          )}
        </div>

        {/* Navigation */}
        <div className="navigation">
          <button
            onClick={() => setCurrentStep(Math.max(0, currentStep - 1))}
            disabled={currentStep === 0}
            className={`nav-button${currentStep === 0 ? ' disabled' : ''}`}
          >
            Back
          </button>

          <span className="progress-text">
            {currentStep + 1} of {questions.length}
          </span>

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


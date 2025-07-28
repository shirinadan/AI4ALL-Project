'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'

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
    } else {
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

    try {
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })

      const data = await res.json()

      if (res.ok && data && typeof data.score !== 'undefined') {
        router.push(`/results?score=${encodeURIComponent(data.score)}`)
      } else {
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
    <div style={{
      margin: 0,
      padding: '2rem',
      fontFamily: 'Inter, sans-serif',
      background: `radial-gradient(circle at 20% 30%, #3b82f6 0%, transparent 50%),
                   radial-gradient(circle at 80% 60%, #9333ea 0%, transparent 50%),
                   linear-gradient(135deg, #1e3a8a, #4f46e5, #9333ea)`,
      backgroundBlendMode: 'screen',
      minHeight: '100vh',
      width: '100vw',
      boxSizing: 'border-box',
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'flex-start',
      paddingTop: '2rem'
    }}>
      <div style={{
        maxWidth: '480px',
        width: '100%',
        background: 'rgba(255,255,255,0.06)',
        backdropFilter: 'blur(8px)',
        borderRadius: '12px',
        padding: '2rem'
      }}>
        {/* Progress Bar */}
        <div style={{
          height: '6px',
          background: 'rgba(255,255,255,0.2)',
          borderRadius: '3px',
          overflow: 'hidden',
          marginBottom: '1.5rem'
        }}>
          <div style={{
            width: `${((currentStep + 1) / questions.length) * 100}%`,
            height: '100%',
            background: 'rgb(255, 255, 255)',
            transition: 'width 0.3s ease'
          }} />
        </div>

        {/* Question */}
        <h2 style={{
          color: 'white',
          fontSize: '1.5rem',
          marginBottom: '1rem',
          textAlign: 'center'
        }}>
          {currentQuestion.title}
        </h2>
        {currentQuestion.subtitle && (
          <p style={{ 
            color: 'rgba(255,255,255,0.8)', 
            marginBottom: '1rem', 
            fontSize: '1rem',
            textAlign: 'center'
          }}>
            {currentQuestion.subtitle}
          </p>
        )}

        {/* Answer Options */}
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          gap: '1rem',
          marginBottom: '2rem'
        }}>
          {currentQuestion.type === 'industry' && INDUSTRIES.map((industry) => (
            <button
              key={industry}
              onClick={() => setSelectedIndustry(industry)}
              style={{
                padding: '16px 24px',
                background: selectedIndustry === industry ? '#e0e7ff' : 'white',
                color: selectedIndustry === industry ? '#3730a3' : '#374151',
                border: 'none',
                borderRadius: '8px',
                fontWeight: '500',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
                textAlign: 'center',
                fontSize: '1rem'
              }}
              onMouseEnter={(e) => {
                if (selectedIndustry !== industry) {
                  e.target.style.background = '#f0f0f0'
                }
              }}
              onMouseLeave={(e) => {
                if (selectedIndustry !== industry) {
                  e.target.style.background = 'white'
                }
              }}
            >
              {industry}
            </button>
          ))}

          {currentQuestion.type === 'funding' && (
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
              gap: '12px'
            }}>
              {FUNDING_TYPES.map((type) => (
                <button
                  key={type}
                  onClick={() => toggleFunding(type)}
                  style={{
                    padding: '12px 16px',
                    background: fundingTypes.includes(type) ? '#e0e7ff' : 'white',
                    color: fundingTypes.includes(type) ? '#3730a3' : '#374151',
                    border: 'none',
                    borderRadius: '8px',
                    fontWeight: '500',
                    cursor: 'pointer',
                    transition: 'all 0.2s ease',
                    textAlign: 'center',
                    fontSize: '0.9rem'
                  }}
                  onMouseEnter={(e) => {
                    if (!fundingTypes.includes(type)) {
                      e.target.style.background = '#f0f0f0'
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (!fundingTypes.includes(type)) {
                      e.target.style.background = 'white'
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
              style={{
                padding: '16px 24px',
                background: 'white',
                color: '#374151',
                border: 'none',
                borderRadius: '8px',
                fontSize: '1rem',
                fontWeight: '500',
                textAlign: 'center',
                outline: 'none',
                width: '100%',
                boxSizing: 'border-box'
              }}
            />
          )}

          {currentQuestion.type === 'market' && (
            <select
              value={market}
              onChange={(e) => setMarket(e.target.value)}
              style={{
                padding: '16px 24px',
                background: 'white',
                color: market ? '#374151' : '#9ca3af',
                border: 'none',
                borderRadius: '8px',
                fontSize: '1rem',
                fontWeight: '500',
                textAlign: 'center',
                outline: 'none',
                cursor: 'pointer',
                width: '100%',
                boxSizing: 'border-box'
              }}
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
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <button
            onClick={() => setCurrentStep(Math.max(0, currentStep - 1))}
            disabled={currentStep === 0}
            style={{
              padding: '12px 24px',
              background: 'transparent',
              color: currentStep === 0 ? 'rgba(255,255,255,0.4)' : 'white',
              border: '2px solid',
              borderColor: currentStep === 0 ? 'rgba(255,255,255,0.4)' : 'white',
              borderRadius: '8px',
              fontSize: '1rem',
              fontWeight: '500',
              cursor: currentStep === 0 ? 'not-allowed' : 'pointer',
              transition: 'all 0.2s ease'
            }}
          >
            Back
          </button>

          <span style={{
            color: 'rgba(255,255,255,0.8)',
            fontSize: '0.9rem'
          }}>
            {currentStep + 1} of {questions.length}
          </span>

          <button
            onClick={handleNext}
            disabled={!canProceed()}
            style={{
              padding: '12px 24px',
              background: canProceed() ? 'white' : 'rgba(255,255,255,0.3)',
              color: canProceed() ? '#3730a3' : 'rgba(255,255,255,0.6)',
              border: 'none',
              borderRadius: '8px',
              fontSize: '1rem',
              fontWeight: canProceed() ? '600' : '500',
              cursor: canProceed() ? 'pointer' : 'not-allowed',
              transition: 'all 0.2s ease'
            }}
          >
            {currentStep === questions.length - 1 ? 'Submit' : 'Next'}
          </button>
        </div>
      </div>
    </div>
  )
}
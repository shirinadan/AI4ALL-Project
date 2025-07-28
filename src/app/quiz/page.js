
'use client'

import { useState } from 'react'

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

  function handleSubmit() {
    const payload = { 
      industry: selectedIndustry,
      fundingTypes, 
      foundedYear, 
      market 
    }
    console.log('Form submission:', payload)
    // TODO: send to your backend or navigate to results
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
      minHeight: '100vh',
      width: '100vw',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '20px',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
      margin: 0,
      boxSizing: 'border-box',
      overflow: 'auto'
    }}>
      <div style={{
        width: '100%',
        maxWidth: '480px',
        textAlign: 'center'
      }}>
        {/* Logo */}
        <h1 style={{
          color: 'white',
          fontSize: '2.5rem',
          fontWeight: '300',
          marginBottom: '3rem',
          letterSpacing: '1px'
        }}>
          BizLens
        </h1>

        {/* Progress Bar */}
        <div style={{
          width: '100%',
          height: '4px',
          backgroundColor: 'rgba(255,255,255,0.3)',
          borderRadius: '2px',
          marginBottom: '2rem',
          overflow: 'hidden'
        }}>
          <div style={{
            width: `${((currentStep + 1) / questions.length) * 100}%`,
            height: '100%',
            backgroundColor: 'white',
            transition: 'width 0.3s ease'
          }} />
        </div>

        {/* Question */}
        <div style={{
          marginBottom: '2rem'
        }}>
          <h2 style={{
            color: 'white',
            fontSize: '1.5rem',
            fontWeight: '500',
            marginBottom: '0.5rem',
            lineHeight: '1.3'
          }}>
            {currentQuestion.title}
          </h2>
          {currentQuestion.subtitle && (
            <p style={{
              color: 'rgba(255,255,255,0.8)',
              fontSize: '1rem',
              margin: 0
            }}>
              {currentQuestion.subtitle}
            </p>
          )}
        </div>

        {/* Answer Options */}
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          gap: '12px',
          marginBottom: '2rem'
        }}>
          {currentQuestion.type === 'industry' && INDUSTRIES.map((industry) => (
            <button
              key={industry}
              onClick={() => setSelectedIndustry(industry)}
              style={{
                padding: '16px 24px',
                backgroundColor: selectedIndustry === industry ? '#e0e7ff' : 'white',
                color: selectedIndustry === industry ? '#3730a3' : '#374151',
                border: 'none',
                borderRadius: '8px',
                fontSize: '1rem',
                fontWeight: '500',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
                textAlign: 'center'
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
                    backgroundColor: fundingTypes.includes(type) ? '#e0e7ff' : 'white',
                    color: fundingTypes.includes(type) ? '#3730a3' : '#374151',
                    border: 'none',
                    borderRadius: '8px',
                    fontSize: '0.9rem',
                    fontWeight: '500',
                    cursor: 'pointer',
                    transition: 'all 0.2s ease',
                    textAlign: 'center'
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
                backgroundColor: 'white',
                color: '#374151',
                border: 'none',
                borderRadius: '8px',
                fontSize: '1rem',
                fontWeight: '500',
                textAlign: 'center',
                outline: 'none'
              }}
            />
          )}

          {currentQuestion.type === 'market' && (
            <select
              value={market}
              onChange={(e) => setMarket(e.target.value)}
              style={{
                padding: '16px 24px',
                backgroundColor: 'white',
                color: market ? '#374151' : '#9ca3af',
                border: 'none',
                borderRadius: '8px',
                fontSize: '1rem',
                fontWeight: '500',
                textAlign: 'center',
                outline: 'none',
                cursor: 'pointer'
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
              backgroundColor: 'transparent',
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
              backgroundColor: canProceed() ? 'white' : 'rgba(255,255,255,0.3)',
              color: canProceed() ? '#3730a3' : 'rgba(255,255,255,0.6)',
              border: 'none',
              borderRadius: '8px',
              fontSize: '1rem',
              fontWeight: '600',
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
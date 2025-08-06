'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import '../globals.css'
import '../layout.js'
import './page.css'

const fundingTypeFields = [
  'seed',
  'venture',
  'undisclosed',
  'convertible_note',
  'debt_financing',
  'angel',
  'post_ipo_equity',
  'post_ipo_debt',
  'secondary_market',
  'product_crowdfunding'
]

const roundFields = [
  'round_A',
  'round_B',
  'round_C',
  'round_D',
  'round_E',
  'round_F',
  'round_G',
  'round_H',
  'round_I',
  'round_J'
]

export default function BizLensQuiz() {
  const [formData, setFormData] = useState({
    funding_rounds: 0,
    founded_year: 2012,
    seed: 0,
    venture: 0,
    undisclosed: 0,
    convertible_note: 0,
    debt_financing: 0,
    angel: 0,
    post_ipo_equity: 0,
    post_ipo_debt: 0,
    secondary_market: 0,
    product_crowdfunding: 0,
    round_A: 0,
    round_B: 0,
    round_C: 0,
    round_D: 0,
    round_E: 0,
    round_F: 0,
    round_G: 0,
    round_H: 0,
    round_I: 0,
    round_J: 0,
    round_depth: 0,
    market: 'Software',
    country_code: 'USA',
    state_code: 'CA'
  })

  const router = useRouter()

  function handleChange(e) {
    const { name, value, type, checked } = e.target
    setFormData(prev => ({
      ...prev,
      [name]:
        type === 'checkbox'
          ? (checked ? 1 : 0)
          : type === 'number'
          ? Number(value)
          : value
    }))
  }

  async function handleSubmit(e) {
    e.preventDefault()
    try {
      const res = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
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

  return (
    <div className="quiz-wrapper">
      <form className="quiz-container" onSubmit={handleSubmit}>
        <h2 className="question-title">Enter startup details</h2>

        <label>Funding Rounds
          <input
            type="number"
            name="funding_rounds"
            value={formData.funding_rounds}
            onChange={handleChange}
          />
        </label>

        <label>Founded Year
          <input
            type="number"
            name="founded_year"
            value={formData.founded_year}
            onChange={handleChange}
          />
        </label>

        <fieldset>
          <legend>Funding Types</legend>
          {fundingTypeFields.map(ft => (
            <label key={ft} style={{ display: 'block' }}>
              <input
                type="checkbox"
                name={ft}
                checked={formData[ft] === 1}
                onChange={handleChange}
              />
              {ft.replace(/_/g, ' ')}
            </label>
          ))}
        </fieldset>

        <fieldset>
          <legend>Rounds</legend>
          {roundFields.map(rf => (
            <label key={rf} style={{ display: 'block' }}>
              <input
                type="checkbox"
                name={rf}
                checked={formData[rf] === 1}
                onChange={handleChange}
              />
              {rf.replace('_', ' ')}
            </label>
          ))}
        </fieldset>

        <label>Round Depth
          <input
            type="number"
            name="round_depth"
            value={formData.round_depth}
            onChange={handleChange}
          />
        </label>

        <label>Market
          <select
            name="market"
            value={formData.market}
            onChange={handleChange}
          >
            <option value="Software">Software</option>
            <option value="Hardware">Hardware</option>
            <option value="Finance">Finance</option>
          </select>
        </label>

        <label>Country Code
          <select
            name="country_code"
            value={formData.country_code}
            onChange={handleChange}
          >
            <option value="USA">USA</option>
            <option value="CAN">CAN</option>
          </select>
        </label>

        <label>State Code
          <select
            name="state_code"
            value={formData.state_code}
            onChange={handleChange}
          >
            <option value="CA">CA</option>
            <option value="NY">NY</option>
            <option value="TX">TX</option>
          </select>
        </label>

        <button type="submit" className="answer-button">
          Predict
        </button>
      </form>
    </div>
  )
}


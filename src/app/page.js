import React from 'react';
import './globals.css';
import Link from 'next/link';
import './page.css';

export default function Home() {
  return (
    <div className='wrapper'>
      <h1>How successful is your startup?</h1>
      <p>Assess your startup's potential with our AI-powered evaluation tool.</p>
      <Link href="/quiz">
        <button>Start Evaluation</button>
      </Link>
    </div>
  );
}
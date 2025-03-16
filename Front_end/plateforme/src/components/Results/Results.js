import React, { useState } from 'react';
import './Results.css';

export default function Results({ prediction }) {
  const [feedback, setFeedback] = useState('');

  const handleFeedbackChange = (e) => {
    setFeedback(e.target.value);
  };

  const handleSubmitFeedback = () => {
    if (feedback.trim() === '') {
      alert("Please enter feedback before submitting.");
      return;
    }

    console.log("Doctor's Feedback Submitted:", feedback);
    alert("Feedback submitted successfully!");
    setFeedback('');
  };

  return (
    <div className="result-container">
      <div className="result-box">
        <h2 className="result-title">Transplant Success Prediction</h2>

        {prediction ? (
          <div className="prediction-result">
            <span className="result-label">Prediction:</span>
            <span className={`result-value ${prediction === 'Survive' ? 'success' : 'fail'}`}>
              {prediction === 'Survive' ? '✅ Survive' : '❌ Not Survive'}
            </span>
          </div>
        ) : (
          <p className="no-prediction">No prediction available. Please submit the form first.</p>
        )}

        <div className="feedback-section">
          <label className="feedback-label">Doctor's Feedback:</label>
          <textarea
            className="feedback-input"
            placeholder="Enter feedback about the prediction result"
            value={feedback}
            onChange={handleFeedbackChange}
          />
        </div>

        <button type="button" className="result-button" onClick={handleSubmitFeedback}>
          Submit Feedback
        </button>
      </div>
    </div>
  );
}

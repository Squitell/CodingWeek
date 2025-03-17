import React, { useState } from 'react';
import './Results.css';
import First from '../../assets/correlation_matrix.png';
import Second from '../../assets/XGBoost_feature_importance.png';
import three from  '../../assets/XGBoost_shap_summary.png';

export default function Results({ prediction }) {
  const [feedback, setFeedback] = useState('');
  const [plotsVisible, setPlotsVisible] = useState(true); // Show plots by default

  const handleFeedbackChange = (e) => setFeedback(e.target.value);

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
          <>
            <div className="prediction-result">
              <span className="result-label">Prediction:</span>
              <span className={`result-value ${prediction === 'Survive' ? 'success' : 'fail'}`}>
                {prediction === 'Survive' ? '✅ Survive' : '❌ Not Survive'}
              </span>
            </div>

            {/* Toggle Button for Plots */}
            <button className="plot-toggle-button" onClick={() => setPlotsVisible(!plotsVisible)}>
              {plotsVisible ? "Hide Plots 📉" : "Show Plots 📊"}
            </button>

            {/* Plots Section (Only Visible if toggled) */}
            {plotsVisible && (
              <div className="plots-container">
                <img src={First} alt="Survival Distribution" className="plot-image" />
                <img src={Second} alt="Feature Importance" className="plot-image" />
                <img src={three} alt="shap analysis" className="plot-image" />
              </div>
            )}
          </>
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

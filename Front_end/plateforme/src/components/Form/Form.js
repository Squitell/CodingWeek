import React, { useState } from 'react';
import axios from 'axios';
import './Form.css';

export default function Form({ setPrediction }) {
  const [formData, setFormData] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const response = await axios.post('http://localhost:8000/predict', formData);
      setPrediction(response.data.prediction);
    } catch (err) {
      setError("Error submitting form. Please try again.");
      console.error("Submission error:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="form-container">
      <form className="form-box" onSubmit={handleSubmit}>
        <h2 className="form-title">Patient Information</h2>

        {/* Name Input */}
        <label className="form-label">
          Name
          <input name="Name" type="text" className="form-input" placeholder="Enter patient's name" onChange={handleChange} />
        </label>

        {/* Dynamic Inputs */}
        {[
          { label: "Recipient Gender", name: "Recipientgender", options: [0, 1] },
          { label: "Stem Cell Source", name: "Stemcellsource", options: [0, 1] },
          { label: "Donor Age", name: "Donorage", numeric: true },
          { label: "Donor Age ≤ 35", name: "Donorage35", numeric: true },
          { label: "IIIV", name: "IIIV", options: [0, 1] },
          { label: "Gender Match", name: "Gendermatch", options: [0, 1] },
          { label: "Donor ABO", name: "DonorABO", options: [1, -1, 2, 0] },
          { label: "Recipient ABO", name: "RecipientABO", options: [1, -1, 2, 0] },
          { label: "Recipient Rh", name: "RecipientRh", options: [0, 1] },
          { label: "ABO Match", name: "ABOmatch", options: [0, 1] },
          { label: "CMV Status", name: "CMVstatus", options: [3, 2, 1, 0] },
          { label: "Donor CMV", name: "DonorCMV", options: [0, 1] },
          { label: "Recipient CMV", name: "RecipientCMV", options: [0, 1] },
          { label: "Disease", name: "Disease", options: ["ALL", "AML", "chronic", "nonmalignant", "lymphoma"] },
          { label: "Risk Group", name: "Riskgroup", options: [0, 1] },
          { label: "Tx Post Relapse", name: "Txpostrelapse", options: [0, 1] },
          { label: "Disease Group", name: "Diseasegroup", options: [0, 1] },
          { label: "HLA Match", name: "HLAmatch", options: [0, 1, 3, 2] },
          { label: "HLA Mismatch", name: "HLAmismatch", options: [0, 1] },
          { label: "Antigen", name: "Antigen", options: [-1, 1, 0, 2] },
          { label: "Alel", name: "Alel", options: [-1, 0, 2, 1, 3] },
          { label: "Recipient Age", name: "Recipientage", numeric: true },
          { label: "Recipient Age ≤ 10", name: "Recipientage10", numeric: true },
          { label: "Relapse", name: "Relapse", options: [0, 1] },
          { label: "aGvHD III IV", name: "aGvHDIIIIV", options: [0, 1] },
          { label: "ext cGvHD", name: "extcGvHD", options: [0, 1] },
          { label: "CD34 kg x10^6", name: "CD34kgx10d6", numeric: true },
          { label: "CD3 d CD34", name: "CD3dCD34", numeric: true },
          { label: "CD3 d kg x10^8", name: "CD3dkgx10d8", numeric: true },
          { label: "R Body Mass", name: "Rbodymass", numeric: true },
          { label: "ANC Recovery", name: "ANCrecovery", numeric: true },
          { label: "PLT Recovery", name: "PLTrecovery", numeric: true },
          { label: "Survival Time", name: "survival_time", numeric: true },
        ].map(({ label, name, options, numeric }) => (
          <label className="form-label" key={name}>
            {label}
            {numeric ? (
              <input name={name} type="number" className="form-input" onChange={handleChange} />
            ) : (
              <select name={name} className="form-input" onChange={handleChange}>
                <option value="">Select</option>
                {options.map(opt => (
                  <option key={opt} value={opt}>{opt}</option>
                ))}
              </select>
            )}
          </label>
        ))}

        {/* Show loading or error messages */}
        {loading && <p className="loading-message">Processing...</p>}
        {error && <p className="error-message">{error}</p>}

        <button type="submit" className="form-button" disabled={loading}>
          {loading ? "Submitting..." : "Submit Information"}
        </button>
      </form>
    </div>
  );
}

import React, { useState } from "react";
import "./App.css";

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState("");
  const [loading, setLoading] = useState(false);
  const [completed, setCompleted] = useState(false);

  const API_URL = import.meta.env.VITE_API_URL;

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setImage(file);
    setPreview(URL.createObjectURL(file));
    setPrediction("");
    setCompleted(false);
  };

  const handlePredict = async () => {
    if (!image || loading) return;

    setLoading(true);
    setPrediction("");

    const formData = new FormData();
    formData.append("file", image);

    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(text);
      }

      const data = await res.json();
      setPrediction(formatPrediction(data));
      setCompleted(true);
    } catch (err) {
      console.error("Prediction Error:", err);
      setPrediction("Server error. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const formatPrediction = (data) => {
    return (
      `Image Type: ${data.image_type}\n` +
      `Status: ${data.status}\n` +
      `Disease Diagnosed: ${data.disease}`
    );
  };

  return (
    <div className="container">
      <header className="header">
        <h1>MediScan</h1>
        <p>Medical Image Disease Classification</p>
      </header>

      <div className="upload-section">
        <label className="file-label">Upload Medical Image</label>
        <input type="file" accept="image/*" onChange={handleFileChange} />
      </div>

      {preview && (
        <div className="report-card">
          <h2>Scan Report</h2>

          <div className="report-grid">
            <div className="card image-card">
              <h3>Uploaded Image</h3>
              <img src={preview} alt="Scan preview" />
            </div>

            <div className="card">
              <h3>Details</h3>
              <p><b>File:</b> {image?.name}</p>
              <p><b>Type:</b> {image?.type}</p>
              <p><b>Resolution:</b> Auto-detected</p>
            </div>

            <div className="card">
              <h3>Prediction</h3>
              <pre className="prediction-box">
                {prediction || "No prediction yet"}
              </pre>
            </div>
          </div>
        </div>
      )}

      <button
        className={`predict-btn ${completed ? "done" : ""}`}
        onClick={handlePredict}
        disabled={!image || loading}
      >
        {loading
          ? "Analyzing Scan..."
          : completed
          ? "Completed ✓"
          : "Predict"}
      </button>

      {loading && <div className="loader">Processing image…</div>}
    </div>
  );
}

export default App;

import { useRef, useState } from "react";
import "./index.css";

type VisualFeature = {
  score: number;
  level: "Low" | "Moderate" | "High";
  bar: string;
};

type FeatureSet = {
  Texture: VisualFeature;
  "Surface detail": VisualFeature;
  "Color variation": VisualFeature;
  "Edge density": VisualFeature;
  "Pattern scale": VisualFeature;
  Brightness: VisualFeature;
};

type Prediction = {
  className: "CINNAMON" | "NEEM" | "NONE";
  confidence: number;

  probabilities: {
    CINNAMON: number;
    NEEM: number;
    NONE: number;
  };

  gradcam: string;
  features: FeatureSet | null;
};

function App() {
  const [image, setImage] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [fileName, setFileName] = useState("");
  const [dragging, setDragging] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [error, setError] = useState("");

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = (selectedFile: File) => {
    if (!selectedFile.type.startsWith("image/")) {
      alert("Please select an image file.");
      return;
    }

    setFile(selectedFile);
    setFileName(selectedFile.name);
    setPrediction(null);
    setError("");

    const reader = new FileReader();

    reader.onload = () => {
      setImage(reader.result as string);
    };

    reader.readAsDataURL(selectedFile);
  };

  const handleFileInput = (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const selectedFile = event.target.files?.[0];

    if (selectedFile) {
      handleFile(selectedFile);
    }
  };

  const handleDrop = (
    event: React.DragEvent<HTMLDivElement>
  ) => {
    event.preventDefault();
    setDragging(false);

    const droppedFile = event.dataTransfer.files?.[0];

    if (droppedFile) {
      handleFile(droppedFile);
    }
  };

  const analyzeImage = async () => {
    if (!file) return;

    setAnalyzing(true);
    setPrediction(null);
    setError("");

    try {
      const formData = new FormData();

      formData.append("file", file);

      const response = await fetch(
        "http://127.0.0.1:8000/predict",
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      console.log("TreeBark AI Response:", data);

      if (!response.ok) {
        throw new Error(
          data.error ||
          `Server error: ${response.status}`
        );
      }

      if (!data.success) {
        throw new Error(
          data.error ||
          "Prediction failed."
        );
      }

      let backendFeatures: FeatureSet | null = null;

      if (data.features) {
        backendFeatures = {
          Texture: data.features.Texture,
          "Surface detail":
            data.features["Surface detail"],
          "Color variation":
            data.features["Color variation"],
          "Edge density":
            data.features["Edge density"],
          "Pattern scale":
            data.features["Pattern scale"],
          Brightness:
            data.features.Brightness,
        };
      }

      setPrediction({
        className: data.prediction,
        confidence: Number(data.confidence),

        probabilities: {
          CINNAMON: Number(
            data.probabilities.CINNAMON
          ),

          NEEM: Number(
            data.probabilities.NEEM
          ),

          NONE: Number(
            data.probabilities.NONE
          ),
        },

        gradcam: data.gradcam || "",

        features: backendFeatures,
      });

    } catch (err) {
      console.error("Prediction error:", err);

      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError(
          "Could not connect to the TreeBark AI backend."
        );
      }

    } finally {
      setAnalyzing(false);
    }
  };

  const removeImage = () => {
    setImage(null);
    setFile(null);
    setFileName("");
    setPrediction(null);
    setError("");

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  return (
    <div className="app">

      {/* NAVBAR */}

      <nav className="navbar">

        <div className="brand">

          <div className="brand-icon">
            🌿
          </div>

          <div>
            <h1>TreeBark AI</h1>

            <span>
              Vision-Based Bark Classification
            </span>
          </div>

        </div>

        <div className="nav-status">
          <span className="status-dot"></span>
          Hybrid CNN
        </div>

      </nav>


      {/* MAIN */}

      <main className="main">


        {/* HERO */}

        <section className="hero">

          <div className="hero-badge">

            <span>AI-POWERED</span>

            <span>•</span>

            <span>
              91.95% TEST ACCURACY
            </span>

          </div>

          <h2>
            Identify Tree Bark
            <br />

            <span>
              with Artificial Intelligence
            </span>
          </h2>

          <p>
            Upload a dried bark image and let our custom
            Hybrid CNN classify it as Cinnamon, Neem, or None.
          </p>

        </section>


        {/* WORKSPACE */}

        <section className="workspace">


          {/* UPLOAD CARD */}

          <div className="card upload-card">

            <div className="card-header">

              <div>

                <span className="section-number">
                  01
                </span>

                <h3>
                  Upload Bark Image
                </h3>

              </div>

              {image && (

                <button
                  className="clear-button"
                  onClick={removeImage}
                >
                  Clear
                </button>

              )}

            </div>


            {!image ? (

              <div
                className={`drop-zone ${
                  dragging ? "dragging" : ""
                }`}

                onDragOver={(event) => {
                  event.preventDefault();
                  setDragging(true);
                }}

                onDragLeave={() => {
                  setDragging(false);
                }}

                onDrop={handleDrop}

                onClick={() =>
                  fileInputRef.current?.click()
                }
              >

                <div className="upload-icon">
                  ↑
                </div>

                <h4>
                  Drop your bark image here
                </h4>

                <p>
                  or click to browse from your device
                </p>

                <span className="file-types">
                  JPG, JPEG or PNG
                </span>

                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/png,image/jpeg,image/jpg"
                  onChange={handleFileInput}
                  hidden
                />

              </div>

            ) : (

              <div className="image-preview-container">

                <img
                  src={image}
                  alt="Uploaded bark"
                  className="image-preview"
                />

                <div className="file-name">
                  {fileName}
                </div>

              </div>

            )}


            {/* ANALYZE BUTTON */}

            <button
              className={`analyze-button ${
                !image || analyzing
                  ? "disabled"
                  : ""
              }`}

              disabled={!image || analyzing}

              onClick={analyzeImage}
            >

              {analyzing ? (

                <>
                  <span className="spinner"></span>
                  Analyzing...
                </>

              ) : (

                <>
                  Analyze Bark
                  <span>→</span>
                </>

              )}

            </button>


            {/* ERROR */}

            {error && (

              <div className="error-message">
                {error}
              </div>

            )}

          </div>


          {/* RESULT CARD */}

          <div className="card result-card">

            <div className="card-header">

              <div>

                <span className="section-number">
                  02
                </span>

                <h3>
                  Classification Result
                </h3>

              </div>

              <span className="ai-label">
                AI PREDICTION
              </span>

            </div>


            {!prediction ? (

              <div className="empty-result">

                <div className="empty-icon">
                  ✦
                </div>

                <h4>
                  Waiting for an image
                </h4>

                <p>
                  Upload a bark image and click
                  <br />

                  <strong>
                    Analyze Bark
                  </strong>{" "}
                  to begin.
                </p>

              </div>

            ) : (

              <div className="result-content">


                {/* PREDICTION */}

                <div className="prediction-main">

                  <span className="prediction-label">
                    PREDICTED CLASS
                  </span>

                  <h4>
                    {prediction.className}
                  </h4>

                  <div className="confidence">

                    <span>
                      Confidence
                    </span>

                    <strong>
                      {prediction.confidence.toFixed(2)}%
                    </strong>

                  </div>

                  <div className="confidence-track">

                    <div
                      className="confidence-fill"

                      style={{
                        width:
                          `${prediction.confidence}%`,
                      }}
                    />

                  </div>

                </div>


                {/* PROBABILITIES */}

                <div className="probability-section">

                  <span className="prediction-label">
                    CLASS PROBABILITIES
                  </span>

                  <ProbabilityBar
                    label="Cinnamon"
                    value={
                      prediction.probabilities.CINNAMON
                    }
                  />

                  <ProbabilityBar
                    label="Neem"
                    value={
                      prediction.probabilities.NEEM
                    }
                  />

                  <ProbabilityBar
                    label="None"
                    value={
                      prediction.probabilities.NONE
                    }
                  />

                </div>

              </div>

            )}

          </div>

        </section>


        {/* EXPLAINABLE AI */}

        <section className="card explainability">

          <div className="card-header">

            <div>

              <span className="section-number">
                03
              </span>

              <h3>
                Explainable AI
              </h3>

            </div>

            <span className="xai-label">
              GRAD-CAM
            </span>

          </div>


          <div className="xai-content">


            {/* GRAD-CAM IMAGE */}

            <div className="xai-placeholder">

              {prediction?.gradcam ? (

                <img
                  src={prediction.gradcam}
                  alt="Grad-CAM visualization"
                  className="gradcam-image"
                />

              ) : (

                <>

                  <div className="xai-symbol">
                    ◎
                  </div>

                  <h4>
                    Model Attention Visualization
                  </h4>

                  <p>
                    Grad-CAM will highlight the image
                    regions that contributed to the
                    model's prediction.
                  </p>

                  <span>
                    Available after analysis
                  </span>

                </>

              )}

            </div>


            {/* XAI INFORMATION */}

            <div className="xai-info">

              <h4>
                Why this prediction?
              </h4>

              <p>
                Grad-CAM identifies the image regions
                that contributed to the model's prediction.
                The characteristics below describe measurable
                properties of those important regions.
              </p>


              {prediction?.features ? (

                <div className="visual-features">

                  <span className="feature-heading">
                    VISUAL CHARACTERISTICS
                  </span>

                  <FeatureAnalysis
                    name="Texture"
                    feature={
                      prediction.features.Texture
                    }
                  />

                  <FeatureAnalysis
                    name="Surface detail"
                    feature={
                      prediction.features[
                        "Surface detail"
                      ]
                    }
                  />

                  <FeatureAnalysis
                    name="Color variation"
                    feature={
                      prediction.features[
                        "Color variation"
                      ]
                    }
                  />

                  <FeatureAnalysis
                    name="Edge density"
                    feature={
                      prediction.features[
                        "Edge density"
                      ]
                    }
                  />

                  <FeatureAnalysis
                    name="Pattern scale"
                    feature={
                      prediction.features[
                        "Pattern scale"
                      ]
                    }
                  />

                  <FeatureAnalysis
                    name="Brightness"
                    feature={
                      prediction.features.Brightness
                    }
                  />

                </div>

              ) : (

                <div className="feature-list">

                  <div>
                    <span>3×3</span>
                    Fine-grained patterns
                  </div>

                  <div>
                    <span>5×5</span>
                    Broader patterns
                  </div>

                  <div>
                    <span>XAI</span>
                    Grad-CAM explanation
                  </div>

                </div>

              )}

            </div>

          </div>

        </section>


        {/* MODEL INFO */}

        <section className="model-info">

          <div>
            <span>MODEL</span>

            <strong>
              Custom Hybrid CNN
            </strong>
          </div>

          <div>
            <span>CLASSES</span>

            <strong>
              3
            </strong>
          </div>

          <div>
            <span>INPUT</span>

            <strong>
              224 × 224
            </strong>
          </div>

          <div>
            <span>TEST ACCURACY</span>

            <strong>
              91.95%
            </strong>
          </div>

        </section>

      </main>


      {/* FOOTER */}

      <footer>

        <span>
          TreeBark AI
        </span>

        <span>
          Custom CNN • Explainable AI • Computer Vision
        </span>

      </footer>

    </div>
  );
}


/* ============================================================
   PROBABILITY BAR
   ============================================================ */

function ProbabilityBar({
  label,
  value,
}: {
  label: string;
  value: number;
}) {

  return (

    <div className="probability">

      <div className="probability-top">

        <span>
          {label}
        </span>

        <strong>
          {value.toFixed(2)}%
        </strong>

      </div>

      <div className="probability-track">

        <div
          className="probability-fill"

          style={{
            width:
              `${value}%`,
          }}

        />

      </div>

    </div>

  );
}


/* ============================================================
   FEATURE ANALYSIS
   ============================================================ */

function FeatureAnalysis({
  name,
  feature,
}: {
  name: string;
  feature: VisualFeature;
}) {

  return (

    <div className="visual-feature">

      <div className="visual-feature-top">

        <span className="visual-feature-name">
          {name}
        </span>

        <span className="visual-feature-level">
          {feature.level}
        </span>

      </div>

      <div className="visual-feature-bar">

        <span>
          {feature.bar}
        </span>

      </div>

    </div>

  );
}


export default App;
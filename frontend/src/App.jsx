import { useState, useRef, useCallback } from "react";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

const TABS = ["Image", "Video"];

export default function App() {
  const [tab, setTab] = useState("Image");
  const [dragging, setDragging] = useState(false);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState("");
  const [result, setResult] = useState(null); // { type: "image"|"video", data }
  const [error, setError] = useState(null);

  const canvasRef = useRef(null);
  const fileRef = useRef(null);

  // ── File selection ──────────────────────────────────────────────────────────
  const accept = tab === "Image" ? "image/*" : "video/*";

  const handleFile = useCallback((f) => {
    if (!f) return;
    setFile(f);
    setResult(null);
    setError(null);
    const url = URL.createObjectURL(f);
    setPreview(url);
  }, []);

  const onDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  };

  // ── Submit ──────────────────────────────────────────────────────────────────
  const handleSubmit = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);

    const fd = new FormData();
    fd.append("file", file);

    try {
      if (tab === "Image") {
        setProgress("Running detection…");
        const res = await fetch(`${API_BASE}/detect/image`, {
          method: "POST",
          body: fd,
        });
        if (!res.ok) throw new Error(await res.text());
        const json = await res.json();
        setResult({
          type: "image",
          detections: json.detections,
          width: json.width,
          height: json.height,
        });
        // Draw boxes after state update (use timeout so canvas is rendered)
        setTimeout(() => drawBoxes(json.detections), 50);
      } else {
        setProgress("Uploading video…");
        const res = await fetch(`${API_BASE}/detect/video`, {
          method: "POST",
          body: fd,
        });
        if (!res.ok) throw new Error(await res.text());
        setProgress("Receiving annotated video…");
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        setResult({ type: "video", url });
      }
    } catch (e) {
      setError(e.message || "Something went wrong");
    } finally {
      setLoading(false);
      setProgress("");
    }
  };

  // ── Canvas drawing ──────────────────────────────────────────────────────────
  const drawBoxes = (detections) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const img = new Image();
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0);
      detections.forEach(({ x, y, w, h, label, conf }) => {
        ctx.strokeStyle = "#e040fb";
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, w, h);
        const text = `${label} ${conf.toFixed(2)}`;
        ctx.font = "bold 14px monospace";
        const tw = ctx.measureText(text).width;
        ctx.fillStyle = "rgba(224,64,251,0.85)";
        ctx.fillRect(x, Math.max(0, y - 22), tw + 8, 22);
        ctx.fillStyle = "#fff";
        ctx.fillText(text, x + 4, Math.max(0, y - 6));
      });
    };
    img.src = preview;
  };

  // ── Reset ───────────────────────────────────────────────────────────────────
  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    setProgress("");
  };

  // ── UI ──────────────────────────────────────────────────────────────────────
  return (
    <div
      style={{
        minHeight: "100vh",
        background: "var(--color-background-tertiary)",
        fontFamily: "var(--font-sans)",
        color: "var(--color-text-primary)",
      }}
    >
      {/* Header */}
      <header
        style={{
          background: "var(--color-background-primary)",
          borderBottom: "1px solid var(--color-border-tertiary)",
          padding: "18px 32px",
          display: "flex",
          alignItems: "center",
          gap: 12,
        }}
      >
        <div
          style={{
            width: 36,
            height: 36,
            borderRadius: 10,
            background: "#e040fb",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 18,
          }}
        >
          🕳️
        </div>
        <div>
          <div style={{ fontWeight: 500, fontSize: 17 }}>Pothole Detection</div>
          <div style={{ fontSize: 12, color: "var(--color-text-secondary)" }}>
            YOLOv8 Nano · Upload image or video
          </div>
        </div>
      </header>

      <main style={{ maxWidth: 860, margin: "0 auto", padding: "32px 20px" }}>
        {/* Tabs */}
        <div
          style={{
            display: "flex",
            gap: 6,
            marginBottom: 24,
            background: "var(--color-background-primary)",
            borderRadius: "var(--border-radius-lg)",
            padding: 4,
            width: "fit-content",
            border: "1px solid var(--color-border-tertiary)",
          }}
        >
          {TABS.map((t) => (
            <button
              key={t}
              onClick={() => {
                setTab(t);
                reset();
              }}
              style={{
                padding: "7px 22px",
                borderRadius: 10,
                border: "none",
                cursor: "pointer",
                fontWeight: 500,
                fontSize: 14,
                background: tab === t ? "#e040fb" : "transparent",
                color: tab === t ? "#fff" : "var(--color-text-secondary)",
                transition: "all .15s",
              }}
            >
              {t}
            </button>
          ))}
        </div>

        {/* Drop zone */}
        {!result && (
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            onClick={() => fileRef.current?.click()}
            style={{
              border: `2px dashed ${dragging ? "#e040fb" : "var(--color-border-secondary)"}`,
              borderRadius: "var(--border-radius-lg)",
              padding: 40,
              textAlign: "center",
              background: dragging
                ? "rgba(224,64,251,0.05)"
                : "var(--color-background-primary)",
              cursor: "pointer",
              transition: "all .15s",
              marginBottom: 20,
            }}
          >
            <input
              ref={fileRef}
              type="file"
              accept={accept}
              style={{ display: "none" }}
              onChange={(e) => handleFile(e.target.files[0])}
            />
            <div style={{ fontSize: 40, marginBottom: 12 }}>
              {tab === "Image" ? "🖼️" : "🎬"}
            </div>
            <div style={{ fontWeight: 500, marginBottom: 6 }}>
              Drop your {tab.toLowerCase()} here
            </div>
            <div style={{ fontSize: 13, color: "var(--color-text-secondary)" }}>
              {tab === "Image" ? "PNG, JPG, WEBP" : "MP4, MOV, AVI"} · click to
              browse
            </div>
          </div>
        )}

        {/* Preview */}
        {preview && !result && (
          <div
            style={{
              marginBottom: 20,
              borderRadius: "var(--border-radius-lg)",
              overflow: "hidden",
              border: "1px solid var(--color-border-tertiary)",
              background: "var(--color-background-primary)",
            }}
          >
            {tab === "Image" ? (
              <img
                src={preview}
                alt="preview"
                style={{
                  width: "100%",
                  maxHeight: 400,
                  objectFit: "contain",
                  display: "block",
                }}
              />
            ) : (
              <video
                src={preview}
                controls
                style={{ width: "100%", maxHeight: 400, display: "block" }}
              />
            )}
            <div
              style={{
                padding: "12px 16px",
                display: "flex",
                gap: 10,
                alignItems: "center",
              }}
            >
              <span
                style={{
                  fontSize: 13,
                  color: "var(--color-text-secondary)",
                  flex: 1,
                }}
              >
                {file?.name}
              </span>
              <button
                onClick={reset}
                style={{
                  padding: "6px 14px",
                  borderRadius: 8,
                  border: "1px solid var(--color-border-secondary)",
                  background: "transparent",
                  cursor: "pointer",
                  fontSize: 13,
                  color: "var(--color-text-secondary)",
                }}
              >
                Remove
              </button>
              <button
                onClick={handleSubmit}
                disabled={loading}
                style={{
                  padding: "6px 20px",
                  borderRadius: 8,
                  border: "none",
                  background: loading
                    ? "var(--color-border-secondary)"
                    : "#e040fb",
                  color: "#fff",
                  fontWeight: 500,
                  fontSize: 13,
                  cursor: loading ? "default" : "pointer",
                }}
              >
                {loading ? progress || "Processing…" : "Detect Potholes"}
              </button>
            </div>
          </div>
        )}

        {/* Loading bar */}
        {loading && (
          <div style={{ marginBottom: 20 }}>
            <div
              style={{
                height: 4,
                background: "var(--color-border-tertiary)",
                borderRadius: 2,
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  height: "100%",
                  background: "#e040fb",
                  animation: "pulse-bar 1.4s ease-in-out infinite",
                  borderRadius: 2,
                }}
              />
            </div>
            <div
              style={{
                fontSize: 13,
                color: "var(--color-text-secondary)",
                marginTop: 8,
                textAlign: "center",
              }}
            >
              {progress}
            </div>
          </div>
        )}

        {/* Error */}
        {error && (
          <div
            style={{
              padding: 16,
              borderRadius: "var(--border-radius-md)",
              background: "var(--color-background-danger)",
              border: "1px solid var(--color-border-danger)",
              color: "var(--color-text-danger)",
              fontSize: 14,
              marginBottom: 20,
            }}
          >
            ⚠️ {error}
          </div>
        )}

        {/* Image result */}
        {result?.type === "image" && (
          <div
            style={{
              borderRadius: "var(--border-radius-lg)",
              overflow: "hidden",
              border: "1px solid var(--color-border-tertiary)",
              background: "var(--color-background-primary)",
            }}
          >
            <div
              style={{
                padding: "12px 16px",
                borderBottom: "1px solid var(--color-border-tertiary)",
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
              }}
            >
              <span style={{ fontWeight: 500 }}>
                Detection result · {result.detections.length} pothole
                {result.detections.length !== 1 ? "s" : ""} found
              </span>
              <button
                onClick={reset}
                style={{
                  padding: "5px 14px",
                  borderRadius: 8,
                  border: "1px solid var(--color-border-secondary)",
                  background: "transparent",
                  cursor: "pointer",
                  fontSize: 13,
                  color: "var(--color-text-secondary)",
                }}
              >
                Try another
              </button>
            </div>
            <canvas
              ref={canvasRef}
              style={{ width: "100%", display: "block" }}
            />
            {result.detections.length > 0 && (
              <div
                style={{
                  padding: 16,
                  display: "flex",
                  flexWrap: "wrap",
                  gap: 8,
                }}
              >
                {result.detections.map((d, i) => (
                  <span
                    key={i}
                    style={{
                      padding: "4px 12px",
                      borderRadius: 20,
                      background: "rgba(224,64,251,0.1)",
                      border: "1px solid rgba(224,64,251,0.3)",
                      fontSize: 13,
                      color: "#e040fb",
                    }}
                  >
                    {d.label} · {(d.conf * 100).toFixed(0)}%
                  </span>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Video result */}
        {result?.type === "video" && (
          <div
            style={{
              borderRadius: "var(--border-radius-lg)",
              overflow: "hidden",
              border: "1px solid var(--color-border-tertiary)",
              background: "var(--color-background-primary)",
            }}
          >
            <div
              style={{
                padding: "12px 16px",
                borderBottom: "1px solid var(--color-border-tertiary)",
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
              }}
            >
              <span style={{ fontWeight: 500 }}>Annotated video ready</span>
              <div style={{ display: "flex", gap: 8 }}>
                <a
                  href={result.url}
                  download="pothole_annotated.mp4"
                  style={{
                    padding: "5px 14px",
                    borderRadius: 8,
                    background: "#e040fb",
                    color: "#fff",
                    fontWeight: 500,
                    fontSize: 13,
                    textDecoration: "none",
                  }}
                >
                  Download
                </a>
                <button
                  onClick={reset}
                  style={{
                    padding: "5px 14px",
                    borderRadius: 8,
                    border: "1px solid var(--color-border-secondary)",
                    background: "transparent",
                    cursor: "pointer",
                    fontSize: 13,
                    color: "var(--color-text-secondary)",
                  }}
                >
                  Try another
                </button>
              </div>
            </div>
            <video
              src={result.url}
              controls
              autoPlay
              style={{ width: "100%", maxHeight: 480, display: "block" }}
            />
          </div>
        )}
      </main>

      <style>{`
        @keyframes pulse-bar {
          0%   { width: 0%; margin-left: 0; }
          50%  { width: 60%; margin-left: 20%; }
          100% { width: 0%; margin-left: 100%; }
        }
      `}</style>
    </div>
  );
}

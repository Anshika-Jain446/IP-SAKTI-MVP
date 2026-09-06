import { useState } from "react";

const API = "http://localhost:8000";

function App() {
  const [form, setForm] = useState({
    product_name: "Ashwa Joint Relief",
    ingredients: "Ashwagandha, Turmeric",
    purpose: "Joint pain",
    product_type: "Ayurvedic formulation",
    jurisdiction: "India",
    based_on_traditional_knowledge: "Not sure",
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  function update(key, value) {
    setForm((old) => ({ ...old, [key]: value }));
  }

  async function analyze(e) {
    e.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API}/api/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...form,
          ingredients: form.ingredients
            .split(",")
            .map((x) => x.trim())
            .filter(Boolean),
        }),
      });

      if (!response.ok) throw new Error("Backend returned an error.");
      setResult(await response.json());
    } catch (err) {
      setError(
        "Could not connect to the backend. Make sure FastAPI is running on port 8000."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <header className="hero">
        <div>
          <div className="eyebrow">CODEHUNTERS HACKATHON MVP</div>
          <h1>IP-SAKTI</h1>
          <p>
            Evidence-first assistant for preliminary IP, Traditional Knowledge
            and ABS assessment.
          </p>
        </div>
        <div className="architecture-pill">
          Intake → Classify → Route → Retrieve → Verify → Act
        </div>
      </header>

      <main className="layout">
        <section className="card">
          <h2>1. Product Intake</h2>
          <p className="muted">
            Start with what the user actually knows. The system should not
            assume missing facts.
          </p>

          <form onSubmit={analyze}>
            <label>Product name</label>
            <input
              value={form.product_name}
              onChange={(e) => update("product_name", e.target.value)}
              required
            />

            <label>Ingredients / components</label>
            <input
              value={form.ingredients}
              onChange={(e) => update("ingredients", e.target.value)}
              placeholder="Comma separated"
            />

            <label>Intended use</label>
            <textarea
              value={form.purpose}
              onChange={(e) => update("purpose", e.target.value)}
            />

            <label>Product type</label>
            <select
              value={form.product_type}
              onChange={(e) => update("product_type", e.target.value)}
            >
              <option>Ayurvedic formulation</option>
              <option>Herbal product</option>
              <option>Food</option>
              <option>Cosmetic</option>
              <option>Other</option>
            </select>

            <label>Jurisdiction</label>
            <select
              value={form.jurisdiction}
              onChange={(e) => update("jurisdiction", e.target.value)}
            >
              <option>India</option>
            </select>

            <label>Based on traditional knowledge?</label>
            <select
              value={form.based_on_traditional_knowledge}
              onChange={(e) =>
                update("based_on_traditional_knowledge", e.target.value)
              }
            >
              <option>Not sure</option>
              <option>Yes</option>
              <option>No</option>
            </select>

            <button disabled={loading}>
              {loading ? "Analyzing..." : "Analyze Product →"}
            </button>
          </form>

          {error && <div className="error">{error}</div>}
        </section>

        <section className="results">
          {!result && (
            <div className="empty card">
              <div className="big-icon">🔎</div>
              <h2>Your analysis will appear here</h2>
              <p>
                The prototype will classify the product, route it to IP/TK/ABS,
                retrieve evidence, validate the response and create an action
                plan.
              </p>
            </div>
          )}

          {result && (
            <>
              <div className="card">
                <div className="result-header">
                  <div>
                    <div className="eyebrow">PRELIMINARY CLASSIFICATION</div>
                    <h2>{result.classification.label}</h2>
                  </div>
                  <div className="status">{result.validation.status}</div>
                </div>
                <ul>
                  {result.classification.reasons.map((r) => (
                    <li key={r}>{r}</li>
                  ))}
                </ul>
              </div>

              <div className="grid">
                <div className="card">
                  <h3>🚦 Domain Router</h3>
                  <div className="chips">
                    {result.domains.map((d) => (
                      <span className="chip" key={d}>{d}</span>
                    ))}
                  </div>
                  <p className="muted">
                    These are the knowledge areas selected for this case.
                  </p>
                </div>

                <div className="card">
                  <h3>📊 Confidence</h3>
                  <div className="confidence">{result.confidence.score}%</div>
                  <strong>{result.confidence.label}</strong>
                  <p className="muted">{result.confidence.meaning}</p>
                </div>
              </div>

              <div className="card">
                <h3>🔎 Retrieved Evidence</h3>
                {result.evidence.length === 0 ? (
                  <p>No evidence was retrieved.</p>
                ) : (
                  result.evidence.map((e) => (
                    <article className="evidence" key={e.id}>
                      <div className="evidence-top">
                        <strong>{e.title}</strong>
                        <span>{Math.round(e.score * 100)}%</span>
                      </div>
                      <p>{e.text}</p>
                      <small>
                        {e.source} · {e.domain} ·{" "}
                        <a href={e.source_url} target="_blank" rel="noreferrer">
                          Official source
                        </a>
                      </small>
                    </article>
                  ))
                )}
              </div>

              <div className="card">
                <h3>🤖 AI / Reasoning Layer</h3>
                <p>{result.reasoning.answer}</p>
                <div className="validation">
                  <strong>Claim validation:</strong>{" "}
                  {result.validation.message}
                </div>
              </div>

              <div className="card">
                <h3>🚀 Action Plan</h3>
                <ol>
                  {result.action_plan.map((step) => (
                    <li key={step}>{step}</li>
                  ))}
                </ol>
              </div>

              <div className="disclaimer">
                Prototype only — not legal advice, not a patentability
                determination, and not a substitute for qualified professional
                review.
              </div>
            </>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;

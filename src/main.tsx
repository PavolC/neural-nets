import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { startAnalytics } from "./analytics";
import "./styles.css";

// Off unless VITE_GOATCOUNTER is set at build time. See src/analytics.ts.
startAnalytics();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);

import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.tsx";
import "./index.css";

const urlSearchParams = new URLSearchParams(window.location.search);
const hl = urlSearchParams.get("hl");
ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App language={hl} />
  </React.StrictMode>
);

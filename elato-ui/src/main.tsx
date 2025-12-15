import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./App.css";
import { ActiveUserProvider } from "./state/ActiveUserContext";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <ActiveUserProvider>
      <App />
    </ActiveUserProvider>
  </React.StrictMode>,
);

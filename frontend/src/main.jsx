import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";
import { assetUrl } from "./assetUrl.js";
import "./index.css";

const fonts = document.createElement("style");
fonts.textContent = `
@font-face {
  font-family: "NCL Gesrob";
  src: url("${assetUrl("fonts/ncl-gesrob.demo.otf")}") format("opentype");
  font-weight: 400;
  font-style: normal;
  font-display: swap;
}
@font-face {
  font-family: "0xProto Nerd Font";
  src: url("${assetUrl("fonts/0xProto-Regular.woff2")}") format("woff2");
  font-weight: 400;
  font-style: normal;
  font-display: swap;
}
`;
document.head.appendChild(fonts);

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <App />
  </StrictMode>
);

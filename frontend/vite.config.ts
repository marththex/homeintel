import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Proxying API routes keeps the app same-origin in dev too (see src/api.ts).
const API_ROUTES = ["/chat", "/health", "/stats", "/file", "/thumb", "/visual-search", "/transcribe"];

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true,
    proxy: Object.fromEntries(API_ROUTES.map((r) => [r, "http://localhost:8000"])),
  },
});

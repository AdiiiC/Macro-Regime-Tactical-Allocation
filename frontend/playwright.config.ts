import { defineConfig, devices } from "@playwright/test";

/**
 * Smoke-test config. Expects the FastAPI backend to be reachable on the
 * Vite proxy target (default http://localhost:8001). The dev server is
 * started automatically and reused if already running.
 */
export default defineConfig({
  testDir: "./e2e",
  timeout: 30_000,
  expect: { timeout: 10_000 },
  // The smoke suite runs against a single-worker dev backend whose heavier
  // analytical endpoints are CPU-bound; run specs serially so parallel browser
  // contexts don't contend for that one worker and time out.
  fullyParallel: false,
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? "list" : "line",
  use: {
    baseURL: "http://localhost:5180",
    trace: "on-first-retry",
  },
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
  ],
  webServer: {
    command: "npm run dev",
    url: "http://localhost:5180",
    reuseExistingServer: true,
    timeout: 60_000,
  },
});

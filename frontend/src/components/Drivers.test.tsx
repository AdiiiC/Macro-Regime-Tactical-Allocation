import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import Drivers from "./Drivers";
import type { DriversResult } from "../types";

const data: DriversResult = {
  market: "us",
  regime: "Recession",
  as_of: "2026-06-30",
  drivers: [
    { feature: "Core_CPI_YoY", z_score: -1.36, direction: "depressed", regime_avg: 2.1 },
    { feature: "VIX_Chg3", z_score: 1.26, direction: "elevated", regime_avg: 0.4 },
  ],
};

describe("Drivers panel", () => {
  it("renders the regime and humanized feature labels", () => {
    render(<Drivers data={data} />);
    expect(screen.getByText("Recession")).toBeInTheDocument();
    expect(screen.getByText("Core CPI · y/y")).toBeInTheDocument();
    expect(screen.getByText("VIX · 3m chg.")).toBeInTheDocument();
  });

  it("renders signed z-scores", () => {
    render(<Drivers data={data} />);
    expect(screen.getByText("-1.36σ")).toBeInTheDocument();
    expect(screen.getByText("+1.26σ")).toBeInTheDocument();
  });

  it("shows the as-of date", () => {
    render(<Drivers data={data} />);
    expect(screen.getByText("as of 2026-06-30")).toBeInTheDocument();
  });
});

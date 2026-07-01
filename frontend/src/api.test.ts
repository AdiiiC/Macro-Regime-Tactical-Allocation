import { describe, expect, it } from "vitest";
import {
  assetLabel,
  num,
  orderAssets,
  pct,
  signedNum,
  signedPct,
} from "./api";

describe("number formatters", () => {
  it("formats percentages", () => {
    expect(pct(0.1234)).toBe("12.3%");
    expect(pct(0.1234, 2)).toBe("12.34%");
  });

  it("signs percentages", () => {
    expect(signedPct(0.05)).toBe("+5.0%");
    expect(signedPct(-0.05)).toBe("-5.0%");
    expect(signedPct(0)).toBe("+0.0%");
  });

  it("formats plain numbers", () => {
    expect(num(0.8532)).toBe("0.85");
    expect(signedNum(1.2, 1)).toBe("+1.2");
    expect(signedNum(-1.2, 1)).toBe("-1.2");
  });
});

describe("asset labels", () => {
  it("maps known US and India tickers", () => {
    expect(assetLabel("US_Equity")).toBe("US Equity");
    expect(assetLabel("Nifty_50")).toBe("Nifty 50");
    expect(assetLabel("G_Sec_Long")).toBe("G-Sec (Long)");
  });

  it("humanizes unknown keys", () => {
    expect(assetLabel("Some_New_Asset")).toBe("Some New Asset");
  });
});

describe("orderAssets", () => {
  it("orders known US assets first, appends unknowns", () => {
    const out = orderAssets(["Cash", "Nifty_50", "US_Equity"]);
    expect(out).toEqual(["US_Equity", "Cash", "Nifty_50"]);
  });

  it("returns India keys unchanged when none are in the US order", () => {
    const india = ["Nifty_50", "Bank_Nifty", "Gold_INR"];
    expect(orderAssets(india)).toEqual(india);
  });
});

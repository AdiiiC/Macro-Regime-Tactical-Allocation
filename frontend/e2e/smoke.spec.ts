import { expect, test } from "@playwright/test";

test.describe("Meridian dashboard smoke", () => {
  test("US market loads core panels", async ({ page }) => {
    await page.goto("/?market=us");

    // Brand + core regime panel resolve out of the loading state.
    await expect(page.getByText("MERIDIAN", { exact: true })).toBeVisible();
    await expect(
      page.getByText("Current Market Regime", { exact: false })
    ).toBeVisible();

    // A regime label chip renders (one of the four states).
    await expect(
      page.getByText(/Expansion|Slowdown|Recession|Recovery/).first()
    ).toBeVisible();

    // PDF memo export link points at the US report.
    const memo = page.getByRole("link", { name: /Memo/i });
    await expect(memo).toHaveAttribute("href", /\/api\/report\/pdf\?market=us/);
  });

  test("switching to India degrades gracefully", async ({ page }) => {
    await page.goto("/?market=us");
    await expect(page.getByText("MERIDIAN", { exact: true })).toBeVisible();

    await page.getByRole("button", { name: "INDIA", exact: true }).click();

    // India-specific asset appears in the allocation panel.
    await expect(page.getByText("Nifty 50").first()).toBeVisible();

    // Risk/stress panels show the neutral unavailable note rather than crashing.
    await expect(
      page.getByText(/Asset return data is not available for India/).first()
    ).toBeVisible();
  });

  test("theme toggle switches to light mode", async ({ page }) => {
    await page.goto("/?market=us");
    await expect(page.getByText("MERIDIAN", { exact: true })).toBeVisible();

    const html = page.locator("html");
    await expect(html).toHaveAttribute("data-theme", "dark");

    await page.getByRole("button", { name: "☾" }).click();
    await expect(html).toHaveAttribute("data-theme", "light");
  });
});

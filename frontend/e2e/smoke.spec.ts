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

  test("switching to India loads a live market", async ({ page }) => {
    await page.goto("/?market=us");
    await expect(page.getByText("MERIDIAN", { exact: true })).toBeVisible();

    await page.getByRole("button", { name: "INDIA", exact: true }).click();

    // India-specific asset appears in the allocation panel.
    await expect(page.getByText("Nifty 50").first()).toBeVisible();

    // India now carries live returns, so the backtest panel renders its
    // equity chart rather than an unavailable note.
    await expect(
      page.getByText("Strategy backtest", { exact: false }).first()
    ).toBeVisible();

    // Memo export retargets at the India report.
    await expect(
      page.getByRole("link", { name: /Memo/i })
    ).toHaveAttribute("href", /\/api\/report\/pdf\?market=india/);
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

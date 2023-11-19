from playwright.sync_api import Playwright, sync_playwright, expect
import time


def run(playwright: Playwright) -> None:
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    # insert
    page.goto("http://127.0.0.1:8000/admin/dashboard")
    page.get_by_role("link", name=" Supplier").click()
    page.get_by_role("button", name=" Create New Supplier").click()
    page.get_by_label("Supplier Name").click()
    page.get_by_label("Supplier Name").fill("test-supplier")
    page.get_by_role("button", name=" Add New Supplier").click()
    page.get_by_text("test-supplier").click()
    time.sleep(5)

    # update
    page.get_by_role("row", name="test-supplier  Update  Delete").get_by_role("button").first.click()
    page.get_by_label("Supplier Name").click()
    page.get_by_label("Supplier Name").fill("test-supplier-updated")
    page.get_by_role("button", name=" Update Supplier").click()
    page.get_by_text("test-supplier-updated").click()
    time.sleep(5)

    # delete
    page.get_by_role("row", name="test-supplier-updated  Update  Delete").get_by_role("button").nth(1).click()
    page.get_by_text("supplier - 10").click()
    time.sleep(5)
    # ---------------------
    context.close()
    browser.close()


with sync_playwright() as playwright:
    run(playwright)

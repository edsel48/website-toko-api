from playwright.sync_api import Playwright, sync_playwright, expect
import time


def run(playwright: Playwright) -> None:
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    # insert
    page.goto("http://127.0.0.1:8000/admin/dashboard")
    page.get_by_role("link", name=" Category", exact=True).click()
    page.get_by_role("button", name=" Create New Category").click()
    page.get_by_label("Category Name").click()
    page.get_by_label("Category Name").fill("test-category-1")
    page.locator("div").filter(has_text="New Category Category Name Add New Category").first.click()
    page.get_by_role("button", name=" Add New Category").click()
    time.sleep(5)
    # update
    page.get_by_role("row", name="test-category-1  Update  Delete").get_by_role("button").first.click()
    page.get_by_label("Category Name").click()
    page.get_by_label("Category Name").fill("test-category-1-updated")
    page.get_by_role("button", name=" Update Category").click()
    time.sleep(5)
    # delete
    page.get_by_role("row", name="test-category-1-updated  Update  Delete").get_by_role("button").nth(1).click()

    time.sleep(5)
    # ---------------------
    context.close()
    browser.close()


with sync_playwright() as playwright:
    run(playwright)

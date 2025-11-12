"""
Metadata and Tags Example

This minimal example demonstrates experiment organization with metadata.
Focus is on CLI flags (--name, --tag, --description), not the code.
"""

import random
import time

import yanex

# Get parameters
num_pages = yanex.get_param("num_pages", default=10)
delay_ms = yanex.get_param("delay_ms", default=100)

print(f"Scraping {num_pages} pages with {delay_ms}ms delay...")

# Simulate scraping
total_items = 0
for page in range(num_pages):
    time.sleep(delay_ms / 1000.0)
    items_found = random.randint(5, 20)
    total_items += items_found
    if (page + 1) % 5 == 0:
        print(f"  Processed {page + 1}/{num_pages} pages...")

print(f"Scraping complete! Found {total_items} items total.")

# Log results
yanex.log_metrics(
    {
        "pages_scraped": num_pages,
        "total_items": total_items,
        "items_per_page": total_items / num_pages,
    }
)

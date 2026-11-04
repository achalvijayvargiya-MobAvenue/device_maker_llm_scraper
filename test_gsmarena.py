import asyncio
import json
from app.gsmarena.scraper import GSMArenaScraper

async def main():
    brand = "Xiaomi"
    model = "Redmi Note 7"

    print(f"Searching GSMArena for: {brand} {model}...\n")
    
    async with GSMArenaScraper() as scraper:
        spec = await scraper.lookup(brand=brand, model=model)
        
        if spec:
            print("Device found! Here are the specs:")
            # spec is a pydantic model, we can dump it to dict for a pretty print
            print(json.dumps(spec.model_dump(), indent=4))
        else:
            print("Device not found on GSMArena.")

if __name__ == "__main__":
    asyncio.run(main())

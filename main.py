"""
Review Sentiment & Summary Analyzer — Apify Actor (versione Gemini)
"""

import asyncio
import json
import re

import requests
from apify import Actor

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"


def parse_reviews(reviews_text: str, max_reviews: int) -> list[str]:
    lines = [line.strip() for line in reviews_text.splitlines()]
    reviews = [line for line in lines if line]
    return reviews[:max_reviews]


def build_prompt(reviews: list[str]) -> str:
    numbered_reviews = "\n".join(f"{i + 1}. {r}" for i, r in enumerate(reviews))
    return f"""Analizza queste {len(reviews)} recensioni di clienti. Per ognuna, determina il sentiment e fino a 3 temi ricorrenti brevi.

Poi genera un riassunto complessivo in 2-3 frasi e i 5 temi più ricorrenti in totale.

Rispondi SOLO con un oggetto JSON valido in questo formato esatto, senza testo aggiuntivo prima o dopo:

{{
  "reviews": [
    {{"index": 1, "sentiment": "positivo", "themes": ["tema1", "tema2"]}}
  ],
  "overall_summary": "riassunto in 2-3 frasi",
  "top_themes": ["tema1", "tema2", "tema3", "tema4", "tema5"]
}}

Il campo "sentiment" deve essere sempre uno tra: "positivo", "negativo", "neutro".

Recensioni:
{numbered_reviews}"""


def call_gemini(prompt: str, api_key: str) -> dict:
    response = requests.post(
        GEMINI_API_URL,
        headers={
            "content-type": "application/json",
            "x-goog-api-key": api_key,
        },
        json={"contents": [{"parts": [{"text": prompt}]}]},
        timeout=90,
    )
    response.raise_for_status()
    data = response.json()
    raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()
    return json.loads(cleaned)


async def main() -> None:
    async with Actor:
        actor_input = await Actor.get_input() or {}
        reviews_text = actor_input.get("reviewsText", "")
        max_reviews = int(actor_input.get("maxReviews", 100))
        api_key = actor_input.get("geminiApiKey", "").strip()

        if not api_key:
            await Actor.fail(status_message="Manca la chiave API Gemini nell'input.")
            return

        reviews = parse_reviews(reviews_text, max_reviews)
        if not reviews:
            await Actor.fail(status_message="Il campo recensioni è vuoto.")
            return

        Actor.log.info(f"Analisi di {len(reviews)} recensioni in corso...")
        prompt = build_prompt(reviews)

        try:
            result = call_gemini(prompt, api_key)
        except requests.exceptions.HTTPError as e:
            Actor.log.error(f"Errore API Gemini: {e}")
            await Actor.fail(status_message="Chiave API Gemini non valida o quota esaurita.")
            return
        except json.JSONDecodeError as e:
            Actor.log.error(f"JSON non valido: {e}")
            await Actor.fail(status_message="Risposta AI malformata. Riprova.")
            return
        except Exception as e:
            Actor.log.error(f"Errore imprevisto: {e}")
            await Actor.fail(status_message="Errore imprevisto durante l'analisi AI.")
            return

        analyzed_reviews = result.get("reviews", [])
        if not analyzed_reviews:
            await Actor.push_data({"note": "Nessuna recensione analizzata.", "raw_result": result})
            return

        for item in analyzed_reviews:
            index = item.get("index")
            original_text = reviews[index - 1] if index and 0 < index <= len(reviews) else None
            await Actor.push_data({
                "reviewText": original_text,
                "sentiment": item.get("sentiment"),
                "themes": item.get("themes", []),
            })
            await Actor.charge(event_name="review_analyzed", count=1)

        await Actor.push_data({
            "type": "summary",
            "totalReviewsAnalyzed": len(analyzed_reviews),
            "overallSummary": result.get("overall_summary"),
            "topThemes": result.get("top_themes", []),
        })

        Actor.log.info(f"Analisi completata: {len(analyzed_reviews)} recensioni elaborate.")


if __name__ == "__main__":
    asyncio.run(main())

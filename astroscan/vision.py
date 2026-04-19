"""Vision AI for chart/diagram analysis and fallback OCR.

Uses free OpenRouter models to analyze astrological charts, natal wheels,
aspect grids, and other visual content that pure OCR can't capture.

Tested free models (as of April 2026):
- google/gemma-4-26b-a4b-it:free — Best quality, 262K context
- google/gemma-3-27b-it:free — 27B, 131K context
- nvidia/nemotron-nano-12b-v2-vl:free — Fast, reliable
- google/gemma-3-12b-it:free — Good balance
- google/gemma-3-4b-it:free — Fallback
"""
from __future__ import annotations
from pathlib import Path
import base64
import time
import asyncio
import httpx

from astroscan.config import Config


# ── Prompts ──────────────────────────────────────────────────────────────

OCR_PROMPT = """You are OCR-ing a photographed page from an astrology textbook.
Extract ALL text from this page exactly as written. Preserve:
- Paragraph structure and spacing
- Headings and subheadings (use # markdown headers)
- Lists and bullet points
- Table structures (use markdown tables)
- Italic/bold formatting (use *italic* and **bold**)
- Astrological symbols — write as unicode characters or names in parentheses
- Any footnotes, page numbers, or references

Be extremely accurate. Every word matters. This is a specialized textbook and
the exact terminology must be preserved. Output clean markdown."""

CHART_ANALYSIS_PROMPT = """You are analyzing a page from an astrology textbook. 
Examine this image carefully for ANY charts, diagrams, tables, wheels, grids, 
or visual elements.

For EACH visual element found, provide:

1. **Type**: (natal_wheel / aspect_grid / house_diagram / planet_table / 
   sign_reference / needle_chart / other)
2. **Description**: Extremely detailed description of everything shown — 
   every line, symbol, number, label, arrow, ring, section
3. **Astrological Meaning**: What concept or principle this visual teaches
4. **Elements Identified**: List EVERY zodiac sign, planet, house number, 
   aspect symbol, and any other astrological notation visible
5. **Relationships Shown**: What connections between elements are being 
   demonstrated

Be exhaustive. Every symbol matters. The knowledge from these diagrams 
will be used to build an application, so nothing can be missed.

If the page contains NO visual elements (text only), respond with exactly:
TEXT_ONLY_PAGE"""

KNOWLEDGE_EXTRACTION_PROMPT = """You are building a knowledge base from an astrology textbook page.
This is NOT conventional astrology — it is "fixed astrology" with specific, 
different definitions. Extract knowledge EXACTLY as the book presents it.

From the following page text, extract structured knowledge entries.
For each distinct concept, definition, or rule found, output JSON:

```json
[
  {
    "category": "house|sign|planet|aspect|retrograde|rule|definition|relationship",
    "title": "Short title for this knowledge",
    "content": "Complete explanation from the book",
    "tags": ["relevant", "tags"],
    "is_definition": true/false,
    "is_rule": true/false,
    "related_concepts": ["other concepts mentioned"]
  }
]
```

CRITICAL: Use ONLY what the book says. Do NOT add conventional astrology 
knowledge. If the book says houses ARE signs, that is the truth for this system.

Page text:
{text}"""


class VisionAI:
    """OpenRouter vision AI client with model rotation and rate limiting."""
    
    def __init__(self, config: Config):
        self.api_key = config.openrouter_api_key
        self.models = config.vision_models
        self.rate_limit = config.rate_limit
        self._model_index = 0
        self._last_request_time = 0.0
    
    def _next_model(self) -> str:
        """Rotate to next model for rate limit distribution."""
        model = self.models[self._model_index % len(self.models)]
        self._model_index += 1
        return model
    
    async def _rate_limit_wait(self):
        """Respect rate limits between requests."""
        min_interval = 60.0 / self.rate_limit.get("requests_per_minute", 15)
        elapsed = time.time() - self._last_request_time
        if elapsed < min_interval:
            await asyncio.sleep(min_interval - elapsed)
        self._last_request_time = time.time()
    
    async def _call_vision(self, prompt: str, image_b64: str,
                           preferred_model: str | None = None) -> tuple[str, str]:
        """Call a vision model with automatic retry and model rotation.
        
        Returns:
            Tuple of (response_text, model_used)
        """
        max_retries = self.rate_limit.get("max_retries", 3)
        retry_delay = self.rate_limit.get("retry_delay_seconds", 10)
        
        # Build model list: preferred first, then rotate through others
        models_to_try = []
        if preferred_model:
            models_to_try.append(preferred_model)
        models_to_try.extend([m for m in self.models if m != preferred_model])
        
        async with httpx.AsyncClient(timeout=120) as client:
            for attempt in range(max_retries):
                for model in models_to_try:
                    await self._rate_limit_wait()
                    
                    try:
                        resp = await client.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            headers={
                                "Authorization": f"Bearer {self.api_key}",
                                "Content-Type": "application/json",
                            },
                            json={
                                "model": model,
                                "messages": [{
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": prompt},
                                        {"type": "image_url", "image_url": {
                                            "url": f"data:image/jpeg;base64,{image_b64}"
                                        }},
                                    ],
                                }],
                                "max_tokens": 4096,
                            },
                        )
                        
                        if resp.status_code == 200:
                            text = resp.json()["choices"][0]["message"]["content"]
                            return text, model
                        elif resp.status_code == 429:
                            print(f"    ⚠ Rate limited on {model.split('/')[-1]}, trying next...")
                            continue
                        else:
                            print(f"    ⚠ {model.split('/')[-1]}: HTTP {resp.status_code}")
                            continue
                            
                    except Exception as e:
                        print(f"    ⚠ {model.split('/')[-1]}: {e}")
                        continue
                
                # All models failed this attempt — wait and retry
                if attempt < max_retries - 1:
                    print(f"    ⏳ All models busy, waiting {retry_delay}s (attempt {attempt + 1}/{max_retries})...")
                    await asyncio.sleep(retry_delay)
        
        return "", ""
    
    async def _call_text(self, prompt: str, preferred_model: str | None = None) -> tuple[str, str]:
        """Call a text-only model (no image)."""
        max_retries = self.rate_limit.get("max_retries", 3)
        retry_delay = self.rate_limit.get("retry_delay_seconds", 10)
        
        models_to_try = []
        if preferred_model:
            models_to_try.append(preferred_model)
        models_to_try.extend([m for m in self.models if m != preferred_model])
        
        async with httpx.AsyncClient(timeout=120) as client:
            for attempt in range(max_retries):
                for model in models_to_try:
                    await self._rate_limit_wait()
                    try:
                        resp = await client.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            headers={
                                "Authorization": f"Bearer {self.api_key}",
                                "Content-Type": "application/json",
                            },
                            json={
                                "model": model,
                                "messages": [{"role": "user", "content": prompt}],
                                "max_tokens": 4096,
                            },
                        )
                        if resp.status_code == 200:
                            text = resp.json()["choices"][0]["message"]["content"]
                            return text, model
                        elif resp.status_code == 429:
                            continue
                    except Exception:
                        continue
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
        
        return "", ""
    
    async def extract_text(self, image_path: str | Path) -> tuple[str, str]:
        """Extract text from a page image using vision OCR.
        
        Returns:
            Tuple of (extracted_text, model_used)
        """
        image_bytes = Path(image_path).read_bytes()
        image_b64 = base64.b64encode(image_bytes).decode()
        
        print("  📝 Vision OCR: extracting text...")
        text, model = await self._call_vision(OCR_PROMPT, image_b64)
        
        if text:
            print(f"    ✅ {len(text)} chars extracted ({model.split('/')[-1]})")
        else:
            print("    ❌ Text extraction failed on all models")
        
        return text, model
    
    async def analyze_charts(self, image_path: str | Path) -> tuple[str, str]:
        """Analyze charts/diagrams on a page image.
        
        Returns:
            Tuple of (analysis_text, model_used)
            Empty string if no charts found (TEXT_ONLY_PAGE response).
        """
        image_bytes = Path(image_path).read_bytes()
        image_b64 = base64.b64encode(image_bytes).decode()
        
        print("  👁 Vision AI: analyzing charts/diagrams...")
        analysis, model = await self._call_vision(CHART_ANALYSIS_PROMPT, image_b64)
        
        if "TEXT_ONLY_PAGE" in analysis:
            print("    ℹ No charts/diagrams detected")
            return "", model
        elif analysis:
            print(f"    ✅ Chart analysis: {len(analysis)} chars ({model.split('/')[-1]})")
        else:
            print("    ❌ Chart analysis failed on all models")
        
        return analysis, model
    
    async def extract_knowledge(self, page_text: str, page_number: int) -> tuple[str, str]:
        """Extract structured knowledge entries from page text.
        
        Returns:
            Tuple of (json_string_of_entries, model_used)
        """
        if not page_text or len(page_text) < 50:
            return "[]", ""
        
        prompt = KNOWLEDGE_EXTRACTION_PROMPT.replace("{text}", page_text)
        print("  🧠 Extracting knowledge entries...")
        result, model = await self._call_text(prompt)
        
        if result:
            print(f"    ✅ Knowledge extracted ({model.split('/')[-1]})")
        
        return result, model

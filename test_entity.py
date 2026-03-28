import spacy
from typing import List, Dict, Any

class LightweightEntityExtractor:
    def __init__(self):
        self._entity_nlp = None
        self.enable_entity_awareness = True
        self.entity_stopwords = {"the", "and", "with"}
        # self.entity_label_whitelist = {"DISEASE", "DRUG", "GENE", "CHEMICAL"}
        self.last_entity_metadata: List[Dict[str, Any]] = []

    def _ensure_entity_pipeline(self):
        if not self.enable_entity_awareness:
            return None
        if self._entity_nlp is not None:
            return self._entity_nlp
        try:
            self._entity_nlp = spacy.load("en_ner_bc5cdr_md")  # Make sure this model is installed
            print("Loaded spaCy NER model.")
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.enable_entity_awareness = False
            return None
        return self._entity_nlp

    def _extract_entities(self, text: str, update_last: bool = True) -> List[Dict[str, Any]]:
        if not text or not self.enable_entity_awareness:
            if update_last:
                self.last_entity_metadata = []
            return []
        nlp = self._ensure_entity_pipeline()
        if nlp is None:
            if update_last:
                self.last_entity_metadata = []
            return []
        doc = nlp(text)
        entities: List[Dict[str, Any]] = []
        for ent in doc.ents:
            print(ent)
            label = getattr(ent, "label_", "").upper()
            cleaned_tokens = [tok for tok in ent.text.strip().split() if tok.lower() not in self.entity_stopwords]
            cleaned = " ".join(cleaned_tokens).strip()
            if not cleaned:
                continue
            if cleaned.lower() in self.entity_stopwords or len(cleaned) <= 3:
                continue
            # if self.entity_label_whitelist and label not in self.entity_label_whitelist:
            #     continue
            profile = {
                "surface": cleaned,
                "original": ent.text,
                "start": ent.start_char,
                "end": ent.end_char,
                "label": label,
            }
            entities.append(profile)
        if update_last:
            self.last_entity_metadata = entities
        return entities

    def _augment_query_with_entities(self, text: str) -> str:
        if not text:
            return text
        entities = self._extract_entities(text)
        if not entities:
            return text
        sections = [text]
        descriptions = []
        keywords = []
        for ent in entities:
            parts = [ent["surface"]]
            if ent["label"]:
                parts.append(f"label={ent['label']}")
            descriptions.append("- " + "; ".join(parts))
            keywords.append(ent["surface"])
        if descriptions:
            sections.append("Entity profiles:\n" + "\n".join(descriptions))
        if keywords:
            sections.append("Related keywords: " + ", ".join(sorted(set(keywords))))
        return "\n\n".join(sections)

# ---------------------- TEST CODE ----------------------

if __name__ == "__main__":
    text = "A 3-month-old baby died suddenly at night while asleep. His mother noticed that he had died only after she awoke in the morning. No cause of death was determined based on the autopsy. Which of the following precautions could have prevented the death of the baby? "
    text += """{
  "A": "Placing the infant in a supine position on a firm mattress while sleeping",
  "B": "Keeping the infant covered and maintaining a high room temperature",
  "C": "Application of a device to maintain the sleeping position",
  "D": "Avoiding pacifier use during sleep"
}"""

    extractor = LightweightEntityExtractor()

    print("🧪 Extracted Entities:")
    entities = extractor._extract_entities(text)
    for e in entities:
        print(e)

    print("\n🧠 Augmented Query:")
    augmented = extractor._augment_query_with_entities(text)
    print(augmented)
